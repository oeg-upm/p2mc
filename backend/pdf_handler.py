import json
import re
from collections.abc import Callable
from pathlib import Path
import requests
import traceback
from backend.model_card_generation_pipeline import ModelCardGenerator
from backend.utils.XMLParser import XMLParser
from backend.parsers.scipdf_parser import SciPdfParser
from backend.parsers.lightocr_parser import LightOcrParser

from backend import DATA_DIR

REQUEST_TIMEOUT_SECONDS = 600


class PDFHandler:
    def __init__(self, logger: Callable[[str], None] | None = None):
        self._log = logger or self._default_log

        self._pdf_dir = DATA_DIR / "raw" / "pdfs"
        self._xml_dir = DATA_DIR / "interim" / "scipdf_xml"
        self._json_dir = DATA_DIR / "interim" / "lightocr_json"
        self._modelcards_dir = DATA_DIR / "processed" / "modelcards"

        self._log("PDFHandler: initializing SciPdfParser")
        self._scipdf_parser = SciPdfParser()
        self._log("PDFHandler: SciPdfParser ready")

        self._log("PDFHandler: initializing LightOcrParser")
        self._lightocr_parser = LightOcrParser()
        self._log("PDFHandler: LightOcrParser ready")

        self._log("PDFHandler: initializing ModelCardGenerator")
        self._mcg = ModelCardGenerator()
        self._log("PDFHandler: ModelCardGenerator ready")
        

    @staticmethod
    def _default_log(message: str) -> None:
        print(message, flush=True)


    def _extract_id_from_url(self, url):
        match = re.search(r"(\d{4}\.\d{4,5})", url)
        if match:
            return match.group(1)
        raise ValueError(f"Failed to parse arXiv ID from {url}")
    
    def _download_pdf(self, pdf_url, save_path):
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            response = requests.get(
                pdf_url,
                stream=True,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            with open(save_path, 'wb') as pdf_file:
                for chunk in response.iter_content(chunk_size=8192):
                    pdf_file.write(chunk)
            return save_path
        except requests.exceptions.RequestException as e:
            raise RuntimeError(
                f"Something went wrong when trying to download {pdf_url}: {e}"
            ) from e

    def _process_with_scipdf(self, pdf_path, xml_save_path):
        result = self._scipdf_parser.process(pdf_path, xml_save_path)
        if not result or not Path(xml_save_path).is_file():
            raise RuntimeError(f"SciPDF did not generate XML at {xml_save_path}")
        return result

    def _process_with_lightocr(self, pdf_path, json_save_path):
        result = self._lightocr_parser.process(pdf_path, json_save_path)
        if not result or not Path(json_save_path).is_file():
            raise RuntimeError(f"LightOCR did not generate JSON at {json_save_path}")
        return result

    def _extract_values(self, xml_path, json_path, arxiv_id):
        if not xml_path.exists():
            raise FileNotFoundError(f"SciPDF XML not found at {xml_path}. Unable to extract values from it.")
            
        self._xml_parser = XMLParser(xml_path)
        title = self._xml_parser.get_title()
        full_text = self._xml_parser.get_full_text()
        abstract = self._xml_parser.get_abstract()
        authors = self._xml_parser.get_authors()
        section_dict = self._xml_parser.get_sections(target_sections=['Experiments','Evaluation','Results'])
        sections = "\n\n".join(section_dict.values())
        
        tables = []
        if json_path.exists():
            with open(json_path, 'r', encoding='utf-8') as f:
                tables = json.load(f)
        else:
            self._log(
                f"PDFHandler: OCR JSON not found at {json_path}. "
                "Omitting tables."
            )
        
        extracted_data = {
            "title": title,
            "arxiv_id": arxiv_id,
            "authors": authors,
            "abstract": abstract,
            "full_text": full_text,
            "sections": sections,
            "tables": tables,
        }

        
        return extracted_data
        
    def handle_pdf(self, pdf_url):
        try:
            self._log(f"PDFHandler: starting PDF handling for {pdf_url}")
            paper_id = self._extract_id_from_url(pdf_url)
            if not paper_id:
                self._log(f"PDFHandler: aborting process. Invalid URL: {pdf_url}")
                return {"error": "Invalid URL or missing arXiv ID"}
            self._log(f"PDFHandler: parsed arXiv id {paper_id}")
            
            pdf_path = self._pdf_dir / f"{paper_id}.pdf"
            xml_path = self._xml_dir / f"{paper_id}.xml"
            json_path = self._json_dir / f"{paper_id}.json"
            modelcard_path = self._modelcards_dir / f"{paper_id}_modelcard.json"
    
            self._log(f"PDFHandler: downloading PDF to {pdf_path}")
            self._download_pdf(pdf_url, pdf_path)
            self._log(f"PDFHandler: PDF downloaded at {pdf_path}")

            self._log(f"PDFHandler: processing with SciPDF into {xml_path}")
            self._process_with_scipdf(pdf_path, xml_path)
            self._log(f"PDFHandler: SciPDF XML ready at {xml_path}")

            self._log(f"PDFHandler: processing with LightOnOCR into {json_path}")
            self._process_with_lightocr(pdf_path, json_path)
            self._log(f"PDFHandler: LightOCR JSON ready at {json_path}")

            self._log("PDFHandler: extracting values from generated artifacts")
            extracted_data = self._extract_values(xml_path, json_path, paper_id)
            self._log("PDFHandler: extracted values ready")

            self._log("PDFHandler: generating modelcard")
            modelcard = self._mcg.generate_modelcard(extracted_data)
            self._log("PDFHandler: modelcard generated")
            
            self._log(f"PDFHandler: saving final modelcard to {modelcard_path}")
            self._modelcards_dir.mkdir(parents=True, exist_ok=True)
            with open(modelcard_path, 'w', encoding='utf-8') as f:
                json.dump(modelcard, f, indent=4, ensure_ascii=False)
            self._log(f"PDFHandler: modelcard saved at {modelcard_path}")
                
            return modelcard
            
        except Exception as e:
            self._log(f"PDFHandler: something went wrong when handling {pdf_url}")
            self._log(traceback.format_exc())
            return {"error": str(e), "failed_url": pdf_url}

    def test_handle_pdf(self, pdf_url):

        try:
            paper_id = self._extract_id_from_url(pdf_url)
            if not paper_id:
                self._log(f"PDFHandler: aborting process. Invalid URL: {pdf_url}")
                return {"error": "Invalid URL or missing arXiv ID"}
            
            pdf_path = self._pdf_dir / f"{paper_id}.pdf"
            xml_path = Path("testing_data") / "xml_outputs" / f"{paper_id}.tei.xml"
            json_path = self._json_dir / f"{paper_id}.json"
            modelcard_path = self._modelcards_dir / f"{paper_id}_modelcard.json"
    
            self._log(f"PDFHandler: downloading PDF to {pdf_path}")
            self._download_pdf(pdf_url, pdf_path)
            self._log(f"PDFHandler: processing with SciPDF into {xml_path}")
            self._process_with_scipdf(pdf_path, xml_path)
            self._log(f"PDFHandler: processing with LightOnOCR into {json_path}")
            self._process_with_lightocr(pdf_path, json_path)
            self._log("PDFHandler: generating modelcard")
            extracted_data = self._extract_values(xml_path, json_path, paper_id)

            modelcard = self._mcg.generate_modelcard(extracted_data)

            self._log(f"PDFHandler: saving final modelcard to {modelcard_path}")
            self._modelcards_dir.mkdir(parents=True, exist_ok=True)
            with open(modelcard_path, 'w', encoding='utf-8') as f:
                json.dump(modelcard, f, indent=4, ensure_ascii=False)
                
            return modelcard
            
        except Exception as e:
            self._log(f"PDFHandler: something went wrong when handling {pdf_url}")
            self._log(traceback.format_exc())
            return {"error": str(e), "failed_url": pdf_url}









        
