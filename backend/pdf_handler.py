import json
import re
from pathlib import Path
import requests
import traceback
from model_card_generation_pipeline import ModelCardGenerator
from utils.XMLParser import XMLParser

from parsers.scipdf_parser import SciPdfParser
from parsers.lightocr_parser import LightOcrParser


class PDFHandler:
    def __init__(self):
        self._pdf_dir = Path("data") / "raw" / "pdfs"
        self._xml_dir = Path("data") / "interim" / "scipdf_xml"
        self._json_dir = Path("data") / "interim" / "lightocr_json"
        self._modelcards_dir = Path("data") / "processed" / "modelcards"

        self._scipdf_parser = SciPdfParser()
        self._lightocr_parser = LightOcrParser()

        self._mcg = ModelCardGenerator()
        


    def _extract_id_from_url(self, url):
        match = re.search(r"(\d{4}\.\d{4,5})", url)
        if match:
            return match.group(1)
        print(f"Failed to parse arXiv ID from {url}")
        return None
    
    def _download_pdf(self, pdf_url, save_path):
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            response = requests.get(pdf_url, stream=True, timeout=30)
            response.raise_for_status()
            with open(save_path, 'wb') as pdf_file:
                for chunk in response.iter_content(chunk_size=8192):
                    pdf_file.write(chunk)
            return True
        except requests.exceptions.RequestException as e:
            print(f"Something went wrong when trying to download {pdf_url}: {e}")
            return False

    def _process_with_scipdf(self, pdf_path, xml_save_path):
        self._scipdf_parser.process(pdf_path, xml_save_path)

    def _process_with_lightocr(self, pdf_path, json_save_path):
        self._lightocr_parser.process(pdf_path, json_save_path)

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
            print(f"OCR JSON not found at {json_path}. Omitting tables...")
        
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
            paper_id = self._extract_id_from_url(pdf_url)
            if not paper_id:
                print(f"Aborting process. Invalid URL: {pdf_url}")
                return {"error": "Invalid URL or missing arXiv ID"}
            
            pdf_path = self._pdf_dir / f"{paper_id}.pdf"
            xml_path = self._xml_dir / f"{paper_id}.xml"
            json_path = self._json_dir / f"{paper_id}.json"
            modelcard_path = self._modelcards_dir / f"{paper_id}_modelcard.json"
    
            print(f"Downloading PDF from {pdf_url} to {pdf_path}...")
            self._download_pdf(pdf_url, pdf_path)
            print("Processing with sciPDF...")
            self._process_with_scipdf(pdf_path, xml_path)
            print("Processing with LightOnOCR...")
            self._process_with_lightocr(pdf_path, json_path)
            print("Generating modelcard...")
            extracted_data = self._extract_values(xml_path, json_path, paper_id)
            modelcard = self._mcg.generate_modelcard(extracted_data)
            
            print(f"Saving final modelcard to {modelcard_path}...")
            self._modelcards_dir.mkdir(parents=True, exist_ok=True)
            with open(modelcard_path, 'w', encoding='utf-8') as f:
                json.dump(modelcard, f, indent=4, ensure_ascii=False)
                
            return modelcard
            
        except Exception as e:
            print(f"\n Something went wrong when handling {pdf_url}:")
            print(traceback.format_exc())
            return {"error": str(e), "failed_url": pdf_url}

    def test_handle_pdf(self, pdf_url):

        try:
            paper_id = self._extract_id_from_url(pdf_url)
            if not paper_id:
                print(f"Aborting process. Invalid URL: {pdf_url}")
                return {"error": "Invalid URL or missing arXiv ID"}
            
            pdf_path = self._pdf_dir / f"{paper_id}.pdf"
            xml_path = Path("testing_data") / "xml_outputs" / f"{paper_id}.tei.xml"
            json_path = self._json_dir / f"{paper_id}.json"
            modelcard_path = self._modelcards_dir / f"{paper_id}_modelcard.json"
    
            print(f"Downloading PDF from {pdf_url} to {pdf_path}...")
            self._download_pdf(pdf_url, pdf_path)
            print("Processing with sciPDF...")
            self._process_with_scipdf(pdf_path, xml_path)
            print("Processing with LightOnOCR...")
            self._process_with_lightocr(pdf_path, json_path)
            print("Generating modelcard...")
            extracted_data = self._extract_values(xml_path, json_path, paper_id)

            modelcard = self._mcg.generate_modelcard(extracted_data)

            print(f"Saving final modelcard to {modelcard_path}...")
            self._modelcards_dir.mkdir(parents=True, exist_ok=True)
            with open(modelcard_path, 'w', encoding='utf-8') as f:
                json.dump(modelcard, f, indent=4, ensure_ascii=False)
                
            return modelcard
            
        except Exception as e:
            print(f"\Something went wrong when handling {pdf_url}:")
            print(traceback.format_exc())
            return {"error": str(e), "failed_url": pdf_url}









        