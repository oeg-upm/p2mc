import json
import os
from pathlib import Path

from backend.parsers.extract_tables_lightonocr import (
    LightOnOcrTableExtractor,
)


class LightOcrParser:
    def __init__(self, model_id="lightonai/LightOnOCR-2-1B"):
        print("LightOcrParser: loading LightOnOCR model...")

        self._extractor = LightOnOcrTableExtractor(
            pdf_dir=".",
            model_id=model_id,
        )
        self._extractor.load_models()
        print("LightOnOCR model loaded.")

    def process(self, pdf_path, json_output_path):
        """
        Procesa un unico PDF y guarda el JSON con las tablas extraidas.
        """
        if not pdf_path or not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found for OCR: {pdf_path}")

        if os.path.exists(json_output_path):
            print(
                "OCR JSON already exists. Skipping extraction for: "
                f"{os.path.basename(json_output_path)}"
            )
            return json_output_path

        try:
            document_data = self._extractor.extract_tables_from_pdf(
                Path(pdf_path)
            )
            final_result = {"documents": [document_data]}

            os.makedirs(os.path.dirname(json_output_path), exist_ok=True)

            with open(json_output_path, "w", encoding="utf-8") as file:
                json.dump(final_result, file, indent=2, ensure_ascii=False)

            return json_output_path

        except Exception as exc:
            raise RuntimeError(
                f"Critical error processing {pdf_path} with LightOCR: {exc}"
            ) from exc
