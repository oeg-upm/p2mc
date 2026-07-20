import json
import os
from dataclasses import dataclass
from pathlib import Path

from backend.parsers.extract_tables_lightonocr import (
    LightOnOcrTableExtractor,
)


DEFAULT_LIGHTOCR_MODEL_ID = "lightonai/LightOnOCR-2-1B"
DEFAULT_TARGET_LONGEST = 1024
DEFAULT_MAX_NEW_TOKENS = 1024


@dataclass(frozen=True)
class LightOcrSettings:
    model_id: str
    target_longest: int
    max_new_tokens: int


def _read_positive_int_env(name: str, default: int) -> int:
    raw_value = os.getenv(name)

    if raw_value is None:
        return default

    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc

    if value < 1:
        raise ValueError(f"{name} must be greater than 0")

    return value


def get_lightocr_settings() -> LightOcrSettings:
    return LightOcrSettings(
        model_id=os.getenv(
            "P2MC_LIGHTOCR_MODEL_ID",
            DEFAULT_LIGHTOCR_MODEL_ID,
        ),
        target_longest=_read_positive_int_env(
            "P2MC_LIGHTOCR_TARGET_LONGEST",
            DEFAULT_TARGET_LONGEST,
        ),
        max_new_tokens=_read_positive_int_env(
            "P2MC_LIGHTOCR_MAX_NEW_TOKENS",
            DEFAULT_MAX_NEW_TOKENS,
        ),
    )


class LightOcrParser:
    def __init__(self, model_id: str | None = None):
        settings = get_lightocr_settings()
        model_id = model_id or settings.model_id

        print("LightOcrParser: loading LightOnOCR model...")
        print(
            "LightOcrParser: "
            f"model={model_id}, "
            f"target_longest={settings.target_longest}, "
            f"max_new_tokens={settings.max_new_tokens}"
        )

        self._extractor = LightOnOcrTableExtractor(
            pdf_dir=".",
            model_id=model_id,
            target_longest=settings.target_longest,
            max_new_tokens=settings.max_new_tokens,
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
