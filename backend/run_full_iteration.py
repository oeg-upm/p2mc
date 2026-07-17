import os
import sys
import time
import json
from pathlib import Path
import logging
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

if not os.getenv("HF_TOKEN"):
    raise RuntimeError(f"No se ha encontrado HF_TOKEN en {PROJECT_ROOT / '.env'}")

print("HF_TOKEN cargado correctamente")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
from pdf_handler import PDFHandler


PDF_URL = "https://arxiv.org/pdf/1802.04394.pdf"
PAPER_ID = "1802.04394"

ARTIFACTS = {
    "PDF": Path("data/raw/pdfs") / f"{PAPER_ID}.pdf",
    "GROBID XML": Path("data/interim/scipdf_xml") / f"{PAPER_ID}.xml",
    "LightOCR JSON": Path("data/interim/lightocr_json") / f"{PAPER_ID}.json",
    "ModelCard": Path("data/processed/modelcards")
    / f"{PAPER_ID}_modelcard.json",
}


def show_artifacts() -> None:
    print("\n" + "=" * 80)
    print("ARTEFACTOS")
    print("=" * 80)

    for name, path in ARTIFACTS.items():
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"[OK] {name:<15} {path} ({size_mb:.2f} MB)")
        else:
            print(f"[--] {name:<15} {path} no existe")


def main() -> int:
    started = time.perf_counter()

    print("=" * 80)
    print("P2MC — EJECUCIÓN COMPLETA")
    print("=" * 80)
    print(f"Paper: {PAPER_ID}")
    print(f"URL:   {PDF_URL}")
    print(f"Directorio de trabajo: {Path.cwd()}")
    print()

    print("[1] Inicializando PDFHandler y cargando modelos...")
    handler = PDFHandler()

    print("\n[2] Ejecutando el pipeline completo...")
    result = handler.handle_pdf(PDF_URL)

    elapsed = time.perf_counter() - started

    show_artifacts()

    print("\n" + "=" * 80)
    print("RESULTADO")
    print("=" * 80)

    if not isinstance(result, dict):
        print(f"[ERROR] Resultado inesperado: {type(result).__name__}")
        return 1

    if result.get("error"):
        print("[ERROR] La ejecución no terminó correctamente.")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"\nDuración total: {elapsed:.2f} segundos")
        return 1

    print("[OK] ModelCard generado correctamente.")
    print(f"Campos principales: {list(result.keys())}")
    print(f"Nombre detectado: {result.get('name')}")
    print(f"Duración total: {elapsed:.2f} segundos")

    output_path = ARTIFACTS["ModelCard"]
    print(f"Resultado guardado en: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
