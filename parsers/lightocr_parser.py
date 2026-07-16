import os
import json
from pathlib import Path

# Importamos la clase original de tu script
from parsers.extract_tables_lightonocr import LightOnOcrTableExtractor

class LightOcrParser:
    def __init__(self, model_id="lightonai/LightOnOCR-2-1B"):
        print("LightOcrParser: Cargando el modelo LightOnOCR (esto puede tardar unos segundos)...")
        
        # Le pasamos "." como directorio falso porque el __init__ original lo exige,
        # pero no lo usaremos para nuestra extracción individual.
        self._extractor = LightOnOcrTableExtractor(pdf_dir=".", model_id=model_id)
        
        # Carga el procesador y el modelo en GPU/CPU
        self._extractor.load_models()
        print("🔌 Modelo LightOnOCR cargado exitosamente.")

    def process(self, pdf_path, json_output_path):
        """
        Procesa un único PDF y guarda el JSON con las tablas extraídas.
        Devuelve la ruta del JSON si es exitoso, o None si hay un error.
        """
        if not pdf_path or not os.path.exists(pdf_path):
            print(f"⚠️ Archivo PDF no encontrado para OCR: {pdf_path}")
            return None

        # Si el JSON ya existe, saltamos la inferencia (que es muy costosa computacionalmente)
        if os.path.exists(json_output_path):
            print(f"✅ El JSON ya existe. Saltando extracción OCR para: {os.path.basename(json_output_path)}")
            return json_output_path

        try:
            # Usamos el método interno de la clase original que procesa 1 solo archivo
            document_data = self._extractor.extract_tables_from_pdf(Path(pdf_path))

            # Lo empaquetamos en el mismo formato estándar que el script original
            final_result = {"documents": [document_data]}

            # Nos aseguramos de que la carpeta de destino existe
            os.makedirs(os.path.dirname(json_output_path), exist_ok=True)

            # Guardamos el resultado en disco
            with open(json_output_path, "w", encoding="utf-8") as f:
                json.dump(final_result, f, indent=2, ensure_ascii=False)

            return json_output_path

        except Exception as e:
            print(f"❌ Error crítico en LightOCR procesando {pdf_path}: {str(e)}")
            return None