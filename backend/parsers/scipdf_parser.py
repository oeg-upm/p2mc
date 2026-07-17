import os
import requests

class SciPdfParser:
    def __init__(self, grobid_url="http://localhost:8070"):
        """
        Inicializa el parser y verifica la conexión con el servidor GROBID
        al instanciar la clase.
        """
        print("SciPdfParser: Inicializando y verificando conexión con GROBID...")
        self.grobid_url = grobid_url
        self._check_connection()

    def _check_connection(self):
        """Verifica si el servidor GROBID está activo."""
        try:
            response = requests.get(f"{self.grobid_url}/api/isalive")
            if response.status_code != 200:
                print(f"⚠️ Advertencia: GROBID en {self.grobid_url} no devolvió un estado 200.")
            else:
                print(f"🔌 Conectado exitosamente a GROBID en {self.grobid_url}")
        except requests.exceptions.ConnectionError:
            print(f"❌ Error crítico: No se pudo conectar a GROBID en {self.grobid_url}. ¿Está la imagen Docker corriendo?")

    def process(self, pdf_path, xml_output_path):
        """
        Procesa un único PDF usando el backend de SciPDF (GROBID) 
        y guarda el XML resultante.
        
        Devuelve la ruta del XML si es exitoso, o None si hay un error.
        """
        if not pdf_path or not os.path.exists(pdf_path):
            print(f"⚠️ Archivo PDF no encontrado: {pdf_path}")
            return None

        # Si el archivo XML ya existe de una ejecución previa, saltamos el procesamiento
        if os.path.exists(xml_output_path):
            print(f"✅ El XML ya existe. Saltando extracción para: {os.path.basename(xml_output_path)}")
            return xml_output_path

        endpoint = f"{self.grobid_url}/api/processFulltextDocument"
        
        try:
            with open(pdf_path, 'rb') as f:
                files = {'input': f}
                # Opciones de extracción habituales de SciPDF
                data = {
                    'generateTeiIds': '1',
                    'consolidateHeader': '1',
                    'consolidateConclusion': '1'
                }
                
                res = requests.post(endpoint, files=files, data=data, timeout=120)

            if res.status_code == 200:
                # Nos aseguramos de que el directorio destino existe
                os.makedirs(os.path.dirname(xml_output_path), exist_ok=True)
                
                with open(xml_output_path, 'w', encoding='utf-8') as xml_file:
                    xml_file.write(res.text)
                
                return xml_output_path
            else:
                print(f"⚠️ Error al procesar {pdf_path}: Código HTTP {res.status_code}")
                return None

        except Exception as e:
            print(f"⚠️ Error crítico procesando {pdf_path}: {str(e)}")
            return None