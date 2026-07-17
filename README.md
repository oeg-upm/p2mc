# ML4KGE ModelCard Extraction Pipeline

Pipeline automatizado para la extracción de metadatos, tablas y narrativas de artículos científicos (PDFs) orientados a Knowledge Graph Embeddings (KGE). El sistema procesa los documentos y genera representaciones estructuradas en formato JSON-LD listas para ser publicadas como ModelCards.

## Arquitectura del pipeline

La arquitectura del pipeline consta de los siguientes pasos:
El input el sistema es una URL a un PDF de un paper y su output un JSON-LD con todos los campos extraídos de ese paper.


1. Se descarga el PDF en la carpeta data/raw.
2. Entran en acción dos herramientas: LightOnOCR y SciPDF. Ambas utilizan PDFs para extraer datos "brutos". En este paso acceden al PDF descargado en el paso 1 y devuelven un fichero cada una, ambos almacenados en data/interim. LightOnOCR devuelve un JSON mientras que SciPDF devuelve un XML.
3. Recibiendo esos dos archivos como input (o partes específicas extraídas de estos) una varidedad de modelos incluidas llms extraen datos "limpios".
4. La clase ModelCardGenerator se encarga de coordinar estos modelos y construir con sus resultados un JSON-LD que es la modelcard terminada.

## Dependencias


  * [Ollama](https://ollama.ai/) ejecutándose en local (Modelos requeridos: `qwen2.5`, `llama3.1`, `gemma4`).
  * Un contenedor corriendo Grobid
  * Se pueden encontrar las librarías necesarias en el requirements.txt

## Estructura del Proyecto

El proyecto sigue la convención estándar de Data Science para separar el código de los datos:

```text
final_pipeline/
├── extractors/              # Todos los extractores del paso 3
├── resources/               # tasks.json y el .joblib encargado de la clasificación de modelos
├── templates/               # Plantillas que sigue el JSON-LD
├── utils/                   # Funciones auxiliares genéricas
├── parsers/                 # Wrappers para SciPDF y LightOCR
├── pdf_handler.py           # Orquestador principal
├── model_card_generation_pipeline.py # Ensamblador del JSON-LD
├── data/
│   ├── raw/                 # PDFs originales sin procesar
│   ├── interim/             # Archivos intermedios (XML de Grobid, JSON de tablas)
│   └── processed/           # ModelCards finales generadas (.json)
└── testing_data/            # Datasets de prueba iniciales
```


## Uso y ejecución

La forma principal de ejecutar el pipeline es instanciando el orquestador y pasándole la URL de un artículo de arXiv:

```python
from pdf_handler import PDFHandler

# 1. Instanciar el orquestador
handler = PDFHandler()

# 2. Ejecutar el pipeline para un paper
# El sistema se saltará los pasos intermedios si los archivos ya existen.
modelcard = pdf_handler.test_handle_pdf("http://arxiv.org/pdf/1802.09691v3.pdf")

```

## ⚠️ Notas sobre Resiliencia

* El pipeline es **idempotente**. Si el proceso falla en el paso 4 (LLMs) tras varios minutos de procesamiento, reiniciar la ejecución NO volverá a procesar el PDF ni el OCR si los archivos ya existen en la carpeta `data/interim/`.
* Los modelos LLM están configurados con un sistema de reintentos automáticos para mitigar caídas de red o timeouts locales con Ollama.