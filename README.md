<p align="center">
  <img src="https://github.com/oeg-upm/p2mc/blob/main/resources/figures/p2mc_logo.png" width="300" height="300" margin="auto">
</p>

Automated pipeline for extracting metadata, tables, and narratives from scientific papers (PDFs) oriented towards Knowledge Graph Embeddings (KGE). The system processes documents and generates structured representations in JSON-LD format, ready to be published as ModelCards.

# Pipeline Architecture
![P2MC Workflow](https://github.com/oeg-upm/p2mc/blob/main/resources/figures/workflow.png)
The pipeline architecture, as showin in the figure, consists of the following steps:
1. The PDF file is located and downloaded into the data/raw folder. Two tools are employed: LightOnOCR and SciPDF. Both use PDFs to extract "raw" data. In this step, they access the PDF downloaded in step 1 and return a file each, both stored in data/interim. LightOnOCR returns a JSON, while SciPDF returns an XML.
2. Receiving those two files as input (or specific parts extracted from them), a variety of models, including LLMs, extract "clean" data. The ModelCardGenerator class coordinates these models and builds the finished JSON-LD ModelCard using their results.
3. The URIs of the different elements contained in the ModelCard, when possible, are extracted from existing Knowledge Graphs, namely, from [LPWC](https://linkedpaperswithcode.com/resource/LPWC) and [SemOpenAlex](https://semopenalex.org/resource/semopenalex:UniversalSearch)

# Dependencies
Besides the required packages specified in requirements.txt, the following dependencies are needed to execute the pipeline:
1. Ollama running locally (Required models: qwen2.5, llama3.1, gemma4).
2. A Docker container running Grobid (docker pull lfoppiano/grobid:${latest_grobid_version}-full) .

# Project Structure

The project follows the standard Data Science convention for separating code from data:
<code>
final_pipeline/
├── extractors/              # All extractors from step 3
├── resources/               # tasks.json and the .joblib file for model classification
├── templates/               # Templates followed by the JSON-LD
├── utils/                   # Generic helper functions
├── parsers/                 # Wrappers for SciPDF and LightOCR
├── pdf_handler.py           # Main orchestrator
├── model_card_generation_pipeline.py # JSON-LD assembler
├── data/
│   ├── raw/                 # Original, unprocessed PDFs
│   ├── interim/             # Intermediate files (Grobid XML, table JSONs)
│   └── processed/           # Final generated ModelCards (.json)
└── testing_data/            # Initial test datasets
</code>

# Usage and Execution

The primary way to run the pipeline is by instantiating the orchestrator and passing it the arXiv URL of a paper:

```python
from pdf_handler import PDFHandler

# 1. Instantiate the orchestrator
handler = PDFHandler()

# 2. Run the pipeline for a paper
# The system will skip intermediate steps if the files already exist.
modelcard = pdf_handler.test_handle_pdf("http://arxiv.org/pdf/1802.09691v3.pdf")
```

# ⚠️ Notes on Resilience

The pipeline is idempotent. If the process fails at step 4 (LLMs) after several minutes of processing, restarting the execution will NOT re-process the PDF or the OCR if the files already exist in the data/interim/ folder.

The LLM models are configured with an automatic retry system to mitigate network issues or local timeouts with Ollama.

