import os, Path

# direcciones usadas
FILE_DIR = Path(__file__).resolve()
BASE_DIR = FILE_DIR.parent

OLLAMA_HOST=os.getenv("OLLAMA_HOST", "http://localhost:11434")
GROBID_URL=os.getenv("GROBID_URL", "http://localhost:8070")

LLM_MODEL=os.getenv("LLM_MODEL", "llama2")
QWEN_MODEL=os.getenv("QWEN_MODEL", "qwen2.5")
GEMMA_MODEL=os.getenv("GEMMA_MODEL", "gemma4:e4b")