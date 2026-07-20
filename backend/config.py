import os
from pathlib import Path

# direcciones usadas
FILE_DIR = Path(__file__).resolve()
BASE_DIR = FILE_DIR.parent

DATA_DIR = Path(os.getenv("DATA_DIR",str(BASE_DIR / "data"),)).resolve()

OLLAMA_HOST=os.getenv("OLLAMA_HOST")
GROBID_URL=os.getenv("GROBID_URL")

LLAMA_MODEL=os.getenv("LLAMA_MODEL")
QWEN_MODEL=os.getenv("QWEN_MODEL")
GEMMA_MODEL=os.getenv("GEMMA_MODEL")


RABBITMQ_HOST = os.getenv("RABBITMQ_HOST")
RABBITMQ_USER = os.getenv("RABBITMQ_USER")
RABBITMQ_PASSWORD = os.getenv("RABBITMQ_PASSWORD")
QUEUE_NAME = "p2mc_jobs_queue"
RABBITMQ_RETRY_DELAY_SECONDS = 5
RABBITMQ_HEARTBEAT_SECONDS = 43200
RABBITMQ_BLOCKED_CONNECTION_TIMEOUT_SECONDS = 43200
