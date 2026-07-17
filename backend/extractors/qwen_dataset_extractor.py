import json
import re
import time
from ollama import Client
import pandas as pd
from io import StringIO
from ..config import QWEN_MODEL, OLLAMA_HOST
class QwenDatasetExtractor:
    def __init__(self, model_name=QWEN_MODEL, timeout=600.0, max_retries=3):
        self.model_name = model_name
        self.timeout = timeout
        self.max_retries = max_retries
        self.client = Client(host=OLLAMA_HOST, timeout=self.timeout)
        
        # Detectamos si es un modelo que usa "thinking" (como DeepSeek-R1 o Qwen3)
        self.is_thinking_model = any(x in model_name.lower() for x in ("qwen3", "qwen2.5", "deepseek-r1"))
        
        print(f"Initializing QwenDatasetExtractor ({self.model_name})...")

    def _clean_entity(self, value: str) -> str:
        """Limpieza adaptada de tu código original para limpiar basura de LaTeX/Citas."""
        if value is None:
            return ""
        s = str(value).strip()
        # Elimina citas [1], fórmulas LaTeX $Hits@10$, etc.
        s = re.sub(r"\[[^\]]{1,50}\]", "", s)
        s = re.sub(r"\$([^$]+?)\$", r"\1", s)
        s = re.sub(r"\s+", " ", s).strip(" .,:;-")
        # Si es solo un número (ej. "0.95"), lo ignoramos
        if re.fullmatch(r"[+-]?\d+(?:\.\d+)?", s):
            return ""
        return s if len(s) >= 2 else ""

    def _build_messages(self, text: str, tables: list[str]) -> list[dict]:
        """
        Filtra y comprime las tablas, y fusiona la narrativa en un prompt estructurado
        usando la técnica de 'sándwich' para evitar que el modelo se pierda.
        """
        # 1. Comprimimos el HTML a TSV usando el método que creamos antes
        tablas_tsv = ""
        if tables:
            tablas_tsv = self._fix_tables(tables)
            
        contexto_tablas = ""
        if tablas_tsv:
            contexto_tablas = f"\n--- STRUCTURED TABLES CONTEXT ---\n{tablas_tsv}\n"

        # 2. Construimos el Sándwich
        prompt = (
            # PAN SUPERIOR: Instrucciones claras
            "Task: Extract ALL datasets used to evaluate (e.g., FB15k, WN18) mentioned in the text and tables.\n"
            "NO greetings. NO explanations. Reply ONLY with a valid JSON dictionary.\n\n"
            
            # RELLENO: Los datos (Narrativa + Tablas en TSV)
            "Here is the context:\n"
            "<contexto>\n"
            f"--- NARRATIVE CONTEXT ---\n{text}\n"
            f"{contexto_tablas}"
            "</contexto>\n\n"
            
            # PAN INFERIOR: Recordatorio final de la tarea
            "Based EXCLUSIVELY on the text and tables provided inside the <contexto> tags above, extract the evaluation metrics.\n"
            "Reply with exactly a Python dictionary containing a single key 'datasets' with a list of strings.\n"
            "Example: {\"datasets\": [\"FB15k\", \"WN18\"]}\n"
            "If no datasets are found, reply exactly: {\"datasets\": []}"
        )
        
        # 3. Retornamos la estructura exacta que pide la API
        return [
            {"role": "system", "content": "You are a strict information extraction assistant."},
            {"role": "user", "content": prompt}
        ]

    def _chat_kwargs(self) -> dict:
        """Configura Ollama, apagando el modo 'think' si es necesario para extraer JSON limpio."""
        kw = {
            "format": "json",
            "options": {
                "num_ctx": 24576,     # Vital para meter tablas HTML
                "num_predict": 256,
                "temperature": 0.0,
                "repeat_penalty": 1.0
            }
        }
        # Si la API de Ollama soporta think=False explícito, se añade aquí
        if self.is_thinking_model:
            kw["options"]["think"] = False 
        return kw

    def _fix_tables(self, tsv_tables_list: list) -> str:
        """Simplemente une las tablas TSV con un encabezado numérico."""
        compressed_prompt_text = ""
        for i, tsv_table in enumerate(tsv_tables_list, start=1):
            if not tsv_table or not tsv_table.strip():
                continue
            compressed_prompt_text += f"[Table {i}]:\n{tsv_table}\n\n"
            
        return compressed_prompt_text

    def extract(self, text: str, tables: list[str] = None) -> list[str]:
        """Método público que orquesta la extracción con reintentos automáticos."""
        if tables is None:
            tables = []# Cambiado a lista para coincidir con la firma y el procesado

        messages = self._build_messages(text, tables)
        found_metrics = set()
        
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.chat(
                    model=self.model_name,
                    messages=messages,
                    **self._chat_kwargs()
                )
                raw_content = (response.get("message") or {}).get("content", "").strip()
                parsed_dict = json.loads(raw_content)
                raw_metrics = parsed_dict.get("datasets", [])
                
                if isinstance(raw_metrics, list):
                    # Pasamos por el filtro de limpieza
                    for m in raw_metrics:
                        clean_m = self._clean_entity(m)
                        if clean_m:
                            found_metrics.add(clean_m)
                    
                    # Si todo ha ido bien, rompemos el bucle de reintentos
                    return sorted(list(found_metrics))
                
            except json.JSONDecodeError:
                print(f"[Qwen] Attempt {attempt + 1}: Bad JSON format.")
            except Exception as e:
                err_s = str(e).lower()
                retryable = "timed out" in err_s or "timeout" in err_s or "connection" in err_s
                if attempt < self.max_retries and retryable:
                    wait_time = 2 * (attempt + 1)
                    print(f"[Qwen] Network error, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"[Qwen] Error: {e}")
                    break
                    
        return sorted(list(found_metrics))