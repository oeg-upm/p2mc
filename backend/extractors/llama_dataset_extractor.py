import re
import ast
import time
from ollama import Client
import pandas as pd
from io import StringIO
from ..config import LLM_MODEL, OLLAMA_HOST

class LlamaDatasetExtractor:
    def __init__(self, model_name=LLM_MODEL, timeout=600.0, max_retries=3):
        self.model_name = model_name
        self.timeout = timeout
        self.max_retries = max_retries
        self.client = Client(host=OLLAMA_HOST, timeout=self.timeout)
        print(f"Initializing LlamaDatasetExtractor ({self.model_name})...")

        self.kge_entity_prompt = (
            "You extract structured entities from knowledge graph embedding (KGE) paper text and tables.\n"
            "Given the context below, return ONLY a Python dict with exactly these keys:\n"
            "  'model'    -> list of model / method names (e.g. TransE, RotatE, ComplEx)\n"
            "  'dataset'  -> list of benchmark datasets (e.g. FB15k-237, WN18RR, YAGO3-10)\n"
            "  'metric'   -> list of evaluation metrics (e.g. MRR, Hits@1, Hits@10, MR, AUC)\n"
            "Each value must be a list of strings. Use [] for keys with nothing found.\n"
            "Do not include numeric cell values, hyperparameters (learning rate), or column headers alone.\n"
            "Example: {'model': ['TransE'], 'dataset': ['FB15k-237'], 'metric': ['MRR', 'Hits@1']}\n"
        )

    def _parse_dict_response(self, raw: str) -> dict:
        text = (raw or "").strip()
        if not text:
            return {"model": [], "dataset": [], "metric": []}
        

        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            start, end = text.find("{"), text.rfind("}")
            if start >= 0 and end > start:
                try:
                    parsed = ast.literal_eval(text[start : end + 1])
                except (ValueError, SyntaxError):
                    parsed = None
                    
        out = {"model": [], "dataset": [], "metric": []}
        if not isinstance(parsed, dict):
            return out
            

        out["model"] = parsed.get("model", parsed.get("models", []))
        out["dataset"] = parsed.get("dataset", parsed.get("datasets", []))
        out["metric"] = parsed.get("metric", parsed.get("metrics", []))
        

        for key in ("model", "dataset", "metric"):
            val = out[key]
            if isinstance(val, str):
                val = [val] if val.strip() else []
            elif not isinstance(val, list):
                val = []
            out[key] = [str(x).strip() for x in val if str(x).strip()]
  
        return out

    def _clean_entity(self, value: str) -> str:
        s = str(value).strip()
        s = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", s)
        s = re.sub(r"\[[^\]]{1,50}\]", "", s)
        s = re.sub(r"\$([^$]+?)\$", r"\1", s)
        s = re.sub(r"\s+", " ", s).strip(" .,:;-")
        if re.fullmatch(r"[+-]?\d+(?:\.\d+)?", s):
            return ""
        return s if len(s) >= 2 else ""

    def _build_messages(self, text: str, tables: list[str]) -> list[dict]:
        tablas_tsv = ""
        if tables:
            tablas_tsv = self._fix_tables(tables)
            
        contexto_tablas = ""
        if tablas_tsv:
            contexto_tablas = f"\n--- STRUCTURED TABLES CONTEXT ---\n{tablas_tsv}\n"

        prompt = (
            "Task: Read the context inside the <contexto> tags and follow the extraction rules below.\n\n"
            "<contexto>\n"
            f"--- NARRATIVE CONTEXT ---\n{text}\n"
            f"{contexto_tablas}"
            "</contexto>\n\n"
            "Based EXCLUSIVELY on the text and tables provided inside the <contexto> tags above, follow these exact rules:\n"
            f"{self.kge_entity_prompt}\n"
            "NO greetings. NO markdown. Reply ONLY with the valid Python dictionary."
        )
        
        return [
            {"role": "system", "content": "You are a strict information extraction assistant."},
            {"role": "user", "content": prompt}
        ]

    def _chat_kwargs(self) -> dict:
        return {
            "options": {
                "num_ctx": 16384,
                "num_predict": 256,
                "temperature": 0.0,
                "repeat_penalty": 1.0
            }
        }



    def _fix_tables(self,html_tables_list: list) -> str:
        compressed_prompt_text = ""
        table_counter = 1
        
        for html_fragment in html_tables_list:
            if not html_fragment or not html_fragment.strip():
                continue
                
            try:
                dataframes = pd.read_html(StringIO(html_fragment))
                
                for df in dataframes:
                    compressed_prompt_text += f"[Table {table_counter}]:\n"
                    compressed_prompt_text += df.to_csv(index=False, sep="\t") + "\n\n"
                    table_counter += 1
                    
            except ValueError:
                continue
            except Exception as e:
                print(f"Warning: Could not process a table. Error: {e}")
                continue
                
        return compressed_prompt_text



    def extract(self, text: str, tables: list[str] = None) -> list[str]:
        
        if tables is None:
            tables = []
        messages = self._build_messages(text, tables)
        found_datasets = set()
        
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.chat(
                    model=self.model_name,
                    messages=messages,
                    **self._chat_kwargs()
                )
                
                content = (response.get("message") or {}).get("content", "")
                parsed_dict = self._parse_dict_response(content)
                raw_datasets = parsed_dict.get("dataset", [])
                
                for ds in raw_datasets:
                    cleaned_ds = self._clean_entity(ds)
                    if cleaned_ds:
                        found_datasets.add(cleaned_ds)
                
                break 
                
            except Exception as e:
                err_s = str(e).lower()
                retryable = "timed out" in err_s or "timeout" in err_s or "connection" in err_s
                if attempt < self.max_retries and retryable:
                    wait = 2 * (attempt + 1)
                    print(f"[Llama Dataset Extractor] {e} — retry {attempt + 1}/{self.max_retries} in {wait}s")
                    time.sleep(wait)
                    continue
                else:
                    print(f"[Llama Dataset Extractor] error: {e}")
                    break
        return sorted(list(found_datasets))