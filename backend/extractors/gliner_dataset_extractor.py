import os
import re
import torch
from gliner2 import GLiNER2

class GlinerDatasetExtractor:
    def __init__(self, gliner_model_name="fastino/gliner2-base-v1"):
        # 1. PyTorch JIT workaround to prevent DeBERTa-v3 crashes
        os.environ.setdefault("PYTORCH_JIT", "0")
        os.environ.setdefault("PYTORCH_NVFUSER_DISABLE", "1")
        
        for name, args in [
            ("_jit_set_profiling_executor", (False,)),
            ("_jit_set_profiling_mode",     (False,)),
            ("_jit_override_can_fuse_on_gpu", (False,)),
            ("_jit_override_can_fuse_on_cpu", (False,)),
            ("_jit_set_texpr_fuser_enabled", (False,)),
            ("_jit_set_nvfuser_enabled",     (False,)),
        ]:
            fn = getattr(torch._C, name, None)
            if fn is not None:
                try:
                    fn(*args)
                except Exception:
                    pass

        print(f"Initializing GlinerDatasetExtractor ({gliner_model_name})...")
        self.model = GLiNER2.from_pretrained(gliner_model_name)
        self.labels = ["dataset"]
        self.min_score = 0.65

        self.label_descriptions = {
            "dataset": "Name of a benchmark dataset or knowledge-graph corpus (e.g. WN18, WN18RR, FB15k, FB15k-237, YAGO, NELL)."
        }
        
        # Expanded blacklist from the original benchmark
        self.dataset_blacklist = frozenset({
            "lmf", "sme", "bern", "gru", "n_e", "n_r", "|e|", "|r|",
            "filter", "raw", "baseline", "berlin", "germany",
            "ent", "iw", "sc", "rw", "transe", "distmult", "complex",
            "rotate", "conve", "tucker", "rescal", "analogy", "simple",
            "hole", "transh", "transr", "transd", "mtransh", "quate",
            "pairre", "crosses", "tntcomplex", "convtranse", "rotate3d"
        })
        print("GlinerDatasetExtractor ready.")

    def _clean_entity(self, value: str) -> str:
        """Light cleanup: strip citation markers, LaTeX math mode and whitespace."""
        s = str(value).strip()
        # Remove citations like [1] or (Smith 2020)
        s = re.sub(r"\[[^\]]{1,50}\]", "", s)
        s = re.sub(r"\((?:[^\)]*\d{4}[^\)]*|[^\)]*et\s*al\.?[^\)]*)\)", "", s, flags=re.IGNORECASE)
        # Remove LaTeX math mode wrappers: $...$ -> ...
        s = re.sub(r"\$([^$]+?)\$", r"\1", s)
        # Remove common LaTeX wrappers
        s = re.sub(r"\\(?:textbf|textit|text|mathbf|mathrm|mathit)\{([^{}]*)\}", r"\1", s)
        # Remove sub/super-scripts
        for _ in range(3):
            s = re.sub(r"[_^]\{([^{}]*)\}", r"\1", s)
        # Clean stray braces and tighten specific notations
        s = s.replace("{", "").replace("}", "")
        s = re.sub(r"(\w)\s+(\+\+|--)(?=\s|$)", r"\1\2", s)
        s = re.sub(r"\s+", " ", s).strip(" .,:;-")
        return s

    def _is_valid_dataset(self, name: str) -> bool:
        """Filters empty names, pure numbers, or blacklisted terms."""
        s = str(name).strip().lower()
        if not s or s in self.dataset_blacklist:
            return False
        # Ignore pure numeric extractions (e.g., "12.5")
        if re.fullmatch(r"[+-]?\d+(?:\.\d+)?", s):
            return False
        return len(s) >= 2

    def _chunk_text(self, text: str, chunk_size=300, overlap=30) -> list[str]:
        """Splits full text into manageable chunks for GLiNER."""
        if not text:
            return []
            
        words = text.split()
        step = chunk_size - overlap
        chunks = []
        
        for i in range(0, len(words), step):
            chunk_words = words[i : i + chunk_size]
            if len(chunk_words) >= 5:
                chunks.append(" ".join(chunk_words))
                
        return chunks

    # --- Public Method ---

    def extract(self, full_text: str) -> list[str]:
        """
        Processes the full paper text and returns a list of detected datasets.
        """
        chunks = self._chunk_text(full_text)
        found_datasets = set()

        for chunk in chunks:
            # 1. Usamos extract_entities y le pasamos las descripciones
            try:
                result = self.model.extract_entities(
                    chunk[:3000], # Límite de seguridad
                    self.label_descriptions,
                    include_confidence=True
                )
            except Exception as e:
                print(f"GLiNER2 chunk error: {e}")
                continue
            
            # 2. GLiNER2 devuelve un diccionario: {"entities": {"dataset": [...]}}
            ents_by_label = (result or {}).get("entities", {}) or {}
            dataset_entities = ents_by_label.get("dataset", [])
            
            for item in dataset_entities:
                if isinstance(item, dict):
                    raw_text = str(item.get("text", ""))
                    score = float(item.get("confidence", 1.0) or 1.0)
                else:
                    raw_text = str(item)
                    score = 1.0
                    
                if score < self.min_score:
                    continue
                    
                cleaned_name = self._clean_entity(raw_text)
                
                if self._is_valid_dataset(cleaned_name):
                    found_datasets.add(cleaned_name)
                    
        return sorted(list(found_datasets))