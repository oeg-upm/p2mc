import os
import re
import torch
from gliner2 import GLiNER2

class GlinerMetricExtractor:
    def __init__(self, gliner_model_name="fastino/gliner2-base-v1"):
        # PyTorch JIT workaround to prevent DeBERTa-v3 crashes
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

        print(f"Initializing GlinerMetricExtractor ({gliner_model_name})...")
        self.model = GLiNER2.from_pretrained(gliner_model_name)
        
        # 1. We target metrics instead of datasets
        self.labels = ["metric"]
        self.min_score = 0.65
        
        # 2. Specific description to guide GLiNER2
        self.label_descriptions = {
            "metric": "Name of an evaluation metric used to score a model (e.g. MRR, Hits@1, Hits@3, Hits@10, MR, Accuracy, F1)."
        }
        
        # 3. Specific blacklist for metrics based on the original code
        self.metric_blacklist = frozenset({
            "hyperkg", "beta", "filter", "raw", "baseline", "only", 
            "method", "models", "model", "learning rate", "model size", 
            "wd", "n_e", "n_r", "nce baseline", "nce", "rw"
        })
        print("GlinerMetricExtractor ready.")

    def _clean_entity(self, value: str) -> str:
        """Light cleanup: strip citation markers, LaTeX math mode and whitespace."""
        s = str(value).strip()
        s = re.sub(r"\[[^\]]{1,50}\]", "", s)
        s = re.sub(r"\((?:[^\)]*\d{4}[^\)]*|[^\)]*et\s*al\.?[^\)]*)\)", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\$([^$]+?)\$", r"\1", s)
        s = re.sub(r"\\(?:textbf|textit|text|mathbf|mathrm|mathit)\{([^{}]*)\}", r"\1", s)
        for _ in range(3):
            s = re.sub(r"[_^]\{([^{}]*)\}", r"\1", s)
        s = s.replace("{", "").replace("}", "")
        s = re.sub(r"(\w)\s+(\+\+|--)(?=\s|$)", r"\1\2", s)
        s = re.sub(r"\s+", " ", s).strip(" .,:;-")
        return s

    def _is_valid_metric(self, name: str) -> bool:
        """Filters empty names, pure numbers, or blacklisted terms."""
        s = str(name).strip().lower()
        if not s or s in self.metric_blacklist:
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
        Processes the full paper text and returns a list of detected metrics.
        """
        chunks = self._chunk_text(full_text)
        found_metrics = set()

        for chunk in chunks:
            try:
                result = self.model.extract_entities(
                    chunk[:3000], 
                    self.label_descriptions,
                    include_confidence=True
                )
            except Exception as e:
                print(f"GLiNER2 chunk error: {e}")
                continue
            
            ents_by_label = (result or {}).get("entities", {}) or {}
            
            # Extract only the "metric" entities
            metric_entities = ents_by_label.get("metric", [])
            
            for item in metric_entities:
                if isinstance(item, dict):
                    raw_text = str(item.get("text", ""))
                    score = float(item.get("confidence", 1.0) or 1.0)
                else:
                    raw_text = str(item)
                    score = 1.0
                    
                if score < self.min_score:
                    continue
                    
                cleaned_name = self._clean_entity(raw_text)
                
                if self._is_valid_metric(cleaned_name):
                    found_metrics.add(cleaned_name)
                    
        return sorted(list(found_metrics))