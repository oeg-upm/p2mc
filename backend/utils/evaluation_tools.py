import re
import unicodedata
import torch
from bert_score import BERTScorer
import transformers

device = "cuda" if torch.cuda.is_available() else "cpu"
transformers.logging.set_verbosity_error()


scorer = BERTScorer(lang="en", device=device)
import evaluate
_bert_scorer = None

ALIAS_MAP = {
    "mrr": "mrr",
    "meanreciprocalrank": "mrr",
    "meanrank": "mr",
    "mr": "mr",
    "hit@1": "hits@1",
    "hits@1": "hits@1",
    "hit@3": "hits@3",
    "hits@3": "hits@3",
    "hit@10": "hits@10",
    "hits@10": "hits@10",
    "hits10": "hits@10",
    "f1": "f1",
    "f1score": "f1",
    "f1measure": "f1",
    "accuracy": "accuracy",
    "acc": "accuracy",
    "auc": "auc",
    "map": "map",
}

# ---------------------------------------------------------
# PASO 1: LIMPIEZA BRUTA (Al salir del LLM)
# ---------------------------------------------------------

def clean_raw_llm_entity(value: str) -> str:
    """
    Se aplica INMEDIATAMENTE después de extraer las cadenas del JSON del LLM.
    Elimina artefactos de LaTeX, citas académicas y números aislados.
    """
    if not value:
        return ""
    
    s = str(value).strip()
    
    # Elimina citas tipo [1], [Smith 2020]
    s = re.sub(r"\[[^\]]{1,50}\]", "", s)
    # Elimina citas tipo (Smith et al. 2020)
    s = re.sub(r"\((?:[^\)]*\d{4}[^\)]*|[^\)]*et\s*al\.?[^\)]*)\)", "", s, flags=re.IGNORECASE)
    
    # Elimina el modo matemático de LaTeX: $Hits@10$ -> Hits@10
    s = re.sub(r"\$([^$]+?)\$", r"\1", s)
    
    # Elimina comandos comunes de LaTeX como \textbf{MRR} -> MRR
    s = re.sub(r"\\(?:textbf|textit|text|mathbf|mathrm|mathit)\{([^{}]*)\}", r"\1", s)
    
    # Elimina subíndices/superíndices de LaTeX: _{xxx} o ^{xxx} -> xxx
    for _ in range(3):
        s = re.sub(r"[_^]\{([^{}]*)\}", r"\1", s)
        
    # Limpia llaves sueltas y compacta espacios
    s = s.replace("{", "").replace("}", "")
    s = re.sub(r"\s+", " ", s).strip(" .,:;-")
    
    # Si lo que ha extraído el LLM es un número puro (ej. "0.95" o "45.2"), lo descartamos
    if re.fullmatch(r"[+-]?\d+(?:\.\d+)?", s):
        return ""
        
    return s if len(s) >= 2 else ""

# ---------------------------------------------------------
# PASO 2: NORMALIZACIÓN (Antes de evaluar Exact Match)
# ---------------------------------------------------------

def normalize_entity(entity: str, is_metric: bool = False) -> str:
    """
    Normaliza entidades (métricas o datasets) para permitir coincidencias exactas.
    Limpia mayúsculas, puntuación, acentos y resuelve alias comunes.
    """
    if not entity:
        return ""
        
    # 1. Minúsculas, quitar acentos y limpieza de espacios en los extremos
    s = str(entity).lower().strip()
    s = "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))
    
    if not is_metric:
        # Lógica específica para DATASETS
        # 2. Eliminamos contenido entre paréntesis (ej. "wn18 (filtered)" -> "wn18")
        s = re.sub(r'\s*\(.*?\)\s*', '', s)
        
    else:
        # Lógica específica para METRICAS
        # Quitamos los paréntesis (pero no el texto de dentro) y los porcentajes
        s = s.replace("%", "").replace("(", "").replace(")", "")
        
    # 3. Eliminamos todos los caracteres de puntuación y espacios internos
    # Esto convierte "ROC-AUC", "ROC AUC", "fb15k-237" en "rocauc" y "fb15k237"
    s = re.sub(r'[-\s_/,.]+', '', s)
    
    # 4. Unificamos la familia "Hits" (hit@10, h@10 -> hits@10)
    # (Esto aplica principalmente a métricas, pero no hace daño aplicarlo a todo)
    s = s.replace("hitsat", "hits@").replace("hitat", "hit@")
    s = re.sub(r'^h@', 'hits@', s)
    s = re.sub(r'^hit@', 'hits@', s)
    
    # 5. Fuerza el formato hits@N si se escapa algo raro (ej. "hits10" -> "hits@10")
    if is_metric:
        m = re.search(r"hits@?(\d+)", s)
        if m:
            s = f"hits@{m.group(1)}"

    # 6. Retorna usando el alias si existe, si no, la cadena normalizada
    return ALIAS_MAP.get(s, s)

def calculate_bertscore_complete(predictions, ground_truth):
    """
    Devuelve la Precisión, el Recall y el F1 semántico usando BERTScore.
    """
    global _bert_scorer
    
    if not predictions or not predictions:
        return 0.0, 0.0, 0.0

    if _bert_scorer is None:
        _bert_scorer = BERTScorer(lang="en", device = device)

    cands = [" ".join(str(normalize_entity(p)) for p in predictions)]
    refs = [" ".join(str(normalize_entity(r)) for r in ground_truth)]

    # ¡Aquí ocurre la magia! Extraemos los tres valores
    P, R, F1 = _bert_scorer.score(cands, refs, verbose=False)
    
    # Devolvemos una tupla con los tres números puros
    return P.item(), R.item(), F1.item()

def aplanar(l):
        flat = []
        for i in l:
            if isinstance(i, list):
                flat.extend(aplanar(i))
            else:
                flat.append(str(i))
        return flat
    
def calculate_exact_match_f1(predictions: list[str], ground_truth: list[str]) -> tuple[float, float, float]:
    """
    Calcula Precision, Recall y F1-score usando Exact Match sobre entidades normalizadas.
    
    Returns:
        tuple: (Precision, Recall, F1)
    """
    # 1. Manejo de listas vacías
    if not predictions and not ground_truth:
        return 1.0, 1.0, 1.0 # Ambos vacíos -> coincidencia perfecta
    if not predictions or not ground_truth:
        return 0.0, 0.0, 0.0 # Uno vacío y el otro no -> fallo total

    # 2. Normalización y conversión a sets (ignora orden y duplicados)
    pred_set = {normalize_entity(p) for p in predictions if normalize_entity(p)}
    gt_set = {normalize_entity(g) for g in ground_truth if normalize_entity(g)}

    # 3. Cálculo de intersecciones
    true_positives = len(pred_set & gt_set)
    false_positives = len(pred_set - gt_set)
    false_negatives = len(gt_set - pred_set)

    # 4. Cálculo de métricas finales (protegido contra división por cero)
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    
    f1_score = 0.0
    if (precision + recall) > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)

    return round(precision, 4), round(recall, 4), round(f1_score, 4)