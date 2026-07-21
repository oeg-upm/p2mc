import torch
from gliner import GLiNER

class GlinerExtractor:
    def __init__(self):
        #self.device = "cpu"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Initializing GLiNER...")
        self.gliner_model = GLiNER.from_pretrained("urchade/gliner_multi-v2.1").to(self.device)
        print(f"GLiNER ready for extraction in %s." %self.device)

    def extract(self, text: str, labels: list):
        entities = self.gliner_model.predict_entities(text, labels, threshold=0.65)
        predictions=list(set(ent['text'] for ent in entities))
        clean_predictions = list(set([element.lower() for element in predictions]))
        return clean_predictions