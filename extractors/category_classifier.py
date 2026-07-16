import os
import json
import joblib
from pathlib import Path
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

from utils.XMLParser import XMLParser

class CategoryClassifier:
    def __init__(self, model_path="../utils/pipeline_clasificador_kge.joblib"):
        self.model_path = Path(model_path)
        self.pipeline = None
        
        self.model_path.parent.mkdir(parents=True, exist_ok=True)

        if self.model_path.exists():
            print(f"Loading classification model from {self.model_path}...")
            self.pipeline = joblib.load(self.model_path)
            print("Classification model ready.")
        else:
            print("Classification model not found. Please execute .train() first.")

    def _create_chunks(self, text, chunk_size=500, overlap=50):
        
        words = text.split()
        return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size - overlap)]

    def train(self, dataset_json_path="../data/dataset_con_rutas_xml.json", xml_base_path="../data/"):
        print("Initiating training...")
        
        with open(dataset_json_path, 'r', encoding='utf-8') as f:
            kge_dataset = json.load(f)

        X_chunks, Y_chunks, paper_ids = [], [], []
        
        for i, paper in enumerate(kge_dataset):
            category = paper.get('category', [])
            xml_path = paper.get('xml_file')
            
            if not xml_path:
                continue
                
            filename = Path(xml_path).name.replace("\\", "/")
            full_xml_path = Path(xml_base_path) / filename

            parser = XMLParser(full_xml_path)
            text = parser.get_full_text()
            
            if text and category:
                fragments = self._create_chunks(text)
                X_chunks.extend(fragments)
                Y_chunks.extend([category] * len(fragments))
                paper_ids.extend([i] * len(fragments))

        X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
            X_chunks, Y_chunks, paper_ids, test_size=0.2, random_state=42, stratify=Y_chunks
        )

        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english', max_features=10000, ngram_range=(1, 2))),
            ('clf', LinearSVC(C=1.0, class_weight='balanced', random_state=42))
        ])

        print("Training TF-IDF + LinearSVC model...")
        self.pipeline.fit(X_train, y_train)

        print("Evaluating model...")
        y_pred_chunks = self.pipeline.predict(X_test)
        
        paper_votes, paper_ground_truth = {}, {}
        for p_id, pred, real in zip(ids_test, y_pred_chunks, y_test):
            paper_votes.setdefault(p_id, []).append(pred)
            paper_ground_truth[p_id] = real
            
        final_paper_preds, final_paper_real = [], []
        for p_id, votes in paper_votes.items():
            voto_ganador = Counter(votes).most_common(1)[0][0]
            final_paper_preds.append(voto_ganador)
            final_paper_real.append(paper_ground_truth[p_id])

        joblib.dump(self.pipeline, self.model_path)
        print(f"Model saved at {self.model_path}")

    def extract(self, text):
        if self.pipeline is None:
            raise ValueError("The classification model has not been loaded. Please check its path or make sure to train it first.")
        
        return self.pipeline.predict([text]).tolist()


"""
with open('../data/dataset_con_rutas_xml.json', 'r', encoding='utf-8') as f:
    kge_dataset=json.load(f)
    
def create_chunks(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def train_and_save_classifier():
    resultados = []
    tiempos = []
    scores_f1 = []
    
    base_path = Path(os.getcwd()).parent
    
    X_chunks=[]
    Y_chunks=[]
    paper_ids=[]
    
    for i,paper in enumerate(kge_dataset):
        category=paper.get('category',[])
        xml_path=paper.get('xml_file')
        if not xml_path:
            continue
        archive=Path(xml_path)
        filename=archive.name
        filename=filename.replace("\\","/")
    
        base_path = Path(os.getcwd()).parent
        xml_path = base_path / "data" / filename
    
        parser=XMLParser(xml_path)
        text=parser.get_full_text()
    
        if text and category:
            fragments=create_chunks(text)
            for frag in fragments:
                X_chunks.append(frag)
                Y_chunks.append(category)
                paper_ids.append(i)
                
    print(f"Generated a total of {len(X_chunks)}")
    
    
    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X_chunks, Y_chunks, paper_ids, test_size=0.2, random_state=42, stratify=Y_chunks
    )
    # 4. Definir el Pipeline
    # Usamos stop_words para limpiar ruido y subimos ngram_range para capturar conceptos compuestos (ej: "machine learning")
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            stop_words='english', 
            max_features=10000, 
            ngram_range=(1, 2)
        )),
        ('clf', LinearSVC(C=1.0, class_weight='balanced', random_state=42))
    ])
    
    
    # 5. Entrenar
    pipeline.fit(X_train, y_train)
    
    y_pred_chunks = pipeline.predict(X_test)
    
    # 2. Agrupamos predicciones por Paper ID
    paper_votes = {}
    paper_ground_truth = {}
    
    for i in range(len(ids_test)):
        p_id = ids_test[i]
        pred = y_pred_chunks[i]
        real = y_test[i]
        
        if p_id not in paper_votes:
            paper_votes[p_id] = []
            paper_ground_truth[p_id] = real
        
        paper_votes[p_id].append(pred)
    
    # 3. Calculamos el ganador (Voto Mayoritario) por paper
    final_paper_preds = []
    final_paper_real = []
    
    for p_id, votes in paper_votes.items():
        voto_ganador = Counter(votes).most_common(1)[0][0]
        final_paper_preds.append(voto_ganador)
        final_paper_real.append(paper_ground_truth[p_id])
    
    path = base_path / "pipeline_clasificador_kge.joblib"
    joblib.dump(pipeline, path)
    return path
    
pipeline_path = train_and_save_classifier()

def extract_classification(text):
    classification_pipeline = joblib.load(pipeline_path)
    predicted_class = classification_pipeline.predict([text]).tolist()
    return predicted_class
"""