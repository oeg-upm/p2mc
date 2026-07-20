from datetime import datetime
import joblib
import json
import copy

from backend import BASE_DIR
# from backend.extractors.gliner_dataset_extractor import GlinerDatasetExtractor
# from backend.extractors.gliner_metric_extractor import GlinerMetricExtractor
# from backend.extractors.qwen_metric_extractor import QwenMetricExtractor
# from backend.extractors.llama_dataset_extractor import LlamaDatasetExtractor
from backend.extractors.qwen_dataset_extractor import QwenDatasetExtractor
# from backend.extractors.gliner_extractor import GlinerExtractor
from backend.extractors.llm_extractors import LlamaExtractor
# from backend.extractors.llm_extractors import QwenExtractor
from backend.extractors.gemma_summarizer import GemmaSummarizer
from backend.utils.category_mapper import CategoryMapper
from backend.utils.task_matcher import TaskMatcher
from backend.utils.uri_builder import UriBuilder
from backend.utils.uri_fetcher import UriFetcher

MODEL_TEMPLATE_FILE = "template_1.jsonld"
IMPLEMENTATION_TEMPLATE_FILE = "implementation_template_1.jsonld"

TAXONOMY_CLASSIFIER_FILE = "taxonomy_classifier.joblib"



class ModelCardGenerator:
    def __init__(self):
        # First we load both templates, starting with the base model template.
        model_template_path = BASE_DIR / "templates" / MODEL_TEMPLATE_FILE
        with open(model_template_path, "r", encoding="utf-8") as f:
            self._model_template = json.load(f)

        implementation_template_path = BASE_DIR / "templates" / IMPLEMENTATION_TEMPLATE_FILE
        with open(implementation_template_path, "r", encoding="utf-8") as f:
            self._implementation_template = json.load(f)

        # Load the classifier (TF-IDF + SVC)
        pipeline_path = BASE_DIR / "resources" / TAXONOMY_CLASSIFIER_FILE
        self._classification_pipeline = joblib.load(pipeline_path)

        # Load the URI fetcher which will help us identify different entities in SemOpenAlex and LinkedPapersWithCode by providing us with URIs.
        # On the other hand the URI builder will help creating URIs for those elements we failed to identify or which need our identification.
        self._uri_fetcher = UriFetcher()
        self._uri_builder = UriBuilder()

        # Load llama, qwen and gliner
        # self._gliner = GlinerExtractor()
        # self._qwen = QwenExtractor()
        self._llama = LlamaExtractor()
        # self._gliner_dataset_extractor = GlinerDatasetExtractor()
        # self._gliner_metric_extractor = GlinerMetricExtractor()
        # self._qwen_metric_extractor = QwenMetricExtractor()
        # self._llama_dataset_extractor = LlamaDatasetExtractor()
        self._qwen_dataset_extractor = QwenDatasetExtractor()
        
        self._summarizer = GemmaSummarizer()
        self._task_matcher = TaskMatcher()
        self._category_mapper = CategoryMapper()

        print("Generator ready!")
        

    def _get_timestamp(self):
        timestamp = datetime.now()
        timestamp_string = timestamp.strftime("%Y-%m-%dT%H:%M:%S")
        return timestamp_string
        
    def _get_tsv_tables(self, raw_tables):
        """Transforma el JSON de tablas directamente a formato TSV (Tab-Separated Values)"""
        tsv_tables = []
        if not raw_tables or not isinstance(raw_tables, dict):
            print("Skipping table extraction: Invalid or empty raw_tables data.")
            return tsv_tables
        documents = raw_tables.get("documents", raw_tables.get("tables", {}).get("documents", []))
        
        for doc in documents:
            for table_dict in doc.get("tables", []):
                columns = table_dict.get("evaluation", {}).get("columns", [])
                rows = table_dict.get("rows", [])
                
                if not columns and not rows:
                    continue

                tsv_lines = []
                
                if columns:
                    tsv_lines.append("\t".join(str(c) for c in columns))
                
                for row in rows:
                    tsv_lines.append("\t".join(str(cell) for cell in row))
                tsv_tables.append("\n".join(tsv_lines))
                
        return tsv_tables

    def _match_tasks(self, tasks):
        clean_tasks_dict = {}
        for task in tasks:
            lpwc_task = self._task_matcher.match_task(task)
            if(lpwc_task):
                task_data = {
                    "@id": lpwc_task["uri"],
                    "name": lpwc_task["name"],
                }
                clean_tasks_dict[lpwc_task["uri"]] = task_data
        clean_extracted_tasks = list(clean_tasks_dict.values())
        return clean_extracted_tasks

    def _extract_reference_publication(self, arxiv_id):
        soa_data = self._uri_fetcher.get_soa_data_from_arxiv_id(arxiv_id)
        if not soa_data:
            return None
        publication_authors = []
        for author in soa_data.get("authors"):
            publication_author = {
                "@type": "Person",
                "@id": author,
            }
            publication_authors.append(publication_author)
        
        reference_publication = {
            "@type": "ScholarlyArticle",
            "@id": arxiv_id,
            "author": publication_authors,
            "schema:sameAs": soa_data.get("soa_uri"),
            "url": f"https://arxiv.org/abs/{arxiv_id}",
        }

        return reference_publication
        
    def _extract_classification(self, text):
        predicted_class = self._classification_pipeline.predict([text]).tolist()
        return predicted_class[0]

    def _order_publication_authors(self, reference_authors, publication_authors):
        right_order = {
            author.get('@id'): i 
            for i, author in enumerate(reference_authors) 
            if author.get('@id') is not None
        }
        publication_authors.sort(key=lambda x: right_order.get(x.get('@id'), float('inf')))
        
        return publication_authors
        
    def _clean_empty_fields(self, d):
        if not isinstance(d, (dict, list)):
            return d
            
        if isinstance(d, list):
            # Limpia los elementos de la lista y quita los que se queden vacíos
            cleaned_list = [self._clean_empty_fields(v) for v in d]
            return [v for v in cleaned_list if v not in (None, "", [], {})]
            
        if isinstance(d, dict):
            # Limpia los valores del diccionario
            cleaned_dict = {k: self._clean_empty_fields(v) for k, v in d.items()}
            # Devuelve el diccionario sin las claves cuyos valores acabaron vacíos
            return {k: v for k, v in cleaned_dict.items() if v not in (None, "", [], {})}
        
    def generate_modelcard(self, extracted_data):
        jsonld = copy.deepcopy(self._model_template)
        
        jsonld["dateCreated"] = self._get_timestamp()

        abstract = extracted_data.get("abstract")
        full_text = extracted_data.get("full_text")
        sections = extracted_data.get("sections")
        tsv_tables = self._get_tsv_tables(extracted_data.get("tables"))
        arxiv_id = extracted_data.get("arxiv_id")

        jsonld["@id"] = self._uri_builder.build_modelcard_uri(arxiv_id)
        
        extracted_names = self._llama.extract(full_text, question = "What is the name of the model presented in this paper?")
        if extracted_names:
            jsonld["name"] = extracted_names[0]
        else:
            jsonld["name"] = "Unknown Model"

            
        jsonld["description"] = self._summarizer.summarize(abstract)
        jsonld["keywords"] = self._summarizer.get_keywords(abstract)

        
        # Dataset extraction and identification
        dataset_context = sections or full_text or abstract or ""
        extracted_datasets = self._qwen_dataset_extractor.extract(
            dataset_context,
            tsv_tables,
        )
        for dataset in extracted_datasets:
            
            dataset_uri = self._uri_fetcher.guess_dataset_uri(dataset)
            if not dataset_uri:
                dataset_uri = self._uri_builder.build_dataset_uri(dataset)

            dataset_data = {
                "@id": dataset_uri,
                "name": dataset,
            }
            jsonld["evaluatedOn"].append(dataset_data)

        extracted_category = self._extract_classification(abstract)
        category_object = self._category_mapper.get_category_object(extracted_category)
        jsonld["modelCategory"] = category_object
        
        # Author identification
        for author in extracted_data.get("authors"):
            author_uri = self._uri_fetcher.extract_author_uri(author, arxiv_id)
            if not author_uri:
                author_uri = self._uri_builder.build_author_uri(author)
            author_data = {
                "@id": author_uri,
                "name": author,
            }
            jsonld["author"].append(author_data)

        # Task extraction and identification
        tasks_prediction = self._llama.extract(abstract, question =  "What are the tasks addressed in this paper?")
        jsonld["mlTask"] = self._match_tasks(tasks_prediction)
        
        extracted_repo = self._llama.extract(full_text, question = "Return the URL link for the paper implementation (like GitHub). Return an empty list if not found. DO NOT  invent links")
        jsonld["codeRepository"] = list(set(extracted_repo))

        reference_publication = self._extract_reference_publication(arxiv_id)
        if(reference_publication):
            jsonld["referencePublication"] = reference_publication

        jsonld["referencePublication"]["author"] = self._order_publication_authors(jsonld["author"], jsonld["referencePublication"]["author"])

        clean_jsonld = self._clean_empty_fields(jsonld)
        return clean_jsonld
        







