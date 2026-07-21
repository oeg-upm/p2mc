#from utils import llm_preprocessing
import ast
import json
from ollama import Client
from ..config import LLAMA_MODEL, OLLAMA_HOST
class BaseExtractor:
    def __init__(self, model_name):
        self.model_name = model_name
        print(f"Initializing {self.model_name}...")

    def _build_chat(self, text, question):
        chat = [
            {"role": "system", "content": "You are an assistant for QA tasks. Use only provided context."},
            {"role": "user", "content": f"Context chunk: {text}"}
        ]
        
        prompt = (
            f"Given the following question: {question}\n"
            "Return the answer only in a Python list format, i.e. ['A','B']. "
            "You must return an empty list if there is no answer."
        )
        chat.append({"role": "user", "content": prompt})
        
        return chat

    def extract(self, text, question):
        raise NotImplementedError("Las clases hijas deben implementar este método")


class QwenExtractor(BaseExtractor):
    def __init__(self, qwen_model = "Qwen/Qwen3-1.7B"):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        super().__init__(qwen_model)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype="auto",
            device_map="auto"
        )
        self.max_context_tokens = 32768 - 2048

        self.model.generation_config.max_new_tokens = 128
        self.model.generation_config.repetition_penalty = 1.1
        self.model.generation_config.do_sample = False
        
        print(f"{self.model_name} ready for extraction.")

    def extract(self, input_text, question):
        chat = self._build_chat(input_text, question)

        predictions = llm_preprocessing.query_model_return_list(
            self.model, 
            chat, 
            self.tokenizer, 
            local=True
        )
        return predictions



class LlamaExtractor(BaseExtractor):
    def __init__(self, llama_model = LLAMA_MODEL):
        super().__init__(llama_model)
        self.client = Client(host=OLLAMA_HOST, timeout=600.0)
        
        print(f"{self.model_name} ready for extraction.")

        
        

    def _build_chat(self, text, question):
        prompt = (
            f"Context chunk: {text}\n\n"
            f"Given the following question: {question}\n"
            "Return the answer STRICTLY in JSON format. The JSON must contain a single key."
            "with a list of strings. Example: {\"key\": [\"A\", \"B\"]}. "
            "If there is no answer, return {\"key\": []}. Do not output any other text or markdown."
        )
        
        chat = [
            {"role": "system", "content": "You are a precise data extraction assistant."},
            {"role": "user", "content": prompt}
        ]
        
        return chat

    def extract(self, text, question):
        chat = self._build_chat(text, question)
        predictions = []
        
        try:
            response = self.client.chat(
                model=self.model_name, 
                messages=chat,
                format='json',
                options={
                    "num_ctx": 16384,     # <-- NUEVO: Ampliamos la memoria de lectura
                    "num_predict": 256,  # <-- Ampliamos un poco el margen de respuesta
                    "temperature": 0.0,  
                    "repeat_penalty": 1.0 # <-- NUEVO: Lo bajamos a 1.0 (neutral) para no romper URLs
                }
            )
            
            str_predictions = response['message']['content'].strip()
            
            parsed_dict = json.loads(str_predictions)
            
            # Extraemos la lista usando la clave que le pedimos ('datasets')
            parsed_list = parsed_dict.get('key', [])

            if isinstance(parsed_list, list):
                if parsed_list == ['']:
                    predictions = []
                else:
                    predictions = parsed_list
            else:
                print(f"Llama has not returned a list but {type(parsed_list).__name__}")
                
        except json.JSONDecodeError:
            print(f"Llama has not returned a legible JSON. It returned: {str_predictions}")
            
        except Exception as e:
            print(f"Error when connecting with Llama: {e}")

        return predictions
        
