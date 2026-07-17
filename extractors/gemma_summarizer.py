#from ollama import chat
from ollama import Client

class GemmaSummarizer:
    def __init__(self, model_name='gemma4:e4b', timeout=600.0 ):
        self.model_name = model_name
        print(f"Initializing ({self.model_name}) for summarization...")
        self.timeout = timeout
        self.client = Client(timeout=self.timeout)
        print(f"Initialitation of ({self.model_name}) complete.")

    def summarize(self, text):

        question="Please make a summary of everything said about the model introduced in the following text."
        chat = [
            {"role": "system", "content":
                "You are an assistant for QA tasks. Use only provided context."
            },
            {"role": "user", "content": f"Context chunk: {text}"+" Return the summary directly without any introduction."}
        ]

        prompt=(f"Given the following question: {question}")
        chat.append({"role":"user","content":prompt})
        try:
            response=self.client.chat(model='gemma4:e4b', messages=chat)
            predictions=response['message']['content']
        except Exception as e:
            print(f"Timeout error")
            return None
        summary = predictions.strip()
    
        return summary
        

    def get_keywords(self, text):
        question = "Please extract the keywords that best represent this paper. Return ONLY a comma-separated list of keywords. Do not include any introduction, markdown formatting, or bullet points."
        chat = [
            {
                "role": "system", 
                "content": "You are a precise assistant for keyword extraction tasks. Use only the provided context."
            },
            {
                "role": "user", 
                "content": f"Context chunk: {text}"
            },
            {
                "role": "user", 
                "content": question
            }
        ]

        try:
            
            response = self.client.chat(model=self.model_name, messages=chat)
            predictions = response['message']['content']
        except Exception as e:
            print(f"Something went wrong when extracting keywords: {e}")
            return []

        raw_string = predictions.strip()
        keywords_list = [keyword.strip() for keyword in raw_string.split(',') if keyword.strip()]
        
        return keywords_list


    
    
