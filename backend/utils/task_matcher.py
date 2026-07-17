import json
from rapidfuzz import process, fuzz

# TaskMatcher recieves the name of a task found in the text and tries to relate it to a task from the list found at testing_data/tasks.json which contains the LPWC tasks and their URIs.
# If a match is found then it returns the LPWC task.

class TaskMatcher:
    def __init__(self, json_path="resources/tasks.json"):
        with open(json_path, 'r', encoding='utf-8') as f:
            raw_tasks = json.load(f)
            
        self.master_tasks = {
            task["name"].lower().strip(): {"name": task["name"], "uri": task["uri"]}
            for task in raw_tasks
        }

    def match_task(self, extracted_task):
        if not extracted_task:
            return None
            
        clean_task = extracted_task.lower().strip()
        
        
        if clean_task in self.master_tasks:
            return self.master_tasks[clean_task]
            
        
        best_match, score, _ = process.extractOne(
            clean_task, 
            self.master_tasks.keys(), 
            scorer=fuzz.token_set_ratio
        )
        
        if score > 85:
            return self.master_tasks[best_match]
            
        return None