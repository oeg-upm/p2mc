import requests
import re
import xml.etree.ElementTree as ET


REQUEST_TIMEOUT_SECONDS = 600


class UriFetcher:
    SOA_ENDPOINT = "https://semopenalex.org/sparql"
    LPWC_ENDPOINT = "https://linkedpaperswithcode.com/sparql"
    
    def __init__(self):
        pass

    def _test_lpwc(self):
        try:
            response = requests.get(self.LPWC_ENDPOINT, timeout=5)
            return response.status_code==200

        except requests.exceptions.RequestException:
            return False

    def get_soa_data_from_arxiv_id(self, arxiv_id):
        if self._test_lpwc():
            lpwc_work = self._get_lpwc_work_arxiv_id(arxiv_id)
            if not lpwc_work:
                return None

            soa_work = self._get_soa_uri_from_lpwc(lpwc_work)
            if not soa_work:
                return None
        else:
            doi = self._get_doi_from_arxiv_id(arxiv_id)
            soa_work = self._find_soa_work_from_doi(doi)
            if not soa_work:
                return None
    
        return self._get_soa_data(soa_work)

    def _get_doi_from_arxiv_id(self,arxiv_id):

        try:
            match = re.search(r'(\d{4}\.\d{4,5})', arxiv_id)

            if match:
                clean_id = match.group(1)

                url_doi = f"https://doi.org/10.48550/arxiv.{clean_id}"
                return url_doi
            else:
                return ""

        except Exception as e:
            print("No valid DOI could be found")
            return ""

    def _make_sparql_request(self, endpoint, query):
        headers = {"Accept": "application/sparql-results+json"}
        try:
            response = requests.get(
                endpoint,
                params={"query": query},
                headers=headers,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            data = response.json()
            return data.get("results", {}).get("bindings", [])
        except Exception as e:
            print(f"Connection error on SPARQL request to {endpoint}: {e}")
            return []




    def _get_lpwc_work_arxiv_id(self, arxiv_id):
        query = f"""
        SELECT * WHERE {{
        ?sub ?pred {'"' + arxiv_id + '"'} .
        }} LIMIT 10
        """
        
        bindings = self._make_sparql_request(self.LPWC_ENDPOINT, query)
        if bindings:
            return bindings[0].get("sub", {}).get("value")
        print(f"Failed to retrieve LPWC Work URI through arxiv_id {arxiv_id}")
        return None
        

    def _get_soa_uri_from_lpwc(self, lpwc_work):
        query = f"""
        PREFIX owl: <http://www.w3.org/2002/07/owl#>
        SELECT ?soa WHERE {{
          <{lpwc_work}> owl:sameAs ?soa .
          FILTER(CONTAINS(STR(?soa), "semopenalex.org"))
        }}
        """
        bindings = self._make_sparql_request(self.LPWC_ENDPOINT, query)
        if bindings:
            return bindings[0].get("soa", {}).get("value")
        print(f"Failed to retrieve SOA URI from LPWC work {lpwc_work}")
        return None



    def _get_lpwc_work(self, soa_work):
        query = f"""
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT * WHERE {{
          ?sub ?pred <{soa_work}> .
        }} LIMIT 100
        """
        headers = {
            "Accept": "application/sparql-results+json"
        }
        try:
            response = requests.get(
                self.LPWC_ENDPOINT,
                params={"query": query},
                headers=headers,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
        
            data = response.json()
            lpwc_json_data = data.get("results", {}).get("bindings", [])
            if lpwc_json_data:
                lpwc_work_uri = lpwc_json_data[0].get("sub", {}).get("value")
                return lpwc_work_uri
            else:
                print(f"(Step 2)Failed to retrieve LinkedPapersWithCode Work URI trough SOA Work: {soa_work}")
                return None
        except Exception as e:
            print(f"(Step 2)Connection error when trying to retrieve data from the following page: {soa_work}: {e}")
            return None

    
    def _get_soa_data(self, soa_work):
        query = f"""
        SELECT * WHERE {{
          <{soa_work}> ?pred ?obj .
        }} LIMIT 100
        """
        bindings = self._make_sparql_request(self.SOA_ENDPOINT, query)
        if not bindings:
            print(f"Failed to retrieve SOA data for {soa_work}")
            return None
    
        return {
            "title": self._extract_single_value_by_predicate(bindings, 'http://purl.org/dc/terms/title'),
            "authors": self._extract_values_by_predicate(bindings, 'http://purl.org/dc/terms/creator'),
            "publicationDate": self._extract_single_value_by_predicate(bindings, 'http://prismstandard.org/namespaces/basic/2.0/publicationDate'),
            "soa_uri": soa_work,
        }
    
    def get_author_from_work(self, soa_work, author_name):
        query = f"""
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX dct: <http://purl.org/dc/terms/>
        PREFIX foaf: <http://xmlns.com/foaf/0.1/>
        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
        
        SELECT ?autorID WHERE {{
        <{soa_work}> dct:creator ?autorID .
        ?autorID foaf:name {'"' + author_name + '"'}^^xsd:string.
        }} LIMIT 100
        """
        bindings = self._make_sparql_request(self.SOA_ENDPOINT, query)
        if bindings:
            return bindings[0].get("autorID", {}).get("value")
        print(f"Failed to retrieve author based on {soa_work} and {author_name}")
        return None

    def _find_soa_work_from_doi(self, url_doi):
        query = f"""
        PREFIX dct: <http://purl.org/spar/datacite/>
        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
    
        SELECT ?paper WHERE {{
           ?paper dct:doi "{url_doi}"^^xsd:string .
        }} LIMIT 1
        """

        bindings = self._make_sparql_request(self.SOA_ENDPOINT, query)
        if bindings:
            return bindings[0].get("paper", {}).get("value")
        print(f"Failed to retrieve author based on {url_doi}")
        return None

    def extract_author_uri(self, author_name, arxiv_id):
        if self._test_lpwc():
            lpwc_work = self._get_lpwc_work_arxiv_id(arxiv_id)
            if not lpwc_work:
                return None

            soa_work = self._get_soa_uri_from_lpwc(lpwc_work)
            if not soa_work:
                return None
        else:
            doi=self._get_doi_from_arxiv_id(arxiv_id)
            soa_work = self._find_soa_work_from_doi(doi)
            if not soa_work:
                return None
        return self.get_author_from_work(soa_work, author_name)



#------------------------------------------------------------------------------------------
            

    def _get_lpwc_work_url_abs(self, url_abs):
        query = f"""
        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
        SELECT * WHERE {{
          ?sub ?pred {'"' + url_abs + '"'}^^xsd:anyURI .
        }} LIMIT 100
        """
        bindings = self._make_sparql_request(self.LPWC_ENDPOINT, query)
        if bindings:
            return bindings[0].get("sub", {}).get("value")
        print(f"Failed to retrieve LPWC Work URI through url_abs {url_abs}")
        return None



    def _get_lpwc_data(self, lpwc_work):
        query = f"""
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT * WHERE {{
          <{lpwc_work}> ?pred ?obj .
        }} LIMIT 100
        """
        headers = {
            "Accept": "application/sparql-results+json"
        }
        try:
            response = requests.get(
                self.LPWC_ENDPOINT,
                params={"query": query},
                headers=headers,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
        
            data = response.json()
            lpwc_json_data = data.get("results", {}).get("bindings", [])
            
            
            
            if lpwc_json_data:
                models = self._extract_values_by_predicate(
                    lpwc_json_data, 
                    'https://linkedpaperswithcode.com/property/hasModel'
                )
                
                authors = self._extract_values_by_predicate(
                    lpwc_json_data, 
                    'http://purl.org/dc/terms/creator'
                )
                
                tasks = self._extract_values_by_predicate(
                    lpwc_json_data, 
                    'https://linkedpaperswithcode.com/property/hasTask'
                )
                
                lpwc_data = {
                    "model": models,
                    "authors": authors,
                    "tasks": tasks,
                }
                return lpwc_data
            else:
                print(f"(Step 3)Failed to retrieve LinkedPapersWithCode data from: {lpwc_work}")
                return None
        except Exception as e:
            print(f"(Step 3)Connection error when trying to retrieve data from the following page: {lpwc_work}: {e}")
            return None

    def _extract_values_by_predicate(self, sparql_data, predicate_uri):
        if not sparql_data:
            return []
        return [
            item.get('obj', {}).get('value') 
            for item in sparql_data 
            if item.get('pred', {}).get('value') == predicate_uri
        ]

    def _extract_single_value_by_predicate(self, sparql_data, predicate_uri):
        values = self._extract_values_by_predicate(sparql_data, predicate_uri)
        return values[0] if values else None
    


            
    def extract_dataset_uri(self, dataset_name):
        query = f"""
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT * WHERE {{
            ?sub ?pred {'"' + dataset_name + '"'} .
        }} LIMIT 100
        """
        headers = {
            "Accept": "application/sparql-results+json"
        }
        try:
            response = requests.get(
                self.LPWC_ENDPOINT,
                params={"query": query},
                headers=headers,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
        
            data = response.json()
            dataset_data = data.get("results", {}).get("bindings", [])
            if dataset_data:
                dataset_uri = dataset_data[0].get("sub", {}).get("value")
                return dataset_uri
            else:
                print(f"Failed to retrieve LPWC Dataset URI with the name {dataset_name}")
                return None
        except Exception as e:
            print(f"Connection error when trying to retrieve dataset URI: {dataset_name}: {e}")
            return None

    import re

    def guess_dataset_uri(self, dataset_name):

        # Given that LPWC follows an easily predictable way of naming its URIs for datasets we use the same format to "guess" the URI, which we then check with LPWC
        slug = re.sub(r'[^a-z0-9]+', '-', dataset_name.lower()).strip('-')
    
        predicted_uri = f"https://linkedpaperswithcode.com/dataset/{slug}"

        query = f"ASK {{ <{predicted_uri}> ?p ?o }}"
        headers = {
            "Accept": "application/sparql-results+json"
        }

        try:
            response = requests.get(
                self.LPWC_ENDPOINT,
                params={"query": query},
                headers=headers,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            data = response.json()
        except (
            requests.exceptions.RequestException,
            ValueError,
        ) as e:
            print(
                "Connection error when trying to check guessed dataset URI "
                f"for {dataset_name}: {e}"
            )
            return None

        exists = data.get("boolean", False)
        
        if exists:
            return predicted_uri
        else:
            return None

    import re

    def guess_metric_uri(self, metric_name):

        # Given that LPWC follows an easily predictable way of naming its URIs for datasets we use the same format to "guess" the URI, which we then check with LPWC
        slug = re.sub(r'[^a-z0-9]+', '-', metric_name.lower()).strip('-')

        predicted_uri = f"https://linkedpaperswithcode.com/metric/{slug}"

        query = f"ASK {{ <{predicted_uri}> ?p ?o }}"
        headers = {
            "Accept": "application/sparql-results+json"
        }

        try:
            response = requests.get(
                self.LPWC_ENDPOINT,
                params={"query": query},
                headers=headers,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            data = response.json()
        except (
                requests.exceptions.RequestException,
                ValueError,
        ) as e:
            print(
                "Connection error when trying to check guessed metric URI "
                f"for {metric_name}: {e}"
            )
            return None

        exists = data.get("boolean", False)

        if exists:
            return predicted_uri
        else:
            return None

"""
    def get_soa_data_from_url_abs(self, url_abs):
        lpwc_work = self._get_lpwc_work_url_abs(url_abs)
        if not lpwc_work:
            return None
    
        soa_work = self._get_soa_uri_from_lpwc(lpwc_work)
        if not soa_work:
            return None
    
        return self._get_soa_data(soa_work)

    def fetch_paper_data(self, url_abs):
        lpwc_uri = self._get_lpwc_work_url_abs(url_abs)
        lpwc_data = self._get_lpwc_data(lpwc_uri)
        return {
            #"title": title,
            #"soa_work_uri": soa_uri,
            "lpwc_work_uri": lpwc_uri,
            "lpwc_data": lpwc_data,
        }


    

"""
