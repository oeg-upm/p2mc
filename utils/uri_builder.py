
import hashlib
import re


class UriBuilder():
    BASE_URI = "https://w3id.org/p2mc"

    def _slugify(self, text):
        text = text.lower().strip()
        return re.sub(r'[^a-z0-9]+', '-', text).strip('-')
        
    def _generate_hash(self, text):
        return hashlib.md5(text.lower().strip().encode('utf-8')).hexdigest()[:10]



    def build_modelcard_uri(self, paper_id):

        if not paper_id:
            raise ValueError("Se requiere un paper_id válido para garantizar la persistencia de la URI.")

        return f"{self.BASE_URI}/model/{paper_id}"

    def build_author_uri(self, author_name):
        """Usa el enfoque determinista para autores huérfanos"""
        hash_str = self._generate_hash(author_name)
        # Opcionalmente se puede añadir el apellido al hash para mayor legibilidad
        slug = self._slugify(author_name.split()[-1]) if " " in author_name else self._slugify(author_name)
        return f"{self.BASE_URI}/author/{slug}-{hash_str}"

    def build_dataset_uri(self, dataset_name):
        slug = self._slugify(dataset_name)
        
        # Si por algún motivo el nombre era solo caracteres extraños y el slug queda vacío
        if not slug:
            return f"{self.BASE_URI}/dataset/unknown-dataset"
            
        return f"{self.BASE_URI}/dataset/{slug}"