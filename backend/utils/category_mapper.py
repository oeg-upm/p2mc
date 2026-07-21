


class CategoryMapper:

    BASE_CATEGORY_URI = "https://w3id.org/p2mc/category"
    

    CATEGORIES = {
        "semantic-matching-model": {
            "@id": f"{BASE_CATEGORY_URI}/semantic-matching",
            "name": "Semantic Matching Model"
        },
        "translation-model": {
            "@id": f"{BASE_CATEGORY_URI}/translation",
            "name": "Translation Model"
        },
        "internal-side-information-inside-kgs": {
            "@id": f"{BASE_CATEGORY_URI}/internal-side-information",
            "name": "Internal side information inside KGs"
        },
        "external-extra-information-outside-kgs": {
            "@id": f"{BASE_CATEGORY_URI}/external-extra-information",
            "name": "External extra information outside KGs"
        },
        "other-kgc-technology": {
            "@id": f"{BASE_CATEGORY_URI}/other",
            "name": "Other KGC technologies"
        }
    }

    def get_category_object(self, extracted_category):
        
        if not extracted_category:
            return self.CATEGORIES["other-kgc-technology"]
            
        # Limpiamos el texto para asegurar que coincida con las claves del diccionario
        clean_key = extracted_category.lower().strip().replace(" ", "-")
        
        # Devolvemos la categoría si existe, o un 'fallback' seguro si el modelo alucinó
        return self.CATEGORIES.get(clean_key, self.CATEGORIES["other-kgc-technology"])