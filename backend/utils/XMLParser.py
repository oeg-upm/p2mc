import os
from bs4 import BeautifulSoup
import re

class XMLParser:
    def __init__(self, xml_path):
        self.xml_path = xml_path
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"No se encuentra el archivo: {xml_path}")

        with open(xml_path, 'r', encoding='utf-8') as f:
            # Usamos lxml-xml para manejar el formato TEI correctamente
            self.soup = BeautifulSoup(f, "lxml-xml")

    def get_title(self):
        # 1. Buscamos el bloque que contiene el título oficial del archivo
        title_stmt = self.soup.find('titleStmt')
        
        if title_stmt:
            # 2. Extraemos la etiqueta <title> con los atributos específicos de GROBID
            title_tag = title_stmt.find('title', attrs={'level': 'a', 'type': 'main'})
            
            # 3. Fallback: Si no tiene los atributos exactos, cogemos el primer título que haya
            if not title_tag:
                title_tag = title_stmt.find('title')
                
            if title_tag:
                return title_tag.get_text(strip=True)
                
        return None

    def get_abstract(self):
        """Extrae el contenido del abstract."""
        abstract_tag = self.soup.find('abstract')
        return abstract_tag.get_text(separator=" ", strip=True) if abstract_tag else ""

    def get_full_text(self):
        """Extrae todo el texto del cuerpo del documento."""
        body_tag = self.soup.find('body')
        return body_tag.get_text(separator=" ", strip=True) if body_tag else ""

    def get_sections(self, target_sections=None):
        """
        Extrae texto de secciones específicas.
        :param target_sections: Lista de strings con los nombres de las secciones
                                (ej: ['Introduction', 'Experiments'])
        :return: Diccionario {nombre_seccion: texto}
        """
        results = {}
        body = self.soup.find('body')
        if not body:
            return results

        # En GROBID, las secciones están en <div> y el título en <head>
        divs = body.find_all('div')

        for div in divs:
            head = div.find('head')
            if head:
                section_title = head.get_text(strip=True).lower()

                # Si el usuario no pide secciones específicas, las traemos todas
                # Si pide específicas, filtramos por coincidencia parcial
                if target_sections is None:
                    # Guardamos con el nombre original del XML
                    results[head.get_text(strip=True)] = div.get_text(separator=" ", strip=True)
                else:
                    for target in target_sections:
                        if target.lower() in section_title:
                            results[target] = div.get_text(separator=" ", strip=True)

        return results
    
    def get_arxiv_id(self):
        # Buscamos la etiqueta idno filtrando directamente por su atributo type
        idno_tag = self.soup.find('idno', type='arXiv')
        
        if idno_tag:
            raw_text = idno_tag.get_text(strip=True)
            
            # Aplicamos la expresión regular para capturar el patrón XXXX.XXXX o XXXXX.XXXX
            match = re.search(r'(\d{4,5}\.\d{4,5})', raw_text)
            
            if match:
                return match.group(1)
                
        # Devuelve None si no encuentra la etiqueta o si el texto no coincide con el formato
        return None
    def get_authors(self):
        """
        Extrae la lista de autores principales, ignorando roles o metadatos extra.
        """
        authors_list = []
        
        tei_header = self.soup.find('teiHeader')
        if not tei_header:
            return authors_list
            
        analytic = tei_header.find('analytic')
        if not analytic:
            return authors_list
            
        for author in analytic.find_all('author'):
            pers_name = author.find('persName')
            
            if pers_name:
                name_parts = []
                
                # 1. Buscamos todos los forenames (puede haber type="first" y type="middle")
                for forename in pers_name.find_all('forename'):
                    name_parts.append(forename.get_text(strip=True))
                
                # 2. Buscamos el surname
                surname = pers_name.find('surname')
                if surname:
                    name_parts.append(surname.get_text(strip=True))
                
                # 3. Los unimos con un espacio si encontramos algo
                if name_parts:
                    full_name = " ".join(name_parts)
                    authors_list.append(full_name)
                    
        return authors_list
        




