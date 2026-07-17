import streamlit as st
from pathlib import Path
from PIL import Image, ImageOps


if "selected_example" not in st.session_state:
    st.session_state.selected_example = None

if "selected_json_path" not in st.session_state:
    st.session_state.selected_json_path = None

BASE_DIR = Path(__file__).parent.parent
ASSETS_DIR = BASE_DIR / "assets"

paper_pink = "#FF8585"
pipeline_blue = "#1F81BF"
modelcard_orange = "#FFBD59"

st.set_page_config(
    page_title="P2MC Demo",
    page_icon="📄",
    layout="centered"
)

col_izq, col_centro, col_der = st.columns([1, 2, 1])

with col_centro:
    st.image(str(ASSETS_DIR / "logo.png"), width='stretch')


st.title(
    f":color[Papers]{{foreground='{paper_pink}'}} "
    f":color[2]{{foreground='{pipeline_blue}'}} "
    f":color[Modelcards]{{foreground='{modelcard_orange}'}}", 
    text_alignment="center", 
    anchor=False
)

# Este es el contenedor para los autores.
author_container = st.container(horizontal=True, horizontal_alignment = "center")
author_container.write("Mateo Campaya Pinto")
author_container.write("Elvira Amador Dominguez")
author_container.write("Erick Cedeño")
author_container.write("Daniel Garijo")

st.divider()

# Aquí se muestra el abstract del paper.
st.subheader("Abstract", anchor=False)
texto_abstract = "Since their introduction in 2019, model cards have been widely adopted by the research community, since they provide a human-centered, clear summarization of machine learning approaches. In HuggingFace, these cards are manually filled by the authors when uploading their models, thus leading to missing or inaccurate information. Moreover, these cards are not semantically interoperable, since they are not compliant with any ontology or data model. This paper presents Paper2ModelCard, or P2MC, a framework for end-to-end generation of model cards from research papers. This framework integrates processing tools for correctly parsing the input PDF into a structured format (SciPDF and LightOnOCR-1B), which is then processed by an ensemble of models, each targeted towards efficiently extracting specific information from the input. The FAIR4ML ontology is used as the backbone to generate the model cards, in order to ensure their interoperability. Moreover, P2MC links the generated model cards to existing research knowledge graphs, such as SemOpenAlex or LinkedPapersWithCode."
st.markdown(
    f'<div style="text-align: justify;">{texto_abstract}</div>', 
    unsafe_allow_html=True
)


st.divider()
st.subheader("Execution Demo", anchor=False)
st.write("For executing the pipeline introduce a paper URL and click the button.")


with st.form("my_form"):
    st.write("Inside the form")
    url = st.selectbox(
        "Choose or introduce a PDF URL",
        ["https://arxiv.org/abs/1802.04394", "https://arxiv.org/abs/1711.04071", "https://arxiv.org/abs/1703.10316", "https://arxiv.org/abs/1712.02121", "https://arxiv.org/abs/1707.01476"],
        index=None,
        placeholder="Select one of the proposed PDFs below or introduce a new one by typing in the box and clicking add or pressing enter.",
        accept_new_options=True,
    )
    submitted = st.form_submit_button("Submit")
    if submitted:
        st.write("URL submitted:", url)

st.divider()
st.subheader("Examples", anchor=False)
st.write("Here we show some examples of the pipeline results. You can click on the articles and see how the extracted JSONLD looks like. Give it a try!")

if "selected_example" not in st.session_state:
    st.session_state.selected_example = None

def standardize_image(img_path, target_size=(300, 424)):
    img = Image.open(img_path)

    return ImageOps.fit(img, target_size, Image.Resampling.LANCZOS)

JSON_DIR = ASSETS_DIR / "final_jsons"
IMAGE_DIR = ASSETS_DIR / "paper_front_pages"


ejemplos = [
    {"title": "M-Walk", "key": "btn_ex_1", "model": "M-Walk", "json_path": JSON_DIR / "example1.json", "image_path": IMAGE_DIR / "example1.png", "xml_url": "https://github.com/oeg-upm/p2mc/blob/main/streamlit_front/assets/scipdf_xmls/example1.xml", "json_url": "https://github.com/oeg-upm/p2mc/blob/main/streamlit_front/assets/lightonocr_jsons/example1.json"},
    {"title": "KBGAN", "key": "btn_ex_2", "model": "KBGAN", "json_path": JSON_DIR / "example2.json", "image_path": IMAGE_DIR / "example2.png", "xml_url": "https://github.com/oeg-upm/p2mc/blob/main/streamlit_front/assets/scipdf_xmls/example2.xml", "json_url": "https://github.com/oeg-upm/p2mc/blob/main/streamlit_front/assets/lightonocr_jsons/example2.json"},
    {"title": "ParTrans-X", "key": "btn_ex_3", "model": "ParTrans-X", "json_path": JSON_DIR / "example3.json", "image_path": IMAGE_DIR / "example3.png", "xml_url": "https://github.com/oeg-upm/p2mc/blob/main/streamlit_front/assets/scipdf_xmls/example3.xml", "json_url": "https://github.com/oeg-upm/p2mc/blob/main/streamlit_front/assets/lightonocr_jsons/example3.json"},
    {"title": "ConvKB", "key": "btn_ex_4", "model": "ConvKB", "json_path": JSON_DIR / "example4.json", "image_path": IMAGE_DIR / "example4.png", "xml_url": "https://github.com/oeg-upm/p2mc/blob/main/streamlit_front/assets/scipdf_xmls/example4.xml", "json_url": "https://github.com/oeg-upm/p2mc/blob/main/streamlit_front/assets/lightonocr_jsons/example4.json"},
    {"title": "ConvE", "key": "btn_ex_5", "model": "ConvE", "json_path": JSON_DIR /"example5.json", "image_path": IMAGE_DIR /"example5.png","xml_url":"https://github.com/oeg-upm/p2mc/blob/main/streamlit_front/assets/scipdf_xmls/example5.xml","json_url":"https://github.com/oeg-upm/p2mc/blob/main/streamlit_front/assets/lightonocr_jsons/example5.json"},
]

columnas = st.columns(5)

for col, ej in zip(columnas, ejemplos):
    with col:
        st.write(f"**{ej['model']}**")
        
        img_uniforme = standardize_image(ej["image_path"])
        st.image(img_uniforme, width='stretch')

        if st.button("Show ModelCard", key=ej["key"], type="primary", width='stretch'):
            st.session_state.selected_example = ej["model"]
            st.session_state.selected_json_path = ej["json_path"]
            
        st.link_button("SciPDF XML", url=ej["xml_url"], width='stretch')
        st.link_button("OCR JSON", url=ej["json_url"], width='stretch')


if st.session_state.get("selected_example"):
    with st.container(border=True):
        st.subheader(f"Generated ModelCard for {st.session_state.selected_example}:", anchor=False)
        st.write("ModelCard generated in JSON-LD format:")

        json_file_path = st.session_state.get("selected_json_path")
        
        try:
            with open(json_file_path, "r", encoding="utf-8") as file:
                json_content = file.read()
                
            with st.container(height=400):
                st.code(json_content, language="json")
            
        except FileNotFoundError:
            st.error(f"JSON file not found: {json_file_path}")


