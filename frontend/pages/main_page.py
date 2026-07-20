import streamlit as st
from pathlib import Path
from PIL import Image, ImageOps


if "selected_example" not in st.session_state:
    st.session_state.selected_example = None

if "selected_json_path" not in st.session_state:
    st.session_state.selected_json_path = None

if "selected_artifact_title" not in st.session_state:
    st.session_state.selected_artifact_title = None

if "selected_artifact_path" not in st.session_state:
    st.session_state.selected_artifact_path = None

if "selected_artifact_language" not in st.session_state:
    st.session_state.selected_artifact_language = None

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

# Este es el contenedor para los autores, cada uno enlazado a su ORCID.
autores = [
    {"name": "Mateo Campaya-Pinto", "orcid": ""},
    {"name": "Elvira Amador-Dominguez", "orcid": "https://orcid.org/0000-0001-6838-1266"},
    {"name": "Erick Cede�o", "orcid": "https://orcid.org/0009-0004-4478-6890"},
    {"name": "Daniel Garijo", "orcid": "https://orcid.org/0000-0003-0454-7145"},
]

author_container = st.container(horizontal=True, horizontal_alignment="center")
for autor in autores:
    author_container.link_button(autor["name"], url=autor["orcid"])

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
XML_DIR = ASSETS_DIR / "scipdf_xmls"
OCR_DIR = ASSETS_DIR / "lightonocr_jsons"
IMAGE_DIR = ASSETS_DIR / "paper_front_pages"


ejemplos = [
    {"title": "M-Walk", "key": "btn_ex_1", "model": "M-Walk", "json_path": JSON_DIR / "example1.json", "xml_path": XML_DIR / "example1.xml", "ocr_path": OCR_DIR / "example1.json", "image_path": IMAGE_DIR / "example1.png"},
    {"title": "KBGAN", "key": "btn_ex_2", "model": "KBGAN", "json_path": JSON_DIR / "example2.json", "xml_path": XML_DIR / "example2.xml", "ocr_path": OCR_DIR / "example2.json", "image_path": IMAGE_DIR / "example2.png"},
    {"title": "ParTrans-X", "key": "btn_ex_3", "model": "ParTrans-X", "json_path": JSON_DIR / "example3.json", "xml_path": XML_DIR / "example3.xml", "ocr_path": OCR_DIR / "example3.json", "image_path": IMAGE_DIR / "example3.png"},
    {"title": "ConvKB", "key": "btn_ex_4", "model": "ConvKB", "json_path": JSON_DIR / "example4.json", "xml_path": XML_DIR / "example4.xml", "ocr_path": OCR_DIR / "example4.json", "image_path": IMAGE_DIR / "example4.png"},
    {"title": "ConvE", "key": "btn_ex_5", "model": "ConvE", "json_path": JSON_DIR / "example5.json", "xml_path": XML_DIR / "example5.xml", "ocr_path": OCR_DIR / "example5.json", "image_path": IMAGE_DIR / "example5.png"},
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
            st.session_state.selected_artifact_title = (
                f"Generated ModelCard for {ej['model']}:"
            )
            st.session_state.selected_artifact_path = ej["json_path"]
            st.session_state.selected_artifact_language = "json"
            
        if st.button("Show SciPDF XML", key=f"{ej['key']}_xml", width='stretch'):
            st.session_state.selected_artifact_title = (
                f"SciPDF XML for {ej['model']}:"
            )
            st.session_state.selected_artifact_path = ej["xml_path"]
            st.session_state.selected_artifact_language = "xml"

        if st.button("Show OCR JSON", key=f"{ej['key']}_ocr", width='stretch'):
            st.session_state.selected_artifact_title = (
                f"LightOCR JSON for {ej['model']}:"
            )
            st.session_state.selected_artifact_path = ej["ocr_path"]
            st.session_state.selected_artifact_language = "json"


if st.session_state.get("selected_artifact_path"):
    with st.container(border=True):
        st.subheader(st.session_state.selected_artifact_title, anchor=False)

        artifact_path = st.session_state.get("selected_artifact_path")
        language = st.session_state.get("selected_artifact_language")
        
        try:
            with open(artifact_path, "r", encoding="utf-8") as file:
                artifact_content = file.read()
                
            with st.container(height=400):
                st.code(artifact_content, language=language)
            
        except FileNotFoundError:
            st.error(f"File not found: {artifact_path}")


# ---------------------------------------------------------------------------
# Footer: agradecimientos y logos institucionales
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

st.divider()

st.subheader("Acknowledgements", anchor=False)

texto_agradecimientos = (

    "This work has been partially supported by [NOMBRE DEL PROYECTO/AYUDA], "

    "funded by [ENTIDAD FINANCIADORA]. The authors also acknowledge the support "

    "of the Ontology Engineering Group (OEG) at Universidad Politécnica de Madrid."

)

st.markdown(

    f'<div style="text-align: justify;">{texto_agradecimientos}</div>',

    unsafe_allow_html=True

)



st.write("")  # pequeño espacio antes de los logos



def standardize_logo(img_path, target_size=(200, 120)):
    """
    Ajusta un logo a un lienzo de tamaño fijo (target_size) sin recortarlo:
    lo escala manteniendo su proporción y lo centra sobre fondo transparente.
    """
    img = Image.open(img_path).convert("RGBA")
    img.thumbnail(target_size, Image.Resampling.LANCZOS)

    canvas = Image.new("RGBA", target_size, (0, 0, 0, 0))
    offset_x = (target_size[0] - img.width) // 2
    offset_y = (target_size[1] - img.height) // 2
    canvas.paste(img, (offset_x, offset_y), img)

    return canvas


LOGOS_DIR = ASSETS_DIR / "logos"



logos = [

    {"name": "Comunidad de Madrid", "path": LOGOS_DIR / "comunidad_madrid.png"},

    {"name": "UPM", "path": LOGOS_DIR / "upm_logo.png"},

    {"name": "Escuela", "path": LOGOS_DIR / "etsisi_logo.png"},

    {"name": "Grupo de investigación", "path": LOGOS_DIR / "oeg_logo.png"},

]



logo_cols = st.columns(len(logos))

for col, logo in zip(logo_cols, logos):

    with col:

        if logo["path"].exists():

            logo_uniforme = standardize_logo(logo["path"])

            st.image(logo_uniforme, width='stretch')

        else:

            st.caption(f"⚠️ Logo not found: {logo['path'].name}")

