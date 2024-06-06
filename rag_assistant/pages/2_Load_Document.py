import streamlit as st
from langchain_community.document_loaders.pdf import PyPDFLoader
import os

from utils.config_loader import load_config
from utils.utilsdoc import load_doc, load_store

from utils.utilsvision import load_image

config = load_config()

app_name = config['DEFAULT']['APP_NAME']
collection_name = config['VECTORDB']['collection_name']

st.set_page_config(page_title=f"""📄 {app_name} 🤗""", page_icon="📄")


def main():
    st.title(f"""📄 {app_name} : Chargement du vectorstore 🤗""")

    # with st.form("Upload File"):
    topic_name = st.text_input("Sujet / Thème du document (ex: AWS, Serverless, Architecture, Sécurité, ...)")

    file_type = st.radio("Type du document", ["Whitepaper", "Guide", "Tutorial", "FAQ"], index=None)

    analyse_images = st.checkbox("Analyse images")
    image_only = st.checkbox("Analyse images only")
    restart_image_analysis = st.checkbox("Restart Image Analysis")

    pdfs = st.file_uploader("Document(s) à transmettre", type=['pdf', 'txt', 'md'], accept_multiple_files=True)

    disabled = True
    if (file_type is not None) and (topic_name is not None) and (pdfs is not None) and (len(pdfs)):
        disabled = False

    if st.button("Transmettre", disabled=disabled):
        metadata = {"type": file_type, "topic_name": topic_name}
        docs = []
        if not image_only:
            docs += load_doc(pdfs, metadata)
        if analyse_images:
            image_docs = load_image(pdfs, metadata, restart_image_analysis)
            docs += image_docs
        load_store(docs, collection_name=collection_name)


if __name__ == "__main__":
    main()
