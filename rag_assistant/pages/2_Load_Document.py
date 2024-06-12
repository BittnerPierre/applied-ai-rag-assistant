import streamlit as st
from langchain_community.document_loaders.pdf import PyPDFLoader
import os

from shared.constants import DocumentType, SupportedFileType, Metadata
from utils.config_loader import load_config
from utils.utilsdoc import load_doc, load_store

from utils.utilsvision import load_image

config = load_config()

app_name = config['DEFAULT']['APP_NAME']
collection_name = config['VECTORDB']['collection_name']
upload_directory = config['FILE_MANAGEMENT']['UPLOAD_DIRECTORY']

st.set_page_config(page_title=f"""📄 {app_name} 🤗""", page_icon="📄")


def main():
    st.title(f"""Chargement des connaissances 📄""")

    # with st.form("Upload File"):
    topic_name = st.text_input("Thème du document (ex: API, Cloud, Data, Architecture, Sécurité, ...)")

    file_type = st.radio("Type du document", [e.value for e in DocumentType], index=None)

    with st.container():
        st.write("Traitement des images")
        analyse_images = st.checkbox("Analyser les images")
        image_only = st.checkbox("Traiter uniquement les images (test mode)", disabled=(not analyse_images))
        restart_image_analysis = st.checkbox("Relancer l'analyse d'image (test mode)", disabled=(not analyse_images))

    pdfs = st.file_uploader("Document(s) à transmettre", type=[e.value for e in SupportedFileType], accept_multiple_files=True)

    disabled = True
    if (file_type is not None) and (topic_name is not None) and (pdfs is not None) and (len(pdfs)):
        disabled = False

    if st.button("Transmettre", disabled=disabled):

        file_paths = []
        if not os.path.exists(upload_directory):
            os.makedirs(upload_directory)
        for pdf in pdfs:
            file_path = os.path.join(upload_directory, pdf.name)
            with open(file_path, 'wb') as f:
                f.write(pdf.read())
            file_paths.append(file_path)

        metadata = {Metadata.DOCUMENT_TYPE.value: file_type, Metadata.TOPIC.value: topic_name}
        docs = []
        if not image_only:
            docs += load_doc(pdfs, metadata)
        if analyse_images:
            image_docs = load_image(pdfs, metadata, restart_image_analysis)
            docs += image_docs
        load_store(docs, collection_name=collection_name)

if __name__ == "__main__":
    main()
