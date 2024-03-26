import streamlit as st
from langchain_community.document_loaders.pdf import PyPDFLoader
import os

from utils.utilsdoc import load_doc, load_store, load_text
from utils.config_loader import load_config

config = load_config()
app_name = config['DEFAULT']['APP_NAME']
collection_name = config['VECTORDB']['collection_name']

st.set_page_config(page_title=f"""📄 {app_name} 🤗""", page_icon="📄")


pdf_files_paths = [
    "data/sources/pdf/aws/caf/aws-caf-for-ai.pdf",
    "data/sources/pdf/aws/waf/AWS_Well-Architected_Framework.pdf",
    "data/sources/pdf/Questionnaire d'évaluation des risques applicatifs pour le Cloud Public.pdf",
    "data/sources/pdf/enisa/Cloud Security Guide for SMEs.pdf",
    "data/sources/pdf/aws/caf/aws-caf-for-ai.pdf",
    "data/sources/pdf/12 factor/beyond-the-twelve-factor-app.pdf",
    # Add more paths as needed
]

def main():
    st.title(f"""📄 {app_name} : Chargement du vectorstore 🤗""")

    # with st.form("Upload File"):
    topic_name = st.text_input("Sujet / Thème du document (ex: AWS, Serverless, Architecture, Sécurité, ...)")

    file_type = st.radio("Type du document", ["Whitepaper", "Guide", "Tutorial", "FAQ"], index=None)

    pdfs = st.file_uploader("Document(s) à transmettre", type=['pdf', 'txt', 'md'], accept_multiple_files=True)

    disabled = True
    if (file_type is not None) and (topic_name is not None) and (pdfs is not None) and (len(pdfs)):
        disabled = False

    if st.button("Transmettre", disabled=disabled):
        metadata = {"type": file_type, "topic_name": topic_name}
        docs = load_doc(pdfs, metadata)
        store = load_store(docs, collection_name=collection_name)
        # print(docs)

    if st.button("Load default"):

        docs = []
        for file_path in pdf_files_paths:
            loader = PyPDFLoader(file_path)
            loaded_docs = loader.load()
            filename = os.path.basename(file_path)
            metadata = {"type": "Whitepaper", "topic_name": "Cloud", "filename": filename}
            # Assuming each document supports a metadata dictionary
            for doc in loaded_docs:
                # Update metadata for each document
                doc.metadata.update({
                    "type": "Whitepaper",
                    "topic_name": "Cloud",
                    "filename": filename
                })

            docs.extend(loaded_docs)

        # embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        store = load_store(docs, collection_name=collection_name)



if __name__ == "__main__":
    main()
