import streamlit as st

from utils.utilsdoc import load_doc, load_store, load_text
from utils.config_loader import load_config

config = load_config()
app_name = config['DEFAULT']['APP_NAME']
collection_name = config['VECTORDB']['collection_name']

st.set_page_config(page_title=f"""📄 {app_name} 🤗""", page_icon="📄")


def main():
    st.title(f"""📄 {app_name} : Chargement du vectorstore 🤗""")

    # with st.form("Upload File"):
    topic_name = st.text_input("Sujet / Thème du document (ex: AWS, Serverless, Architecture, Sécurité, ...)")

    file_type = st.radio("Type du document", ["Whitepaper", "Guide", "Tutorial", "FAQ"], index=None)

    # dt_code = st.text_input("DT Code")
    # siren = st.text_input("SIREN")

    pdfs = st.file_uploader("Document(s) à transmettre", type=['pdf', 'txt', 'md'], accept_multiple_files=True)

    disabled = True
    print(file_type)
    print(topic_name)
    if (file_type is not None) and (topic_name is not None) and (pdfs is not None) and (len(pdfs)):
        disabled = False

    if st.button("Transmettre", disabled=disabled):
        metadata = {"type": file_type, "topic_name": topic_name}
        docs = load_doc(pdfs, metadata)
        store = load_store(docs, collection_name=collection_name)
        # print(docs)


if __name__ == "__main__":
    main()
