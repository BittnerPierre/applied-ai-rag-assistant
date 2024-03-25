import streamlit as st

import json

from utils.utilsdoc import get_store, empty_store
from utils.config_loader import load_config
from tools.vectorstore import extract_unique_name


config = load_config()
app_name = config['DEFAULT']['APP_NAME']
collection_name = config['VECTORDB']['collection_name']

st.set_page_config(page_title=f"""📄 {app_name} 🤗""", page_icon="📄")


def main():
    st.title(f"""📄 {app_name} View vDB 🤗""")

    collection_name = st.selectbox("Collection", ["Default", "JSON"])

    store = get_store(collection_name=collection_name)
    collection = store._collection
    st.write("There are", collection.count(), "in the collection")

    st.subheader("File loaded")
    metadatas = collection.get()['metadatas']
    unique_filenames = extract_unique_name(collection_name, 'filename')

    for name in unique_filenames:
        st.markdown(f"""- {name}""")

    st.subheader("Company loaded")
    unique_company_names = extract_unique_name(collection_name, 'company_name')
    for name in unique_company_names:
        st.markdown(f"""- {name}""")

    st.subheader("Document Type")
    unique_document_types = extract_unique_name(collection_name, 'document_type')
    for name in unique_document_types:
        st.markdown(f"""- {name}""")

    with st.form("Search in vDB"):
        search = st.text_input("Text")
        company_name = st.selectbox("Company Name", unique_company_names, index=None)
        filename = st.selectbox("File Name", unique_filenames, index=None)
        filetype = st.selectbox("File type", ("KBIS", "Status"), index=None)
        document_type = st.selectbox("Document Type", unique_document_types, index=None)

        filter = {}
        if filename:
            filter["filename"] = filename
        if document_type:
            filter["document_type"] = document_type
        if company_name:
            filter["company_name"] = company_name
        # if filetype:
        #    filter["type"] = filetype

        result = store.similarity_search(search,  k=5, filter=filter) # , kwargs={"score_threshold": .8}
        if st.form_submit_button("Search"):
            st.write(result)

    st.subheader("Data Management")

    col1, col2 = st.columns(2);
    with col1:
        file_name_to_delete = st.selectbox("Select File Name", unique_filenames, index=None)
        if st.button("Delete File data"):
            collection.delete(where={"filename": f"{file_name_to_delete}"})

    with col2:
        company_name_to_delete = st.selectbox("Select Company Name", unique_company_names, index=None)
        if st.button("Delete Company Data"):
            collection.delete(where={"company_name": f"{company_name_to_delete}"})

    if st.button("Delete collection"):
        empty_store(collection_name=collection_name)

    with st.expander("See All Metadatas", expanded=False):
        st.subheader("Metadatas")
        st.code(json.dumps(metadatas, indent=4, sort_keys=True), language="json")


if __name__ == "__main__":
    main()
