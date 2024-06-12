import streamlit as st

import json

from shared.constants import DocumentType, ChunkType, Metadata
from utils.utilsdoc import get_store, empty_store, extract_unique_name
from utils.config_loader import load_config


config = load_config()
app_name = config['DEFAULT']['APP_NAME']
collection_name = config['VECTORDB']['collection_name']

st.set_page_config(page_title=f"""📄 {app_name} 🤗""", page_icon="📄")


def main():
    st.title(f"""Gestion des connaissances 📄""")

    # collection_name = st.selectbox("Collection", ["Default", "RAG"])

    store = get_store()
    collection = store._collection
    count = collection.count()
    if count > 0:
        st.write(f"There are {count} chunks in the collection.")
    else:
        st.write("Collection is empty. Load knowledge with load document pages.")

    st.subheader("File loaded")

    unique_filenames = extract_unique_name(collection_name,  Metadata.FILENAME.value)

    for name in unique_filenames:
        st.markdown(f"""- {name}""")

    st.subheader("Topic loaded")
    unique_topic_names = extract_unique_name(collection_name, Metadata.TOPIC.value)
    for name in unique_topic_names:
        st.markdown(f"""- {name}""")

    # st.subheader("Document Type")
    # unique_document_types = extract_unique_name(collection_name, 'document_type')
    # for name in unique_document_types:
    #     st.markdown(f"""- {name}""")

    with st.form("Search in vDB"):
        search = st.text_input("Text (*)")

        topic_name = st.selectbox("Topic", unique_topic_names, index=None)
        filename = st.selectbox("File Name", unique_filenames, index=None)
        document_type = st.selectbox("Document Type", [e.value for e in DocumentType], index=None)
        chunk_type = st.selectbox("Chunk Type", [e.value for e in ChunkType], index=None)
        #document_type = st.selectbox("Document Type", unique_document_types, index=None)

        filters = []
        if filename:
            filters.append({Metadata.FILENAME.value: filename})
        if document_type:
            filters.append({Metadata.DOCUMENT_TYPE.value: document_type})
        if topic_name:
            filters.append({Metadata.TOPIC.value: topic_name})
        if document_type:
            filters.append({Metadata.DOCUMENT_TYPE.value: document_type})
        if chunk_type:
            filters.append({Metadata.CHUNK_TYPE.value: chunk_type})
        if st.form_submit_button("Search"):
            # add check for empty string as it is not supported by bedrock (or anthropic?)
            if search != "":
                if len(filters) > 1:
                    where = {"$and": filters}
                elif len(filters) == 1:
                    where = filters[0]
                else:
                    where = {}

                result = store.similarity_search(search, k=5, filter=where)  # , kwargs={"score_threshold": .8}
                st.write(result)
            else:
                st.write("Please, specify a text.")

    st.subheader("Data Management")

    col1, col2 = st.columns(2)
    with col1:
        file_name_to_delete = st.selectbox("Select File Name", unique_filenames, index=None)
        if st.button("Delete File data"):
            collection.delete(where={"filename": {"$eq": f"{file_name_to_delete}"}})

    with col2:
        topic_name_to_delete = st.selectbox("Select Topic", unique_topic_names, index=None)
        if st.button("Delete Topic Data"):
            collection.delete(where={"topic_name": {"$eq": f"{topic_name_to_delete}"}})

    if st.button("Delete collection"):
        empty_store(collection_name=collection_name)

    if collection:
        with st.expander("See All Metadatas", expanded=False):
            st.subheader("Metadatas")
            metadatas = collection.get()['metadatas']
            st.code(json.dumps(metadatas, indent=4, sort_keys=True), language="json")


if __name__ == "__main__":
    main()
