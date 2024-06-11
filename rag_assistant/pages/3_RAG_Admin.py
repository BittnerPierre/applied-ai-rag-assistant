import streamlit as st

import json

from streamlit_pdf_viewer import pdf_viewer
from utils.utilsdoc import get_store, empty_store, extract_unique_name, get_file
from utils.config_loader import load_config


config = load_config()
app_name = config['DEFAULT']['APP_NAME']
collection_name = config['VECTORDB']['collection_name']

st.set_page_config(page_title=f"""📄 {app_name} 🤗""", page_icon="📄")


def main():
    st.title(f"""📄 {app_name} View vDB 🤗""")

    collection_name = st.selectbox("Collection", ["Default", "RAG"])

    store = get_store(collection_name=collection_name)
    collection = store._collection
    st.write("There are", collection.count(), "in the collection")

    st.subheader("File loaded")
    metadatas = collection.get()['metadatas']
    unique_filenames = extract_unique_name(collection_name, 'filename')

    for name in unique_filenames:
        st.markdown(f"""- {name}""")
        show = st.toggle("Show document", False, key=f"{name}")
        if show:
            binary_data = get_file(name)
            st.subheader(f"{name}")
            if binary_data:
                pdf_viewer(input=binary_data, width=700, height=800)


    st.subheader("Topic loaded")
    unique_topic_names = extract_unique_name(collection_name, 'topic_name')
    for name in unique_topic_names:
        st.markdown(f"""- {name}""")

    st.subheader("Document Type")
    unique_document_types = extract_unique_name(collection_name, 'document_type')
    for name in unique_document_types:
        st.markdown(f"""- {name}""")

    with st.form("Search in vDB"):
        search = st.text_input("Text")

        topic_name = st.selectbox("Topic", unique_topic_names, index=None)
        filename = st.selectbox("File Name", unique_filenames, index=None)
        filetype = st.selectbox("File type", ("Whitepaper", "Guide", "Tutorial", "FAQ"), index=None)
        document_type = st.selectbox("Document Type", unique_document_types, index=None)

        filter = {}
        if filename:
            filter["filename"] = filename
        elif document_type:
            filter["document_type"] = document_type
        elif topic_name:
            filter["topic_name"] = topic_name
        # if filetype:
        #    filter["type"] = filetype
        if st.form_submit_button("Search"):
            # add check for empty string as it is not supported by bedrock (or anthropic?)
            results_doc_pages = []
            if search != "":
                result = store.similarity_search(search, k=5, filter=filter)  # , kwargs={"score_threshold": .8}
                st.write(result)
                for doc in result:
                    if (doc.metadata["filename"], doc.metadata["page"]) not in results_doc_pages:
                        results_doc_pages.append((doc.metadata["filename"], doc.metadata["page"]))
                for (filename, page) in results_doc_pages:    
                    binary_data = get_file(filename)
                    if binary_data:
                        key = f"results.{filename}.{page}"
                        pdf_viewer(input=binary_data, width=700, pages_to_render=[page], key=key)

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

    with st.expander("See All Metadatas", expanded=False):
        st.subheader("Metadatas")
        st.code(json.dumps(metadatas, indent=4, sort_keys=True), language="json")


if __name__ == "__main__":
    main()
