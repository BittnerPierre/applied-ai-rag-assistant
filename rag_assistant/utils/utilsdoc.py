import io
import os
import re
import shutil
import uuid

from pathlib import Path
from typing import Union, Optional

import boto3
import chromadb
from PyPDF2 import PdfReader

from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.docstore.document import Document
from langchain.text_splitter import TokenTextSplitter
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS, Chroma

from streamlit.runtime.uploaded_file_manager import UploadedFile

from .utilsllm import load_embeddings
from .config_loader import load_config


config = load_config()


def extract_unique_name(collection_name : str, key : str):
    store = get_store(collection_name=collection_name)
    collection = store._collection
    metadatas = collection.get()['metadatas']

    unique_names = set()
    for item in metadatas:
        if item is not None and key in item:
            unique_names.add(item[key])
    return unique_names


def split_documents(documents: list[Document]):
    # Initialize text splitter
    text_splitter = TokenTextSplitter(chunk_size=1024, chunk_overlap=24)
    chunks = text_splitter.split_documents(documents)

    return chunks


def process_txt_folder(txt_input_folder_name, txt_folder_name):
    # Initialize tokenizer
    # tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    data = []
    sources = []

    text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=24)

    # Iterate over all files in the folder
    for filename in os.listdir(txt_input_folder_name):
        # Only process PDF files
        if filename.endswith(".txt"):
            # Full path to the file
            filepath = os.path.join(txt_input_folder_name, filename)

            # Write the extracted text to a .txt file
            txt_filename = filename
            txt_filepath = os.path.join(txt_folder_name, txt_filename)
            path = Path(txt_filepath)
            if not path.is_file():
                print("Storing text:", txt_filename)
                shutil.copy(filepath, txt_filepath)

            # Read the .txt file
            with open(txt_filepath, 'r') as f:
                data.append(f.read())
            sources.append(filename)

        # Here we split the documents, as needed, into smaller chunks.
        # We do this due to the context limits of the LLMs.
        docs = []
        metadatas = []
        for i, d in enumerate(data):
            splits = text_splitter.create_documents(d)
            docs.extend(splits)
            metadatas.extend([{"source": sources[i]}] * len(splits))

    # Return the array of chunks
    return docs, metadatas


def empty_store(collection_name="Default") -> None:

    vectordb = config['VECTORDB']['vectordb']

    if vectordb == "faiss":
        raise NotImplementedError(f"Sorry, empty_store for {vectordb} not implemented yet!")

    elif vectordb == "chroma":
        persist_directory = config['VECTORDB']['chroma_persist_directory']
        persistent_client = chromadb.PersistentClient(path=persist_directory)

        persistent_client.delete_collection(name=collection_name)
    else:
        raise NotImplementedError(f"{vectordb} empty_store not implemented yet")

    pass


def load_store(documents: list[Document], embeddings: Embeddings = None, collection_name=None, split=True) -> VectorStoreIndexWrapper:

    vectordb = config['VECTORDB']['vectordb']

    # Store embeddings to vector db
    if split:
        documents = split_documents(documents)

    if not embeddings:
        embeddings = load_embeddings()

    if not collection_name:
        collection_name = config['VECTORDB']['collection_name']

    db = None

    if vectordb == "faiss":

        index_dir = config['VECTORDB']['faiss_persist_directory']

        db = FAISS.from_documents(documents, embeddings)

        ## This is generating this error:
        ## TypeError: cannot pickle '_thread.RLock' object
        ## so moving to FAISS.load_local and
        # index_file = f"{index_dir}/docs.index"
        # pkl_file = f"{index_dir}/faiss_store.pkl"
        # faiss.write_index(db.index, index_file)
        # db.index = None
        # with open(pkl_file, "wb") as f:
        #    pickle.dump(db, f)
        db.save_local(index_dir)

    elif vectordb == "chroma":
        persist_directory = config['VECTORDB']['chroma_persist_directory']

        persistent_client = chromadb.PersistentClient(path=persist_directory)
        collection = persistent_client.get_or_create_collection(name=collection_name)

        # # Create a list of unique ids for each document based on the content
        # ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, doc.page_content)) for doc in documents]
        # unique_ids = list(set(ids))
        # # print(f"""Unique Ids : {unique_ids}""")
        #
        # # Ensure that only docs that correspond to unique ids are kept and that only one of the duplicate ids is kept
        # seen_ids = set()
        # unique_docs = [doc for doc, id in zip(documents, ids) if id not in seen_ids and (seen_ids.add(id) or True)]

        # Create a bag (dict) for unique ids for each document based on the content
        id_bag = {}
        unique_docs = []
        for doc in documents:
            id = str(uuid.uuid5(uuid.NAMESPACE_DNS, doc.page_content))
            if id not in id_bag:
                id_bag[id] = True
                unique_docs.append(doc)

        # Get unique ids
        unique_ids = list(id_bag.keys())

        db = Chroma.from_documents(
            documents=unique_docs,
            embedding=embeddings,
            ids=unique_ids,
            client=persistent_client,
            collection_name=collection_name,
            persist_directory=persist_directory
        )
        # Since Chroma 0.4.x the manual persistence method is no longer supported as docs are automatically persisted.
        # db.persist()
    else:
        raise NotImplementedError(f"{vectordb} load_store not implemented yet")

    return db


def get_store(embeddings: Embeddings = None, collection_name=None):

    vectordb = config['VECTORDB']['vectordb']

    if not embeddings:
        embeddings = load_embeddings()

    if not collection_name:
        # "Default"
        collection_name = config['VECTORDB']['collection_name']

    db = None

    if vectordb == "faiss":
        index_dir = config['VECTORDB']['faiss_persist_directory']

        ## add allow_dangerous_deserialisation=True to solve this error
        ## "ValueError: The de-serialization relies loading a pickle file.
        ## Pickle files can be modified to deliver a malicious payload that results in execution of arbitrary code on
        ## your machine.You will need to set `allow_dangerous_deserialization` to `True`
        ## to enable deserialization. If you do this, make sure that you trust the source of the data.
        ## For example, if you are loading a file that you created, and no that no one else has modified the file,
        ## then this is safe to do. Do not set this to `True`
        ## if you are loading a file from an untrusted source (e.g., some random site on the internet.)."
        db = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)

    elif vectordb == "chroma":
        persist_directory = config['VECTORDB']['chroma_persist_directory']
        persistent_client = chromadb.PersistentClient(path=persist_directory)

        db = Chroma(client=persistent_client, collection_name=collection_name, embedding_function=embeddings)
    else:
        raise NotImplementedError(f"{vectordb} get_store not implemented yet")

    return db


def clean_text(s):
    regex_replacements = [
        (re.compile(r'([^\\])\\([^\\])'), r'\1\\\\\2'),
        (re.compile(r',(\s*])'), r'\1'),
    ]
    for regex, replacement in regex_replacements:
        s = regex.sub(replacement, s)
    return s


def load_doc(pdfs: Union[list[UploadedFile], None, UploadedFile], metadata = None) -> Optional[list[Document]]:
    if pdfs is not None:
        docs = []
        if metadata is None:
            metadata = {}
        for pdf in pdfs:
            if pdf.type == "application/pdf":
                reader = PdfReader(pdf)
                for i, page in enumerate(reader.pages, start=1):
                    page_metadata = {'page': i, 'filename': pdf.name, "type": "Text"}
                    file_content = page.extract_text()
                    page_metadata.update(metadata)
                    docs.append(Document(page_content=clean_text(file_content), metadata=page_metadata))
            elif pdf.type == "text/plain":
                page_metadata = {'filename': pdf.name, "type": "Text"}
                file_content = pdf.read().decode()  # read file content and decode it
                docs.append(Document(page_content=clean_text(file_content), metadata=page_metadata))
        persist_uploaded_documents(pdfs)
        return docs
    else:
        return None


def load_text(texts: Union[list[UploadedFile], None, UploadedFile], metadata: dict={}):
    if texts is not None:
        docs = []
        for txt in texts:
            page_metadata = {'filename': txt.name}
            file_content = txt.read().decode()  # read file content and decode it
            docs.append(Document(page_content=file_content, metadata=metadata))
        return docs
    else:
        return None


def persist_uploaded_documents(documents: list[UploadedFile]) -> None:
    """Persist documents uploaded from Streamlit"""
    for document in documents:
        persist_file(document, document.name)


def persist_images(images) -> None:
    """Persist images from pypdf"""
    for image in images:
        persist_file(image, image.name, 'image')


def persist_chat(chat: str, chat_name: str) -> None:
    """Persist chat history"""
    raise NotImplementedError('Function not implemented yet.')


def persist_file(file, filename:str, file_type: str = '') -> None:
    """Persist file to selected storage interface"""
    storage_interface = config.get('DOCUMENTS_STORAGE', 'INTERFACE')

    if storage_interface == 'LOCAL':
        persist_file_locally(file, filename=filename, file_type=file_type)
    elif storage_interface == 'S3':
        persist_file_to_s3(file, filename=filename, file_type=file_type)
    elif storage_interface == 'NONE':
        pass
    else:
        raise NotImplementedError(f"{storage_interface} not implemented yet for storage.")


def persist_file_to_s3(file, filename: str, file_type: str) -> None:
    """Persist file to S3 storage"""
    s3_bucket = config.get('DOCUMENTS_STORAGE', 'S3_BUCKET_NAME')

    folder_name = "images" if file_type == "image" else "docs"
    file_key = f"{folder_name}/{filename}"

    s3_client = boto3.client('s3')

    if file_type == 'image':
        file_object =  io.BytesIO(file.data)
    else:
        file_object = io.BytesIO(file.getvalue())

    s3_client.upload_fileobj(file_object, s3_bucket, file_key)


def persist_file_locally(file, filename:str, file_type: str) -> None:
    """Persist file to local storage"""
    documents_path = config.get('DOCUMENTS_STORAGE', 'DOCUMENTS_PATH')
    folder_name = "images" if file_type == "image" else "docs"

    file_path = os.path.join(documents_path, folder_name)

    if not os.path.exists(file_path):
        os.makedirs(file_path)

    file_path = os.path.join(file_path, filename)

    with open(file_path, 'wb') as f:
        if file_type == 'image':
            f.write(file.data)
        else:
            f.write(file.getvalue())


def get_file(filename: str, file_type: str=''):
    """Get file from selected storage interface"""
    storage_interface = config.get('DOCUMENTS_STORAGE', 'INTERFACE')

    if storage_interface == 'LOCAL':
        return get_file_locally(filename=filename, file_type=file_type)

    if storage_interface == 'S3':
        return get_file_from_s3(filename=filename, file_type=file_type)

    if storage_interface == 'NONE':
        return None

    raise NotImplementedError(f"{storage_interface} not implemented yet for storage.")


def get_file_locally(filename: str, file_type: str):
    """Get file from local storage"""
    documents_path = config.get('DOCUMENTS_STORAGE', 'DOCUMENTS_PATH')
    folder_name = "images" if file_type == "image" else "docs"

    file_path = os.path.join(documents_path, folder_name, filename)

    if os.path.exists(file_path):
        return open(file_path, 'rb').read()

    return None


def get_file_from_s3(filename: str, file_type: str):
    """Get file from S3 storage"""
    s3_bucket = config.get('DOCUMENTS_STORAGE', 'S3_BUCKET_NAME')

    folder_name = "images" if file_type == "image" else "docs"
    file_key = f"{folder_name}/{filename}"

    s3_client = boto3.client('s3')

    response = s3_client.get_object(Bucket=s3_bucket, Key=file_key)
    return response['Body'].read()
