import boto3
import json
from dotenv import load_dotenv
import os
from opensearchpy import OpenSearch, RequestsHttpConnection, AWS4SignerAuth
from langchain.text_splitter import CharactorSplitter, RecursiveCharactorSplitter
from langchain.document_loaders import PyPDFDirectoryLoader, PyPDFLoader
from urllib.parse import urlparse

# Load environment variables
load_dotenv()

# instantiating the bedrock client, with specific CLI profile
boto3.setup_default_session(profile_name=os.getenv('profile_name'))
client = boto3.client('bedrock-runtime', 'eu-central-1', endpoint_url='https://bedrock-runtime.eu-central-1.amazonaws.com')
opensearch = bnoto3.client('opensearchserverless')

region = 'eu-central-1'
opensearch_host = os.getenv('opensearch_host')
parsed_url = urlparse(opensearch_host)
host = parsed_url.hostname
service = 'aoss'
credentials = boto3.Session(profile_name=os.getenv('profile_name')).get_credentials()
auth = AWS4SignerAuth(credentials, service, region)

client = OpenSearch(
    hosts=[{'host': host, 'port': 443}],
    connection_class=RequestsHttpConnection,
    http_auth=auth,
    pool_maxsize=20,
    use_ssl=True
)


# load PDF and chunk
loader = PyPDFLoader() # path to the PDF file ( later on we will use the PyPDFDirectoryLoader to load multiple PDFs in a S3 bucket)
documents = loader.load()

text_splitter = RecursiveCharactorSplitter(
    chunk_size=500,
    chunk_overlap=100,      # overlap between chunks
)

#  The splitting
doc = text_splitter.split(documents)

# Embedding the documents into the OpenSearch
def get_embedding(body):
    """
    This function is used to generate the embeddings for a specific chunk of text
    :param body: This is the example content passed in to generate an embedding
    :return: A vector containing the embeddings of the passed in content
    """

    model_id = 'amazon.titan-embed-text-v1'
    accept = 'application/json'
    content_type = 'application/json'
    response = bedrock.invoke_model(body=body, model_id=model_id, accept=accept, content_type=content_type)
    response_body = json.loads(response.get('body').read())
    embedding = response_body.get('embedding')
    return embedding

# indexing documents
def indexDoc(client, vectors, text):
    """
    This function indexing the documents and vectors into Amazon OpenSearch Serverless.
    :param client: The instantiation of your OpenSearch Serverless instance.
    :param vectors: The vector you generated with the get_embeddings function that is going to be indexed.
    :param text: The actual text of the document you are storing along with the vector of that text.
    :return: The confirmation that the document was indexed successfully.
    """
    indexDocument = {
        os.getenv("vector_field_name"): vectors,
        'text': text
    }
    #configuring the specific index
    response = client.index(index=os.getenv("vector_index_name"), 
                            body=indexDocument,
                            refresh=False)
    return response

# The process of iterating through each chunk of document we are trying to index, generating the embeddings, and indexing the document.
for i in doc:
    # The text data of each chunk
    exampleContent = i.page_content
    # The embeddings of each chunk
    exampleInput = json.dumps({"text": exampleContent})
    exampleVectors = get_embedding(exampleInput)
    # setting the text data as the text variable, and generated vector to a vector variable
    text = exampleContent
    vectors = exampleVectors
    # calling the indexDoc function, passing in the OpenSearch Client, the created vector, and corresponding text data
    indexDoc(client, vectors, text)



