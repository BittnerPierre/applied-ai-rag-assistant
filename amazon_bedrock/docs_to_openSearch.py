import boto3
import json
from dotenv import load_dotenv
import os
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from urllib.parse import urlparse
import click

# Load environment variables
load_dotenv()

# instantiating the bedrock client, with specific CLI profile
boto3.setup_default_session(profile_name=os.getenv('profile_name'))
bedrock = boto3.client('bedrock-runtime', 'eu-central-1', endpoint_url='https://bedrock-runtime.eu-central-1.amazonaws.com')
opensearch = boto3.client("opensearchserverless")
s3 = boto3.client("s3")

bucket_name = os.getenv("bucket_name")
prefix = os.getenv("prefix")

region = 'eu-central-1'
opensearch_host = os.getenv('opensearch_host')
parsed_url = urlparse(opensearch_host)
host = parsed_url.hostname
service = 'aoss'
credentials = boto3.Session(profile_name=os.getenv('profile_name')).get_credentials()
auth = AWSV4SignerAuth(credentials, region, service)

client = OpenSearch(
    hosts=[{'host': host, 'port': 443}],
    http_auth=auth,
    use_ssl=True,
    verify_certs=True,
    connection_class=RequestsHttpConnection,
    pool_maxsize=20
)

def get_embedding(body):
    """
    This function is used to generate the embeddings for a specific chunk of text
    :param body: This is the example content passed in to generate an embedding
    :return: A vector containing the embeddings of the passed in content
    """

    modelId = 'amazon.titan-embed-text-v1'
    accept = 'application/json'
    contentType = 'application/json'
    response = bedrock.invoke_model(body=body, modelId=modelId, accept=accept, contentType=contentType)
    response_body = json.loads(response.get('body').read())
    embedding = response_body.get('embedding')
    return embedding

def index_document(client, vectors, text):
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
    print(response)
    return response

@click.command()
@click.option('--mode', default='local', help='Specify the mode to run the script (local or other).')
def run(mode : str):
    if mode == "local" :
        loader = PyPDFLoader("/Users/loicsteve/Downloads/RefAE_NormeAPI.pdf")
        documents = loader.load()
    else:
        # Create a local directory to store the downloaded PDF files
        local_dir = "/Users/loicsteve/Desktop/applied-ai-rag-assistant/amazon_bedrock/pdfs"
        os.makedirs(local_dir, exist_ok=True)
        # if not os.path.exists(local_dir):
        #     os.makedirs(local_dir)

        #set to track downloaded files names
        downloaded_files = set()

        response = s3.list_objects(Bucket=bucket_name, Prefix=prefix)

        if "Contents" in response:
            for obj in response["Contents"]:
                file_name = os.path.basename(obj["Key"])
                local_file_path = os.path.join(local_dir, file_name)

                # Skip if the file has already been downloaded
                if file_name in downloaded_files:
                    continue
                    
                try:
                    # Ensure parent directory exists for the file
                    os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                    # Download the file from S3
                    s3.download_file(Bucket=bucket_name, Key=obj["Key"], Filename=local_file_path)
                    downloaded_files.add(file_name)  # Add to downloaded files set
                except Exception as e:
                    print(f"Error downloading {file_name}: {e}")

        #Load the PDF files using PyPDFDirectoryLoader
        loader = PyPDFDirectoryLoader(local_dir)
        documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,      # overlap between chunks
        )

    doc = text_splitter.split_documents(documents)


    # The process of iterating through each chunk of document we are trying to index, generating the embeddings, and indexing the document.
    for i in doc:
        # The text data of each chunk
        exampleContent = i.page_content
        # The embeddings of each chunk
        exampleInput = json.dumps({"inputText": exampleContent})
        exampleVectors = get_embedding(exampleInput)
        # setting the text data as the text variable, and generated vector to a vector variable
        text = exampleContent
        vectors = exampleVectors
        # calling the indexDoc function, passing in the OpenSearch Client, the created vector, and corresponding text data
        index_document(client, vectors, text)


if __name__ == '__main__':
    run()

