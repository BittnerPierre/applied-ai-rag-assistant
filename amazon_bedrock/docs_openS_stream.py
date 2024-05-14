import boto3
import json
from dotenv import load_dotenv
import os
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
# from langchain_community.document_loaders.pdf import PyPDFData
from langchain_community.document_loaders import PDFPlumberLoader
from urllib.parse import urlparse
from io import BytesIO
import PyPDF2

# loading in environment variables
load_dotenv()

# instantiating the bedrock client, with specific CLI profile
boto3.setup_default_session(profile_name=os.getenv('profile_name'))
bedrock = boto3.client('bedrock-runtime', 'eu-central-1', endpoint_url='https://bedrock-runtime.eu-central-1.amazonaws.com')
opensearch = boto3.client("opensearchserverless")

# Instantiating the OpenSearch client, with specific CLI profile
#host = os.getenv('opensearch_host')  # cluster endpoint, for example: my-test-domain.us-east-1.aoss.amazonaws.com
region = 'eu-central-1'
opensearch_host = os.getenv('opensearch_host')
parsed_url = urlparse(opensearch_host)
host = parsed_url.hostname
service = 'aoss'
credentials = boto3.Session(profile_name=os.getenv('profile_name')).get_credentials()
auth = AWSV4SignerAuth(credentials, region, service)

s3 = boto3.client("s3")

client = OpenSearch(
    hosts=[{'host': host, 'port': 443}],
    http_auth=auth,
    use_ssl=True,
    verify_certs=True,
    connection_class=RequestsHttpConnection,
    pool_maxsize=20
)

# Embedding the documents into the OpenSearch
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
    print(response)
    return response

# Function to read PDF content from S3 object
def read_pdf_from_s3(bucket_name, key):
    response = s3.get_object(Bucket=bucket_name, Key=key)
    pdf_data = response['Body'].read()
    return pdf_data

# S3 bucket details
bucket_name = os.getenv("bucket_name")
prefix = os.getenv("prefix")

# Load PDF data and process
response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)

if "Contents" in response:
    for obj in response["Contents"]:
        file_name = os.path.basename(obj["Key"])

        # Process PDF data directly from S3
        pdf_data = read_pdf_from_s3(bucket_name, obj["Key"])
        pdf_file = BytesIO(pdf_data)

        # Use PyPDF2 to extract text from PDF in memory
        pdf_reader = PyPDF2.PdfReader(pdf_file)

        # Extract text content from each page
        full_text = ""
        for page_num in range(pdf_reader.getNumPages()):
            page = pdf_reader.getPage(page_num)
            full_text += page.extract_text()

        # Split full text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        doc = text_splitter.split_text(full_text)

        # Embedding and indexing
        for chunk in doc:
            example_content = chunk
            example_input = json.dumps({"inputText": example_content})
            example_vectors = get_embedding(example_input)  # Assuming get_embedding function is defined

            # Index the document chunk
            index_doc(client, example_vectors, example_content)  # Assuming index_doc function is defined