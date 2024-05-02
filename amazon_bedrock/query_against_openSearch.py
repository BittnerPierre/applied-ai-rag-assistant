import boto3
import os
from dotenv import load_dotenv
import os
from opensearchpy import OpenSearch, RequestsHttpConnection, AWS4SignerAuth

# Load environment variables
load_dotenv()

# instantiating the bedrock client, with specific CLI profile
boto3.setup_default_session(profile_name=os.getenv('profile_name'))
bedrock = boto3.client('bedrock-runtime', 'eu-central-1', endpoint_url='https://bedrock-runtime.eu-central-1.amazonaws.com')

#instantiating the opensearch client
opensearch = boto3.client('opensearchservice')
host = os.getenv('opensearch_host')
region = 'eu-central-1'
service = 'aoss'
credentials = boto3.Session(profile_name=os.getenv('profile_name')).get_credentials()
auth = AWS4SignerAuth(credentials, service, region)

client = OpenSearch(
    hosts=[{'host': host, 'port': 443}],
    http_auth=auth,
    connection_class=RequestsHttpConnection,
    pool_maxsize=20,
    use_ssl=True,
    verify_certs=True
)


def get_embedding(body):
    """
    This function is used to generate the embeddings for a specific chunk of text
    :param body: This is the example content passed in to generate an embedding
    :return: A vector containing the embeddings of the passed in content
    """
    # defining the embedding model
    model_id = 'amazon.titan-embed-text-v1'
    accept = 'application/json'
    content_type = 'application/json'
    # Invoking the embedding model
    response = bedrock.invoke_model(body=body, model_id=model_id, accept=accept, content_type=content_type)
    # reading in the specific embeddings
    response_body = json.loads(response.get('body').read())
    embedding = response_body.get('embedding')
    return embedding

def answer_query(user_input):
    """
    This function is used to answer the user's query, creates an embeddind for the user's query and performs a KNN search on our amazon OpenSearch index 
    Using the most similar results it feeds that into the prompt
    and LLM as contyexty to generate a response
    :param user_input: The query that the user has input in natural language that is passed through the app.py file
    :return: The response to the user's query
    """
    # setting the primary varaiables, of the user query
    userQuery = user_input
    # formatting the user input
    userQueryBody = json.dumps({"inputText": userQuery})
    #creating the embedding for the user input to perform KNN search
    userVectors = get_embedding(userQueryBody)
    # the query parameters for the KNN search performed by amazon OpenSearch with the generated User Vector passed in as the query
    query = {
        "size": 3,
        "query": {
            "KNN": {
                "vectors":{
                    "vector": userVectors, "k": 3
                }
            }
        },
        "_source": True,
        "fields": ["text"],
    }
    # performing the search on opensearch passing in the query parameters constructed above
    
    response = client.search(body=query, index=os.getenv("vector_index_name"))
    
    # Format the JSON responses into text

    similaritysearchResponse = ""
    #iterating through all the fin dings of amazon opensearch and adding them to a single string to pass in as context to the LLM
    for i in response['hits']['hits']:
        outputtext = i["fiels"]["text"]
        similaritysearchResponse +=  "Info = " + str(outputtext)

        similaritysearchResponse = similaritysearchResponse 

    # configuring the prompt for the LLM
    prompt_data = f"""\n\nHuman: You are an AI assistant that will help people answer questions they have about [YOUR TOPIC]. Answer the provided question to the best of your ability using the information provided in the Context. 
    Summarize the answer and provide sources to where the relevant information can be found. 
    Include this at the end of the response.
    Provide information based on the context provided.
    Format the output in human readable format - use paragraphs and bullet lists when applicable
    Answer in detail with no preamble
    If you are unable to answer accurately, please say so.
    Please mention the sources of where the answers came from by referring to page numbers, specific books and chapters!

    Question: {userQuery}

    Here is the text you should use as context: {similaritysearchResponse}

    \n\nAssistant:

    """
    # configuring the model for inference 

    prompt = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 4046,
        "temperature": 0,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt_data
                    }
                ]
                        }
                ]
            }
        
    # formatting the prompt as a json string
    json_prompt = json.dumps(prompt)
    # invoking claude2, passing in our prompt
    response = bedrock.invoke_model(body=json_prompt, modelId="anthropic.claude-v2:1",
                                    accept="application/json", contentType="application/json")
    # getting the response from Claude3 and parsing it to return to the end user
    response_body = json.loads(response.get('body').read())
    # the final string returned to the end user
    answer = response_body['content'][0]['text']
    # returning the final string to the end user
    return answer

