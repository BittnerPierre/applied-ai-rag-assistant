from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI, AzureChatOpenAI
import os
from dotenv import load_dotenv, find_dotenv
import openai
from mistralai.client import MistralClient

from .config_loader import load_config

from langchain_openai.embeddings import OpenAIEmbeddings, AzureOpenAIEmbeddings
from openai import AzureOpenAI
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.embeddings.bedrock import BedrockEmbeddings
from langchain_aws import BedrockLLM
import boto3

config = load_config()

# read local .env file
_ = load_dotenv(find_dotenv())

openai.api_key = os.environ['OPENAI_API_KEY']
mistral_api_key = os.environ.get("MISTRAL_API_KEY")

# read local .env file
_ = load_dotenv(find_dotenv())

aws_profile_name = os.getenv("profile_name")
aws_region_name = config["BEDROCK"]["AWS_REGION_NAME"]
bedrock_endpoint_url = config["BEDROCK"]["BEDROCK_ENDPOINT_URL"]

# instantiating the Bedrock client, and passing in the CLI profile
boto3.setup_default_session(profile_name=aws_profile_name)
bedrock = boto3.client('bedrock-runtime', aws_region_name,
                       endpoint_url=bedrock_endpoint_url)

model_kwargs = {
    #"maxTokenCount": 4096,
    #"stopSequences": [],
    "temperature": 0,
    #"topP": 1,
}


def load_model(model_name: str = None, temperature: float = 0, streaming:bool = False) -> BaseChatModel:

    provider = None
    if model_name is None:
        provider = config['MODEL_PROVIDER']['MODEL_PROVIDER']
    elif model_name.startswith("gpt"):
        provider = "OPENAI"
    elif model_name.startswith("mistral"):
        provider = "MISTRAL"
    elif model_name.startswith("anthropic"):
        provider = "BEDROCK"

    if provider == "AZURE":
        llm = AzureChatOpenAI(
            openai_api_version=config['AZURE']['AZURE_OPENAI_API_VERSION'],
            azure_endpoint=config['AZURE']['AZURE_OPENAI_ENDPOINT'],
            azure_deployment=config['AZURE']['AZURE_OPENAI_DEPLOYMENT'],
            api_key=os.environ["AZURE_OPENAI_API_KEY"]
        )
    elif provider == "OPENAI":
        if model_name is None:
            model_name = config['OPENAI']['OPENAI_MODEL_NAME']
        llm = ChatOpenAI(model_name=model_name, temperature=temperature, streaming = streaming)
    elif provider == "MISTRAL":
        if model_name is None:
            model_name = config['MISTRAL']['CHAT_MODEL']
        llm = ChatMistralAI(mistral_api_key=mistral_api_key, model=model_name, temperature=temperature)
    elif provider == "BEDROCK":
        if model_name is None:
            model_name = config['BEDROCK']['CHAT_MODEL']

        llm = BedrockLLM(
            client=bedrock,
            model_id=model_name,
            # model_kwargs=model_kwargs,
            streaming=streaming,
            #callbacks=[StreamingStdOutCallbackHandler()],
        )

    else:
        raise NotImplementedError(f"Model {provider} unknown.")

    return llm


def load_client():
    model = config['MODEL_PROVIDER']['MODEL_PROVIDER']
    if model == "AZURE":
        client = AzureOpenAI(
            api_version=config['AZURE']['AZURE_OPENAI_API_VERSION'],
            azure_endpoint=config['AZURE']['AZURE_OPENAI_ENDPOINT'],
            azure_deployment=config['AZURE']['AZURE_OPENAI_DEPLOYMENT'],
            api_key=os.environ["AZURE_OPENAI_API_KEY"]
        )
    elif model == "MISTRAL":
        client = MistralClient(api_key=mistral_api_key)
    elif model == "BEDROCK":
        bedrock = BedrockLLM()
        client = bedrock
    else:
        raise NotImplementedError(f"{model} chat client not done")

    return client


def load_embeddings(model_name: str = None) -> Embeddings:

    model = None
    if model_name is None:
        model = config['MODEL_PROVIDER']['MODEL_PROVIDER']
    elif model_name.startswith("gpt"):
        model = "OPENAI"
    elif model_name.startswith("mistral"):
        model = "MISTRAL"
    elif model_name.startswith("anthropic"):
        model = "BEDROCK"

    if model == "AZURE":
        embeddings = AzureOpenAIEmbeddings(
            azure_deployment=config['AZURE']['AZURE_OPENAI_EMBEDDING_DEPLOYMENT'],
            azure_endpoint=config['AZURE']['AZURE_OPENAI_ENDPOINT'],
            openai_api_version=config['AZURE']["AZURE_OPENAI_API_VERSION"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"]
        )
    elif model == "OPENAI":
        embeddings = OpenAIEmbeddings()
    elif model == "MISTRAL":
        embeddings = MistralAIEmbeddings()
    elif model == "BEDROCK":
        embeddings = BedrockEmbeddings(credentials_profile_name=aws_profile_name, region_name=aws_region_name)
    else:
        embeddings = OpenAIEmbeddings()

    return embeddings
