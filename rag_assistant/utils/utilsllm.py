import boto3
import openai
import os

from typing import Optional

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings, AzureOpenAIEmbeddings
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_aws.embeddings.bedrock import BedrockEmbeddings
from langchain_aws import ChatBedrock
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.llms import LLM

from llama_index.embeddings.mistralai import MistralAIEmbedding as LIMistralAIEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding as LIOpenAIEmbedding
from llama_index.embeddings.bedrock import BedrockEmbedding as LIBedrockEmbedding
from llama_index.llms.mistralai import MistralAI as LIMistralAI
from llama_index.llms.openai import OpenAI as LIOpenAI
from llama_index.llms.bedrock import Bedrock as LIBedrock


from dotenv import load_dotenv, find_dotenv

from .config_loader import load_config

config = load_config()

# read local .env file
_ = load_dotenv(find_dotenv())

openai.api_key = os.getenv('OPENAI_API_KEY')
mistral_api_key = os.getenv('MISTRAL_API_KEY')
aws_profile_name = os.getenv('profile_name')

bedrock_region_name = config["BEDROCK"]["AWS_REGION_NAME"]
bedrock_embeddings_model = config["BEDROCK"]["EMBEDDINGS_MODEL"]
bedrock_endpoint_url = config["BEDROCK"]["BEDROCK_ENDPOINT_URL"]

# instantiating the Bedrock client, and passing in the CLI profile
bedrock = boto3.client('bedrock-runtime', bedrock_region_name,
                       endpoint_url=bedrock_endpoint_url)

model_kwargs = {
    #"maxTokenCount": 4096,
    #"stopSequences": [],
    "temperature": 0,
    #"topP": 1,
}


def get_model_provider(model_name:str) -> Optional[str]:
    provider = None
    if model_name is None:
        provider = config['MODEL_PROVIDER']['MODEL_PROVIDER']
    elif model_name.startswith("gpt"):
        provider = "OPENAI"
    elif model_name.startswith("mistral-"):
        provider = "MISTRAL"
    elif model_name.startswith("mistral.mi"):
        provider = "BEDROCK"
    elif model_name.startswith("anthropic"):
        provider = "BEDROCK"

    return provider

def load_model(model_name: str = None, temperature: float = 0, streaming:bool = False) -> BaseChatModel:

    provider = get_model_provider(model_name)

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
        llm = ChatOpenAI(model_name=model_name, temperature=temperature, streaming=streaming)
    elif provider == "MISTRAL":
        if model_name is None:
            model_name = config['MISTRAL']['CHAT_MODEL']
        llm = ChatMistralAI(mistral_api_key=mistral_api_key, model=model_name, temperature=temperature)
    elif provider == "BEDROCK":
        if model_name is None:
            model_name = config['BEDROCK']['CHAT_MODEL']

        # ChatBedrock --> must be adapted for system prompt get error "first message must use the "user" role"
        # temperature not supported
        llm = ChatBedrock(
            client=bedrock,
            model_id=model_name,
            streaming=streaming
        )

    else:
        raise NotImplementedError(f"Model {provider} unknown.")

    return llm


def load_llamaindex_model(model_name: str = None, temperature: float = 0) -> LLM:

    provider = get_model_provider(model_name)

    if provider == "AZURE":
        raise NotImplementedError(f"Model {provider} unsupported for LlamaIndex.")
    elif provider == "OPENAI":
        if model_name is None:
            model_name = config['OPENAI']['OPENAI_MODEL_NAME']
        llm = LIOpenAI(model=model_name, temperature=temperature)
    elif provider == "MISTRAL":
        if model_name is None:
            model_name = config['MISTRAL']['CHAT_MODEL']
        llm = LIMistralAI(api_key=mistral_api_key, model=model_name, temperature=temperature)
    elif provider == "BEDROCK":
        if model_name is None:
            model_name = config['BEDROCK']['CHAT_MODEL']

        # ChatBedrock --> must be adapted for system prompt get error "first message must use the "user" role"
        llm = LIBedrock(
            client=bedrock,
            model=model_name,
            temperature=temperature
        )

    else:
        raise NotImplementedError(f"Model {model_name} unknown.")

    return llm


def load_embeddings(model_name: str = None) -> Embeddings:

    provider = get_model_provider(model_name)

    if provider == "AZURE":
        embeddings = AzureOpenAIEmbeddings(
            azure_deployment=config['AZURE']['AZURE_OPENAI_EMBEDDING_DEPLOYMENT'],
            azure_endpoint=config['AZURE']['AZURE_OPENAI_ENDPOINT'],
            openai_api_version=config['AZURE']["AZURE_OPENAI_API_VERSION"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"]
        )
    elif provider == "OPENAI":
        embeddings = OpenAIEmbeddings()
    elif provider == "MISTRAL":
        embeddings = MistralAIEmbeddings()
    elif provider == "BEDROCK":
        embeddings = BedrockEmbeddings(
            client=bedrock,
            region_name=bedrock_region_name,
            model_id=bedrock_embeddings_model)
    else:
        raise NotImplementedError(f"Model {model_name} unknown.")

    return embeddings


def load_llamaindex_embeddings(model_name: str = None) -> BaseEmbedding:

    provider = get_model_provider(model_name)

    if provider == "AZURE":
        raise NotImplementedError(f"Embeddings {provider} unsupported for LlamaIndex.")
    elif provider == "OPENAI":
        embeddings = LIOpenAIEmbedding()
    elif provider == "MISTRAL":
        embeddings = LIMistralAIEmbedding()
    elif provider == "BEDROCK":
        embeddings = LIBedrockEmbedding(
            region_name=bedrock_region_name,
            model_name=bedrock_embeddings_model,
            client=bedrock)
    else:
        raise NotImplementedError(f"Model {model_name} unknown.")

    return embeddings
