from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_openai import AzureChatOpenAI, ChatOpenAI
import os
from dotenv import load_dotenv, find_dotenv
import openai
from mistralai.client import MistralClient

from .config_loader import load_config

from langchain_openai.embeddings import OpenAIEmbeddings, AzureOpenAIEmbeddings
from openai import AzureOpenAI
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings


config = load_config()

# read local .env file
_ = load_dotenv(find_dotenv())

openai.api_key = os.environ['OPENAI_API_KEY']
mistral_api_key = os.environ.get("MISTRAL_API_KEY")


def load_model(model_name: str = None, temperature: float = 0) -> BaseChatModel:

    model = None
    if model_name is None:
        model = config['LLM']['LLM_MODEL']
    elif model_name.startswith("gpt"):
        model = "OPENAI"
    elif model_name.startswith("mistral"):
        model = "MISTRAL"

    if model == "AZURE":
        llm = AzureChatOpenAI(
            openai_api_version=config['AZURE']['AZURE_OPENAI_API_VERSION'],
            azure_endpoint=config['AZURE']['AZURE_OPENAI_ENDPOINT'],
            azure_deployment=config['AZURE']['AZURE_OPENAI_DEPLOYMENT'],
            api_key=os.environ["AZURE_OPENAI_API_KEY"]
        )
    elif model == "OPENAI":
        if model_name is None:
            model_name = config['OPENAI']['OPENAI_MODEL_NAME']
        llm = ChatOpenAI(model_name=model_name, temperature=temperature)
    elif model == "MISTRAL":
        if model_name is None:
            model_name = config['MISTRAL']['CHAT_MODEL']
        llm = ChatMistralAI(mistral_api_key=mistral_api_key, model=model_name, temperature=temperature)
    else:
        raise NotImplementedError(f"Model {model} unknown.")

    return llm


def load_client():
    model = config['LLM']['LLM_MODEL']
    if model == "AZURE":
        client = AzureOpenAI(
            api_version=config['AZURE']['AZURE_OPENAI_API_VERSION'],
            azure_endpoint=config['AZURE']['AZURE_OPENAI_ENDPOINT'],
            azure_deployment=config['AZURE']['AZURE_OPENAI_DEPLOYMENT'],
            api_key=os.environ["AZURE_OPENAI_API_KEY"]
        )
    elif model == "MISTRAL":
        client = MistralClient(api_key=mistral_api_key)
    else:
        raise NotImplementedError(f"{model} chat client not done")

    return client


def load_embeddings(model_name: str = None) -> Embeddings:

    model = None
    if model_name is None:
        model = config['LLM']['LLM_MODEL']
    elif model_name.startswith("gpt"):
        model = "OPENAI"
    elif model_name.startswith("mistral"):
        model = "MISTRAL"

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
    else:
        embeddings = OpenAIEmbeddings()

    return embeddings
