from langchain_openai import AzureChatOpenAI, ChatOpenAI
import os
import sys
from dotenv import load_dotenv, find_dotenv
import openai
# import mistralai

from .config_loader import load_config

from langchain_openai.embeddings import OpenAIEmbeddings, AzureOpenAIEmbeddings
from openai import AzureOpenAI
# from langchain_mistralai.chat_models import ChatMistralAI
# from mistralai.client import MistralClient


config = load_config()

# read local .env file
_ = load_dotenv(find_dotenv())

openai.api_key = os.environ['OPENAI_API_KEY']
mistral_api_key = os.environ.get("MISTRAL_API_KEY")


def load_model(model: str = None):
    model = config['LLM']['LLM_MODEL']
    print(f"Model Type: {model}")
    if model == "AZURE":
        llm = AzureChatOpenAI(
            openai_api_version=config['AZURE']['AZURE_OPENAI_API_VERSION'],
            azure_endpoint=config['AZURE']['AZURE_OPENAI_ENDPOINT'],
            azure_deployment=config['AZURE']['AZURE_OPENAI_DEPLOYMENT'],
            api_key=os.environ["AZURE_OPENAI_API_KEY"]
        )
    elif model == "OPENAI":
        model_name = config['OPENAI']['OPEN_AI_MODEL_NAME']
        llm = ChatOpenAI(model_name=model_name, temperature=0)
    elif model == "MISTRAL":
        # chat = ChatMistralAI(mistral_api_key=mistral_api_key)
        raise NotImplementedError(f"{model} Model not implemented yet.")
    else:
        raise NotImplementedError(f"Model {model} unknown.")

    return llm


def load_client(model: str = None):
    model = config['LLM']['LLM_MODEL']
    print(f"Model Type: {model}")
    if model == "AZURE":
        client = AzureOpenAI(
            api_version=config['AZURE']['AZURE_OPENAI_API_VERSION'],
            azure_endpoint=config['AZURE']['AZURE_OPENAI_ENDPOINT'],
            azure_deployment=config['AZURE']['AZURE_OPENAI_DEPLOYMENT'],
            api_key=os.environ["AZURE_OPENAI_API_KEY"]
        )
    elif model == "MISTRAL":
        raise NotImplementedError(f"{model} chat client not done")
        # client = MistralClient(api_key=mistral_api_key)
    else:
        raise NotImplementedError(f"{model} chat client not done")

    return client




def load_embeddings():
    model = config['LLM']['LLM_MODEL']
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
        raise NotImplementedError(f"{model} chat client not done")
    else:
        embeddings = OpenAIEmbeddings()

    return embeddings