import pytest
import os
from dotenv import load_dotenv, find_dotenv

from trulens_eval import TruChain

from rag_assistant.utils.utilsrag_lc import agent_lc_factory

import numpy as np
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.vectorstores.chroma import Chroma
from langchain_aws import BedrockLLM, ChatBedrock
from langchain_aws.embeddings.bedrock import BedrockEmbeddings

import nest_asyncio

import boto3

from trulens_eval import (
    Feedback,
    TruLlama,
    OpenAI,
    Tru, Select
)

from trulens_eval.app import App

load_dotenv(find_dotenv())

# Set OpenAI API key from Streamlit secrets
openai_api_key = os.getenv('OPENAI_API_KEY')

nest_asyncio.apply()

aws_profile_name = os.getenv("profile_name")
test_name = "Claude_2"
aws_region_name = "eu-central-1"
model_name = "anthropic.claude-v2:1"
bedrock_endpoint_url = "https://bedrock-runtime.eu-central-1.amazonaws.com"
embedded_model_id = "amazon.titan-embed-text-v1"
provider = OpenAI()


def get_openai_api_key():
    _ = load_dotenv(find_dotenv())

    return os.getenv("OPENAI_API_KEY")


def get_hf_api_key():
    _ = load_dotenv(find_dotenv())

    return os.getenv("HUGGINGFACE_API_KEY")


@pytest.fixture()
def qa_chain_prepare(llm_prepare, docs_prepare, embeddings_prepare):

    db = Chroma.from_documents(
        documents=docs_prepare,
        embedding=embeddings_prepare,
        collection_name=f"Test_RAG_bedrock_{test_name}",
    )

    qa_chain = agent_lc_factory(chain_type="stuff",
                                llm=llm_prepare,
                                search_kwargs={"k": 4},
                                search_type="mmr", vectorstore=db)
    return qa_chain


@pytest.fixture()
def prepare_feedbacks(qa_chain_prepare):

    context = App.select_context(qa_chain_prepare)

    qa_relevance = (
        Feedback(provider.relevance_with_cot_reasons, name="Answer Relevance")
        .on_input_output()
    )

    qs_relevance = (
        Feedback(provider.context_relevance_with_cot_reasons, name="Context Relevance")
        .on_input()
        .on(context)
        .aggregate(np.mean)
    )

    groundedness = (
        Feedback(provider.groundedness_measure_with_cot_reasons, name = "Groundedness")
        .on(context.collect())
        .on_output()
    )

    feedbacks = [qa_relevance, qs_relevance, groundedness]
    return feedbacks


def get_prebuilt_trulens_recorder(chain, app_id, feedbacks):
    tru_recorder = TruChain(
        chain,
        app_id=app_id,
        feedbacks=feedbacks
    )
    return tru_recorder


@pytest.fixture(scope="module")
def temp_dir(request):
    # Setup: Create a temporary directory for the test module
    # TODO should do something with the vectordb
    pass


# instantiating the Bedrock client, and passing in the CLI profile
@pytest.fixture(scope="module")
def prepare_bedrock():
    boto3.setup_default_session(profile_name=aws_profile_name)
    bedrock = boto3.client('bedrock-runtime',
                           region_name=aws_region_name,
                           )
    return bedrock


@pytest.fixture
def llm_prepare(prepare_bedrock):
    llm = ChatBedrock(
        client=prepare_bedrock,
        model_id=model_name,
        streaming=False,
    )

    return llm


@pytest.fixture
def embeddings_prepare(prepare_bedrock):
    embed_model = BedrockEmbeddings(
        client=prepare_bedrock,
        model_id=embedded_model_id
    )

    return embed_model


@pytest.fixture
def docs_prepare():
    documents = []
    loader = PyPDFLoader("tests/rag/eval_document.pdf")
    documents.extend(loader.load())
    return documents


@pytest.fixture
def eval_questions_prepare():
    eval_questions = []
    with open('tests/rag/eval_questions.txt', 'r') as file:
        for line in file:
            # Remove newline character and convert to integer
            item = line.strip()
            print(item)
            eval_questions.append(item)
    return eval_questions


def test_qa_chain(qa_chain_prepare, eval_questions_prepare, trulens_prepare, prepare_feedbacks):

    tru_recorder = get_prebuilt_trulens_recorder(chain=qa_chain_prepare,
                                                 app_id=f"Retrieval QA Chain ({test_name})",
                                                 feedbacks=prepare_feedbacks)

    with tru_recorder as recording:
        for question in eval_questions_prepare:
            print(f"question: {str(question)}")
            response = qa_chain_prepare.invoke(question)
            assert response is not None, "L'interprétation n'a pas retourné de résultat."
            print(f"response: {str(response)}")
