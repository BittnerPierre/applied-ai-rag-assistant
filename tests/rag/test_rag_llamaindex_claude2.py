import pytest
import os
from dotenv import load_dotenv, find_dotenv

from llama_index.core import SimpleDirectoryReader, Settings
from llama_index.embeddings.bedrock import BedrockEmbedding
from llama_index.llms.bedrock import Bedrock

import rag_assistant.utils.utilsrag_lc

import shutil

import numpy as np
import nest_asyncio

import rag_assistant.utils.utilsrag_li
from rag_assistant.utils.utilsrag_li import create_sentence_window_engine
from trulens_eval.app import App

import boto3
from trulens_eval import (
    Feedback,
    TruLlama,
    OpenAI,
    Tru, Select
)

load_dotenv(find_dotenv())

# Set OpenAI API key from Streamlit secrets
openai_api_key = os.getenv('OPENAI_API_KEY')

nest_asyncio.apply()

aws_profile_name = os.getenv("profile_name")
test_name = "Claude_2_LlamaIndex"
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
def query_engine(llm_prepare, docs_prepare, embeddings_prepare):
    query_engine = create_sentence_window_engine(
        docs_prepare,
    )
    return query_engine


@pytest.fixture()
def prepare_feedbacks(query_engine):

    context = App.select_context(query_engine)

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


def get_prebuilt_trulens_recorder(query_engine, app_id, feedbacks):
    tru_recorder = TruLlama(
        query_engine,
        app_id=app_id,
        feedbacks=feedbacks
    )
    return tru_recorder


@pytest.fixture(scope="module")
def temp_dir(request):
    dir_name = rag_assistant.utils.utilsrag_li.llama_index_root_dir
    os.makedirs(dir_name, exist_ok=True)
    shutil.rmtree(dir_name)
    # Yield the directory name to the tests
    yield dir_name

    # Teardown: Remove the temporary directory after tests are done
    shutil.rmtree(dir_name, ignore_errors=True)
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
    llm = Bedrock(
        client=prepare_bedrock,
        model=model_name,
    )

    Settings.llm = llm
    return llm


@pytest.fixture
def embeddings_prepare(prepare_bedrock):
    embed_model = BedrockEmbedding(
        client=prepare_bedrock,
        model_name=embedded_model_id
    )
    Settings.embed_model = embed_model
    return embed_model


@pytest.fixture
def docs_prepare():
    documents = SimpleDirectoryReader(
        input_files=["tests/rag/eval_document.pdf",
                     ]
    ).load_data()
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


def test_sentence_window(temp_dir, llm_prepare, docs_prepare, eval_questions_prepare,
                         trulens_prepare, prepare_feedbacks, query_engine):

    tru_recorder = get_prebuilt_trulens_recorder(query_engine,
                                                 app_id=f"Sentence Window Query Engine ({test_name})",
                                                 feedbacks=prepare_feedbacks)

    with tru_recorder as recording:
        for question in eval_questions_prepare:
            response = query_engine.query(question)
            print(f"response: {str(response)}")
            assert response is not None, "L'interprétation n'a pas retourné de résultat."