import pytest
import os
from dotenv import load_dotenv, find_dotenv
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings

import rag_assistant.utils.utilsrag_li

import shutil


import numpy as np

from trulens_eval.feedback import Groundedness
import nest_asyncio

from rag_assistant.utils.utilsrag_lc import agent_lc_factory

load_dotenv(find_dotenv())

# Set OpenAI API key from Streamlit secrets
openai_api_key = os.getenv('OPENAI_API_KEY')

nest_asyncio.apply()


def get_openai_api_key():
    _ = load_dotenv(find_dotenv())

    return os.getenv("OPENAI_API_KEY")


def get_hf_api_key():
    _ = load_dotenv(find_dotenv())

    return os.getenv("HUGGINGFACE_API_KEY")


from trulens_eval import (
    Feedback,
    TruLlama,
    OpenAI,
    Tru, TruChain
)

openai = OpenAI()

qa_relevance = (
    Feedback(openai.relevance_with_cot_reasons, name="Answer Relevance")
    .on_input_output()
)

qs_relevance = (
    Feedback(openai.relevance_with_cot_reasons, name="Context Relevance")
    .on_input()
    .on(TruLlama.select_source_nodes().node.text)
    .aggregate(np.mean)
)

grounded = Groundedness(groundedness_provider=openai)

groundedness = (
    Feedback(grounded.groundedness_measure_with_cot_reasons, name="Groundedness")
        .on(TruLlama.select_source_nodes().node.text)
        .on_output()
        .aggregate(grounded.grounded_statements_aggregator)
)

feedbacks = [qa_relevance, qs_relevance, groundedness]


def get_prebuilt_trulens_recorder(query_engine, app_id):
    tru_recorder = TruChain(
        query_engine,
        app_id=app_id,
        feedbacks=feedbacks
    )
    return tru_recorder


# @pytest.fixture(scope="module")
def temp_dir(request):
    # Setup: Create a temporary directory for the test module
    # TODO should do something with the vectordb
    pass


@pytest.fixture
def llm_prepare():
    # llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
    llm = ChatMistralAI()

    return llm


@pytest.fixture
def embeddings_prepare():
    # embed_model = OpenAIEmbeddings()
    embed_model = MistralAIEmbeddings()

    return embed_model


@pytest.fixture
def docs_prepare():
    loader =  PyPDFLoader("tests/utils/eBook-How-to-Build-a-Career-in-AI.pdf")
    documents = loader.load()
    return documents


@pytest.fixture
def eval_questions_prepare():
    eval_questions = []
    with open('tests/utils/eval_questions.txt', 'r') as file:
        for line in file:
            # Remove newline character and convert to integer
            item = line.strip()
            print(item)
            eval_questions.append(item)
    return eval_questions


@pytest.fixture(scope="module")
def trulens_prepare():
    tru = Tru()
    # tru.reset_database()
    return tru


def test_lc_agent_stuff_4_similarity(llm_prepare, embeddings_prepare, docs_prepare, eval_questions_prepare, trulens_prepare):

    retrieval_qa_chain = agent_lc_factory(docs_prepare, chain_type="stuff", embeddings=embeddings_prepare,
                                          llm=llm_prepare,
                                          search_kwargs={"k": 4},
                                          search_type="similarity")

    response = retrieval_qa_chain("How do I get started on a personal project in AI?")
    print(f"response: {str(response)}")
    assert response is not None, "L'interprétation n'a pas retourné de résultat."