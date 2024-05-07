import pytest
import os
from dotenv import load_dotenv, find_dotenv

from llama_index.core import SimpleDirectoryReader, Settings
from llama_index.embeddings.mistralai import MistralAIEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.mistralai import MistralAI

import rag_assistant.utils.utilsrag_li
from rag_assistant.utils.utilsrag_li import create_automerging_engine, create_sentence_window_engine, create_subquery_engine, \
    create_direct_query_engine, create_li_agent

import shutil


import numpy as np

from trulens_eval.feedback import Groundedness
import nest_asyncio

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
    Tru
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
    tru_recorder = TruLlama(
        query_engine,
        app_id=app_id,
        feedbacks=feedbacks
    )
    return tru_recorder


@pytest.fixture(scope="module")
def temp_dir(request):
    # Setup: Create a temporary directory for the test module
    dir_name = rag_assistant.utils.utilsrag_li.llama_index_root_dir
    os.makedirs(dir_name, exist_ok=True)
    shutil.rmtree(dir_name)
    # Yield the directory name to the tests
    yield dir_name

    # Teardown: Remove the temporary directory after tests are done
    shutil.rmtree(dir_name)


@pytest.fixture
def llm_prepare():
    # llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
    # embed_model = OpenAIEmbedding()
    llm = MistralAI()
    embed_model = MistralAIEmbedding()

    # llm = MistralAI()
    Settings.llm = llm
    Settings.embed_model = embed_model

    return llm


@pytest.fixture
def docs_prepare():
    documents = SimpleDirectoryReader(
        input_files=["tests/utils/eBook-How-to-Build-a-Career-in-AI.pdf"]
    ).load_data()
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


def test_automerging_engine(temp_dir, llm_prepare, docs_prepare, eval_questions_prepare, trulens_prepare):

    query_engine = create_automerging_engine(docs_prepare)

    tru_recorder = get_prebuilt_trulens_recorder(query_engine,
                                                 app_id="Automerging Query Engine")

    with tru_recorder as recording:
        for question in eval_questions_prepare:
            response = query_engine.query(question)
            assert response is not None, "L'interprétation n'a pas retourné de résultat."


def test_automerging_agent(temp_dir, llm_prepare, docs_prepare, eval_questions_prepare, trulens_prepare):

    query_engine = create_automerging_engine(docs_prepare)

    agent = create_li_agent(name="test_automerging_agent", description="Test Automerging Agent",
                            query_engine=query_engine)

    response = agent.chat("How do I get started on a personal project in AI?")
    print(f"response: {str(response)}")
    assert response is not None, "L'interprétation n'a pas retourné de résultat."


def test_sentence_window_agent(temp_dir, llm_prepare, docs_prepare, eval_questions_prepare, trulens_prepare):

    query_engine = create_sentence_window_engine(
        docs_prepare,
    )

    tru_recorder = get_prebuilt_trulens_recorder(query_engine,
                                                 app_id="Sentence Window Query Engine")

    with tru_recorder as recording:
        for question in eval_questions_prepare:
            response = query_engine.query(question)

    print(trulens_prepare.get_leaderboard(app_ids=[]))

    agent = create_li_agent(name="test_sentence_window_agent", description="Test Sentence Window Agent",
                            query_engine=query_engine)

    response = agent.chat("How do I get started on a personal project in AI?")
    print(f"response: {str(response)}")
    assert response is not None, "L'interprétation n'a pas retourné de résultat."


def test_llamaindex_agent(temp_dir, llm_prepare, docs_prepare, eval_questions_prepare, trulens_prepare):

    query_engine = create_direct_query_engine(
        docs_prepare,
    )

    tru_recorder = get_prebuilt_trulens_recorder(query_engine,
                                                 app_id="Direct Query Engine")

    with tru_recorder as recording:
        for question in eval_questions_prepare:
            response = query_engine.query(question)

    print(trulens_prepare.get_leaderboard(app_ids=[]))

    agent = create_li_agent(name="test_direct_query_agent", description="Test Direct Query Agent", query_engine = query_engine)

    response = agent.chat("How do I get started on a personal project in AI?")
    print(f"response: {str(response)}")
    assert response is not None, "L'interprétation n'a pas retourné de résultat."


def test_subquery_agent(temp_dir, llm_prepare, docs_prepare, eval_questions_prepare, trulens_prepare):
    topics = ["AI", "Other"]
    query_engine = create_subquery_engine(
        topics,
        docs_prepare,
    )

    tru_recorder = get_prebuilt_trulens_recorder(query_engine,
                                                 app_id="Sub Query Engine")

    with tru_recorder as recording:
        for question in eval_questions_prepare:
            response = query_engine.query(question)

    print(trulens_prepare.get_leaderboard(app_ids=[]))

    agent = create_li_agent(name="test_subquery_agent", description="Test Subquery Agent",
                            query_engine=query_engine)

    response = agent.chat("How do I get started on a personal project in AI?")
    print(f"response: {str(response)}")
    assert response is not None, "L'interprétation n'a pas retourné de résultat."

