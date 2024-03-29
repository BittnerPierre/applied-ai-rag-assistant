from typing import Sequence

import pytest
import os
from dotenv import load_dotenv, find_dotenv

from llama_index.core import SimpleDirectoryReader, Settings, StorageContext, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.mistralai import MistralAI
from llama_index.llms.openai import OpenAI

from rag_assistant.utils.utilsrag import create_subquery_agent, create_automerging_agent, create_sentence_window_agent, \
    create_direct_query_agent, create_automerging_engine, create_lli_agent, create_direct_query_engine, \
    create_subquery_engine, create_sentence_window_engine

load_dotenv(find_dotenv())

# Set OpenAI API key from Streamlit secrets
openai_api_key = os.getenv('OPENAI_API_KEY')


import numpy as np
from trulens_eval import (
    Feedback,
    TruLlama,
    OpenAI, Tru
)

from trulens_eval.feedback import Groundedness
import nest_asyncio

nest_asyncio.apply()


def get_openai_api_key():
    _ = load_dotenv(find_dotenv())

    return os.getenv("OPENAI_API_KEY")


def get_hf_api_key():
    _ = load_dotenv(find_dotenv())

    return os.getenv("HUGGINGFACE_API_KEY")


openai = OpenAI()

qa_relevance = (
    Feedback(openai.relevance_with_cot_reasons, name="Answer Relevance")
    .on_input_output()
)

qs_relevance = (
    Feedback(openai.relevance_with_cot_reasons, name = "Context Relevance")
    .on_input()
    .on(TruLlama.select_source_nodes().node.text)
    .aggregate(np.mean)
)

#grounded = Groundedness(groundedness_provider=openai, summarize_provider=openai)
grounded = Groundedness(groundedness_provider=openai)

groundedness = (
    Feedback(grounded.groundedness_measure_with_cot_reasons, name="Groundedness")
        .on(TruLlama.select_source_nodes().node.text)
        .on_output()
        .aggregate(grounded.grounded_statements_aggregator)
)

feedbacks = [qa_relevance, qs_relevance, groundedness]


def get_trulens_recorder(query_engine, feedbacks, app_id):
    tru_recorder = TruLlama(
        query_engine,
        app_id=app_id,
        feedbacks=feedbacks
    )
    return tru_recorder


def get_prebuilt_trulens_recorder(query_engine, app_id):
    tru_recorder = TruLlama(
        query_engine,
        app_id=app_id,
        feedbacks=feedbacks
    )
    return tru_recorder


from llama_index.llms.openai import OpenAI


@pytest.fixture
def llm_prepare():
    llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
    # llm = MistralAI()
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



@pytest.fixture
def trulens_prepare():
    tru = Tru()
    tru.reset_database()
    return tru


def test_automerging_agent(llm_prepare, docs_prepare, eval_questions_prepare, trulens_prepare):

    query_engine = create_automerging_engine(llm_prepare, docs_prepare, embed_model="local:BAAI/bge-small-en-v1.5")

    tru_recorder = get_prebuilt_trulens_recorder(query_engine,
                                                 app_id="Automerging Query Engine")

    with tru_recorder as recording:
        for question in eval_questions_prepare:
            response = query_engine.query(question)

    print(trulens_prepare.get_leaderboard(app_ids=[]))

    agent = create_lli_agent(name="test_automerging_agent", description="Test Automerging Agent",
                             query_engine=query_engine)

    response = agent.chat("What are steps to take when finding projects to build your experience?")
    print(f"response: {str(response)}")
    assert response is not None, "L'interprétation n'a pas retourné de résultat."


def test_sentence_window_agent(llm_prepare, docs_prepare, eval_questions_prepare, trulens_prepare):

    query_engine = create_sentence_window_engine(llm_prepare, docs_prepare, embed_model="local:BAAI/bge-small-en-v1.5")

    tru_recorder = get_prebuilt_trulens_recorder(query_engine,
                                                 app_id="Sentence Window Query Engine")

    with tru_recorder as recording:
        for question in eval_questions_prepare:
            response = query_engine.query(question)

    print(trulens_prepare.get_leaderboard(app_ids=[]))

    agent = create_lli_agent(name="test_sentence_window_agent", description="Test Sentence Window Agent",
                             query_engine=query_engine)

    response = agent.chat("What are steps to take when finding projects to build your experience?")
    print(f"response: {str(response)}")
    assert response is not None, "L'interprétation n'a pas retourné de résultat."


def test_llamaindex_agent(llm_prepare, docs_prepare, eval_questions_prepare, trulens_prepare):

    query_engine = create_direct_query_engine(llm_prepare, docs_prepare, embed_model="local:BAAI/bge-small-en-v1.5")

    tru_recorder = get_prebuilt_trulens_recorder(query_engine,
                                                 app_id="Direct Query Engine")

    with tru_recorder as recording:
        for question in eval_questions_prepare:
            response = query_engine.query(question)

    print(trulens_prepare.get_leaderboard(app_ids=[]))

    agent = create_lli_agent(name="test_direct_query_agent", description="Test Direct Query Agent", query_engine = query_engine)

    response = agent.chat("What are steps to take when finding projects to build your experience?")
    print(f"response: {str(response)}")
    assert response is not None, "L'interprétation n'a pas retourné de résultat."


def test_subquery_agent(llm_prepare, docs_prepare, eval_questions_prepare, trulens_prepare):
    topics = ["AI"]
    query_engine = create_subquery_engine(llm_prepare, topics, docs_prepare, embed_model="local:BAAI/bge-small-en-v1.5")

    tru_recorder = get_prebuilt_trulens_recorder(query_engine,
                                                 app_id="Sub Query Engine")

    with tru_recorder as recording:
        for question in eval_questions_prepare:
            response = query_engine.query(question)

    print(trulens_prepare.get_leaderboard(app_ids=[]))

    agent = create_lli_agent(name="test_subquery_agent", description="Test Subquery Agent",
                             query_engine=query_engine)

    response = agent.chat("What are steps to take when finding projects to build your experience?")
    print(f"response: {str(response)}")
    assert response is not None, "L'interprétation n'a pas retourné de résultat."

