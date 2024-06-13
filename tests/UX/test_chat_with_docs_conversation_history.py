import pytest
from streamlit.testing.v1 import AppTest
import os
import sys
from unittest.mock import patch, MagicMock
import datetime
import streamlit as st

# Mock the environment variables and imported functions
os.environ['OPENAI_API_KEY'] = 'test_key'

# Mock the load_config function to provide a test configuration
def mock_load_config():
    return {
        'VECTORDB': {
            'collection_name': 'test_collection'
        },
        'DEFAULT': {
            'APP_NAME': 'My LLM+RAG Assistant'
        }
    }

# Mock the get_store function to avoid actual database calls
def mock_get_store():
    return MagicMock()

# Mock the load_model function to avoid actual model loading
def mock_load_model(streaming=False):
    return MagicMock()

sys.modules['utils.config_loader'] = MagicMock(load_config=mock_load_config)
sys.modules['utils.utilsdoc'] = MagicMock(get_store=mock_get_store)
sys.modules['utils.utilsllm'] = MagicMock(load_model=mock_load_model)

def mock_llm_response(*args, **kwargs):
    # Mock LLM response
    return {
        "answer": "To secure sensitive data, you can use encryption, access controls, and regular audits."
    }



def test_new_chat_session():
    at = AppTest.from_file("/Users/loicsteve/Desktop/applied-ai-rag-assistant/rag_assistant/Hello.py")
    at.run(timeout=10)  # Increase the timeout to 10 seconds
    assert not at.exception
    assert at.title[0].value == "Welcome Page"
    at.switch_page("pages/0_Chat_with_Docs.py").run()
    assert not at.exception
    #print(at.sidebar.button)
    assert at.sidebar.button[0].label == 'New Chat'
    at.sidebar.button[0].click().run()
    # Check that a new chat session is created
    assert len(at.session_state.chat_histories) == 2
 
@patch('langchain_core.runnables.history.RunnableWithMessageHistory.invoke', side_effect=mock_llm_response)
def test_chat_with_docs(mock_load_config):
    at = AppTest.from_file("/Users/loicsteve/Desktop/applied-ai-rag-assistant/rag_assistant/pages/0_Chat_with_Docs.py")
    at.run(timeout=10)
    assert not at.exception
 
    # Create a test session ID and mock chat history
    session_id =  "test_session_id"
    st.session_state["session_id"] = session_id
    st.session_state["chat_histories"] = {
        session_id: [
            {"type": "user", "content": "How to secure sensitive data?"},
            {"type": "ai", "content": "To secure sensitive data, you can use encryption, access controls, and regular audits."}
        ]
    }
 
    at.chat_input[0].set_value("How to secure sensitive data?").run()
    assert at.chat_message[0].markdown[0].value == "How to secure sensitive data?"
    assert at.chat_message[1].markdown[0].value == "To secure sensitive data, you can use encryption, access controls, and regular audits."
    assert at.chat_message[1].avatar == "assistant"
    assert not at.exception
    print(at.columns)
    assert at.columns[3].button[0].label == '🚮'
    at.sidebar.button[0].click().run()
    


    

# def test_delete_chat_session():
#     at = AppTest.from_file("/Users/loicsteve/Desktop/applied-ai-rag-assistant/rag_assistant/pages/0_Chat_with_Docs.py")
#     at.run()
#     assert not at.exception
#     # session_id =  "test_session_id"
#     # st.session_state["session_id"] = session_id
#     print(at.columns)
#     assert at.button[1].label == '🚮'
    # at.button[1].click().run()
    # assert len(at.session_state.chat_histories) == 0
    #delete_button = at.sidebar.button("Delete")
    # assert delete_button is not None
    # delete_button.click().run()
    # # Check that the chat session is deleted
    # assert len(at.session_state.chat_histories) == 0

