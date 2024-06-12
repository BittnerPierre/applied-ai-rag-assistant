import pytest
from unittest.mock import patch, MagicMock
import streamlit as st
from .Chat_with_Docs import get_chat_history, get_session_id, handle_assistant_response

@pytest.fixture
def setup_streamlit_session():
    st.session_state.clear()
    get_session_id()

def test_get_session_id(setup_streamlit_session):
    session_id = get_session_id()
    assert "session_id" in st.session_state
    assert st.session_state.session_id == session_id

def test_get_chat_history(setup_streamlit_session):
    session_id = get_session_id()
    chat_history = get_chat_history(session_id)
    assert session_id in st.session_state.chat_histories
    assert isinstance(chat_history, StreamlitChatMessageHistory)

@patch('Chat_with_Docs.conversational_rag_chain.invoke')
def test_handle_assistant_response(mock_invoke, setup_streamlit_session):
    mock_response = {"answer": "Mocked response"}
    mock_invoke.return_value = mock_response

    user_query = "Test question"
    session_id = get_session_id()
    chat_history = get_chat_history(session_id)

    handle_assistant_response(user_query)
    messages = chat_history.messages

    assert len(messages) == 2
    assert messages[0].content == user_query
    assert messages[0].type == "user"
    assert messages[1].content == "Mocked response"
    assert messages[1].type == "ai"
