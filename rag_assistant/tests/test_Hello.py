from unittest.mock import patch
from streamlit.testing.v1 import AppTest


def test_welcome_page():
    """Test the welcome page of the app."""

    # Create an instance of AppTest
    at = AppTest.from_file("../Hello.py")
    at.run()
    assert not at.exception
    assert at.title[0].value == "Welcome Page"
    assert at.markdown[0].value == "# Welcome to My LLM+RAG Assistant ! 👋"


@patch('utils.config_loader.load_config')
def test_config_welcome_page(mock_load_config):
    """Test the welcome page of the app with a dynamic app name."""

    #Set up the test configuration
    mock_load_config.return_value = {
        'DEFAULT': {
            'APP_NAME': 'My LLM+RAG Assistant'
        }
    }

    # Create an instance of AppTest
    at = AppTest.from_file("../Hello.py")
    at.run()
    assert not at.exception
    assert at.title[0].value == "Welcome Page"
    assert at.markdown[0].value == "# Welcome to My LLM+RAG Assistant ! 👋"