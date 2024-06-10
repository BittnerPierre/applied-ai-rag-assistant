import streamlit as st
import sys
import logging
import os
from utils.config_loader import load_config


logger = logging.getLogger()

# Check if the directory exists, if not create it
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Create a file handler for the logger
file_handler = logging.FileHandler(os.path.join(log_dir, 'applied_ai_rag_assistant.log'))
file_handler.setLevel(logging.WARN)

# Create a logging format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add the file handler to the logger
logger.addHandler(file_handler)

config = load_config()

app_name = config['DEFAULT']['APP_NAME']

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)


def main():
    st.title("Welcome Page")

    st.write(f"""# Welcome to {app_name} ! 👋""")

    st.markdown(
        """
        LLM+RAG Assistant for Application Architecture.
    """
    )
    st.write(sys.version)


if __name__ == "__main__":
    main()
