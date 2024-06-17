import streamlit as st
import sys
from utils.config_loader import load_config
from utils.utilsllm import get_model_provider, get_model_name, get_embeddings_model_name

config = load_config()

app_name = config['DEFAULT']['APP_NAME']
vectordb = config['VECTORDB']['vectordb']

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

model_provider = get_model_provider()
model_name = get_model_name(provider=model_provider)
embeddings_model = get_embeddings_model_name(provider=model_provider)

def main():
    st.title(f"""Welcome to {app_name} ! 👋""")

    st.markdown(
        f"""
        {app_name} on '**{model_provider}**' with '**{model_name}**' LLM.
        
        Knowledge base on '**{vectordb}**' with embedding model : '**{embeddings_model}**'.
    """
    )
    st.write(sys.version)


if __name__ == "__main__":
    main()
