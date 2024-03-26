import streamlit as st
from langchain.schema.vectorstore import VectorStore
from openai import OpenAI
import os
from utils.config_loader import load_config

from utils.utilsrag import invoke
from utils.utilsdoc import load_doc, load_store, get_store
from utils.utilsllm import load_embeddings, load_model

from dotenv import load_dotenv, find_dotenv

config = load_config()

app_name = config['DEFAULT']['APP_NAME']
LLM_MODEL = config['LLM']['LLM_MODEL']


__template__ = """Use the following pieces of context to answer the question at the end. 
   If you don't know the answer, just say that you don't know, don't try to make up an answer. 
   Use three sentences maximum and keep the answer as concise as possible. 
   Always answer in French. 
   {context}
   Question: {question}
   Helpful Answer:"""

__template2__ = """You are an assistant designed to guide users through a structured risk assessment questionnaire for cloud deployment. 
    The questionnaire is designed to cover various pillars essential for cloud architecture,
     including security, compliance, availability, access methods, data storage, processing, performance efficiency,
      cost optimization, and operational excellence.

    For each question, you are to follow the "Chain of Thought" process. This means that for each user's response, you will:
    
    - Acknowledge the response,
    - Reflect on the implications of the choice,
    - Identify any risks associated with the selected option,
    - Suggest best practices and architecture patterns that align with the user’s selection,
    - Guide them to the next relevant question based on their previous answers.
    
    Your objective is to ensure that by the end of the questionnaire, the user has a clear understanding of the appropriate architecture and services needed for a secure, efficient, and compliant cloud deployment. Remember to provide answers in a simple, interactive, and concise manner.
    
    Process:
    
    1. Begin by introducing the purpose of the assessment and ask the first question regarding data security and compliance.
    2. Based on the response, discuss the chosen level of data security, note any specific risks or requirements, and recommend corresponding cloud services or architectural patterns.
    3. Proceed to the next question on application availability. Once the user responds, reflect on the suitability of their choice for their application's criticality and suggest availability configurations.
    4. For questions on access methods and data storage, provide insights on securing application access points or optimizing data storage solutions.
    5. When discussing performance efficiency, highlight the trade-offs between performance and cost, and advise on scaling strategies.
    6. In the cost optimization section, engage in a brief discussion on budgetary constraints and recommend cost-effective cloud resource management.
    7. Conclude with operational excellence, focusing on automation and monitoring, and propose solutions for continuous integration and deployment.
    8. After the final question, summarize the user's choices and their implications for cloud architecture.
    9. Offer a brief closing statement that reassures the user of the assistance provided and the readiness of their cloud deployment strategy.
    
    Keep the interactions focused on architectural decisions without diverting to other unrelated topics. 
    You are not to perform tasks outside the scope of the questionnaire, 
    such as executing code or accessing external databases. 
    Your guidance should be solely based on the information provided by the user in the context of the questionnaire.
    Always answer in French. 
    {context}
    Question: {question}
    Helpful Answer:"""

def load_sidebar():
    with st.sidebar:
        st.header("Parameters")
        st.sidebar.checkbox("Azure", LLM_MODEL == "azure", disabled=True)



def main():

    st.title("📄Chat with Doc 🤗")

    load_sidebar()

    # for openai only
    model_name = st.sidebar.radio("Model", ["gpt-3.5-turbo", "gpt-4"],
                                  captions=["GPT 3.5 Turbo", "GPT 4"],
                                  index=1, disabled=LLM_MODEL == "azure")

    template = st.sidebar.text_area("Prompt", __template2__)

    st.sidebar.subheader("RAG params")
    chain_type = st.sidebar.radio("Chain type",
                                  ["stuff", "map_reduce", "refine", "map_rerank"])

    st.sidebar.subheader("Search params")
    k = st.sidebar.slider('Number of relevant chunks', 1, 10, 4, 1)

    search_type = st.sidebar.radio("Search Type", ["similarity", "mmr",
                                                   "similarity_score_threshold"])

    st.sidebar.subheader("Chain params")
    verbose = st.sidebar.checkbox("Verbose")

    # llm = load_model(model_name)
    embeddings = load_embeddings()

    load_dotenv(find_dotenv())
    # Set OpenAI API key from Streamlit secrets
    openai_api_key = os.getenv('OPENAI_API_KEY')

    client = OpenAI(api_key=openai_api_key)

    # Set a default model
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = model_name

    llm = load_model(model_name)

    st.header("Question Answering Assistant")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("What do you want to know?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            store: VectorStore = get_store(embeddings)

            output = invoke(prompt, template, llm, chain_type, store, search_type, k, verbose)

            st.write(output)
            st.session_state.messages.append({"role": "assistant", "content": output})


if __name__ == "__main__":
    main()
