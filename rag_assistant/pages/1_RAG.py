from json import JSONDecodeError

import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import ToolException, Tool
# from openai import OpenAI
import os

from llama_index.llms.mistralai import MistralAI

from utils.config_loader import load_config

from utils.utilsrag import create_sentence_window_agent, create_automerging_agent, \
    create_subquery_agent, create_direct_query_agent

from utils.utilsllm import load_embeddings, load_model

from dotenv import load_dotenv, find_dotenv

# LLAMAINDEX
import chromadb


from langchain.agents import initialize_agent, create_react_agent, AgentExecutor
from langchain.agents.agent_types import AgentType

# LLAMA INDEX SUITE
from llama_index.core import VectorStoreIndex, StorageContext, SimpleDirectoryReader
from llama_index.core import Settings

from llama_index.llms.openai import OpenAI

from llama_index.vector_stores.chroma import ChromaVectorStore

config = load_config()

app_name = config['DEFAULT']['APP_NAME']
LLM_MODEL = config['LLM']['LLM_MODEL']


__template__ = """Answer the following questions as best you can. You have access to the following tools:

            {tools}

            Use the following format:

            Question: the input question you must answer
            Thought: you should always think about what to do
            Action: the action to take, should be one of [{tool_names}]
            Action Input: the input to the action
            Observation: the result of the action
            ... (this Thought/Action/Action Input/Observation can repeat N times)
            Thought: I now know the final answer
            Final Answer: the final answer to the original input question

            Begin!

            Question: {input}
            Thought:{agent_scratchpad}"""

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

topics = ["Cloud", "Security", "GenAI", "Application", "Architecture", "AWS", "Other"]

def load_sidebar():
    with st.sidebar:
        st.header("Parameters")
        st.sidebar.checkbox("Azure", LLM_MODEL == "azure", disabled=True)



@st.cache_resource(ttl="1h")
def configure_agent(model_name, advanced_rag = None):

    load_dotenv(find_dotenv())
    # Set OpenAI API key from Streamlit secrets
    openai_api_key = os.getenv('OPENAI_API_KEY')

    ## LLAMA
    loader = SimpleDirectoryReader(input_dir=f"data/sources/pdf/", recursive=True, required_exts=[".pdf"])
    all_docs = loader.load_data()

    # nltk.download('averaged_perceptron_tagger')
    agent_li = None
    llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
    # llm = MistralAI() # not working yet as utilrags use OpenAIAgent.from_tools method

    if advanced_rag == "sentence_window":

        name = "sentence_window_query_engine"
        description = f"useful for when you want to answer queries that require knowledge on {topics}"
        agent_li = create_sentence_window_agent(llm=llm, documents=all_docs, name=name, description=description)

    elif advanced_rag == "automerging":

        name = "automerging_query_engine"
        description = f"useful for when you want to answer queries that require knowledge on {topics}"
        agent_li = create_automerging_agent(llm=llm, documents=all_docs, name=name, description=description)

    elif advanced_rag == "subquery":

        name = "sub_question_query_engine"
        description = f"useful for when you want to answer queries that require knowledge on {topics}"
        agent_li = create_subquery_agent(llm=llm, topics=topics, documents=all_docs, name=name, description=description)

    elif advanced_rag == "direct_query":

        name = "direct_query_engine"
        description = f"useful for when you want to answer queries that require knowledge on {topics}"
        agent_li = create_direct_query_agent(llm=llm, documents=all_docs, name=name, description=description)

    def _handle_error(error: ToolException) -> str:
        if error == JSONDecodeError:
            return "Reformat in JSON and try again"
        elif error.args[0].startswith("Too many arguments to single-input tool"):
            return "Format in a SINGLE STRING. DO NOT USE MULTI-ARGUMENTS INPUT."
        return (
                "The following errors occurred during tool execution:"
                + error.args[0]
                + "Please try another tool.")

    lc_tools = [
        Tool(
            name=f"LlamaIndex RAG",
            func=agent_li.chat,
            description=f"""Useful when you need to answer questions. "
                        "DO NOT USE MULTI-ARGUMENTS INPUT.""",
            handle_tool_error=_handle_error,
        ),
    ]

    ## END LLAMA
    llm_agent = load_model(model_name)

    agent = create_react_agent(
        llm=llm_agent,
        tools=lc_tools,
        prompt=PromptTemplate.from_template(__template__)
    )

    agent_executor = AgentExecutor(agent=agent, tools=lc_tools)
    return agent_executor


def main():

    st.title("📄Chat with Doc 🤗")

    load_sidebar()

    # for openai only
    model_name = st.sidebar.radio("Model", ["gpt-3.5-turbo", "gpt-4"],
                                  captions=["GPT 3.5 Turbo", "GPT 4"],
                                  index=0, disabled=LLM_MODEL == "azure")

    template = st.sidebar.text_area("Prompt", __template2__)

    st.sidebar.subheader("RAG params")
    advanced_rag = st.sidebar.radio("Advanced RAG (Llamaindex)", ["direct_query", "subquery", "automerging",
                                                                  "sentence_window"])

    chain_type = st.sidebar.radio("Chain type",
                                  ["stuff", "map_reduce", "refine", "map_rerank"])

    st.sidebar.subheader("Search params")
    k = st.sidebar.slider('Number of relevant chunks', 1, 10, 4, 1)

    search_type = st.sidebar.radio("Search Type", ["similarity", "mmr",
                                                   "similarity_score_threshold"])

    st.sidebar.subheader("Chain params")
    verbose = st.sidebar.checkbox("Verbose")

    # llm = load_model(model_name)

    agent = configure_agent(model_name, advanced_rag)

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
            # store: VectorStore = get_store(embeddings)


            #output = agent.chat(prompt).response


            response = agent.invoke({"input": prompt})
            #output = agent.run(prompt)
            # output = invoke(prompt, template, llm, chain_type, store, search_type, k, verbose)

            st.write(response['output'])
            st.session_state.messages.append({"role": "assistant", "content": response['output']})


if __name__ == "__main__":
    main()
