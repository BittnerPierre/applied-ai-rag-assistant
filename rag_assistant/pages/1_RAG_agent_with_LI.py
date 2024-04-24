from json import JSONDecodeError

import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import ToolException, Tool

from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

from llama_index.embeddings.mistralai import MistralAIEmbedding
from llama_index.llms.mistralai import MistralAI

from utils.config_loader import load_config

from utils.utils_rag_li import agent_li_factory

from utils.utilsllm import load_model

from dotenv import load_dotenv, find_dotenv

from langchain.agents import create_react_agent, AgentExecutor

from llama_index.core import SimpleDirectoryReader, Settings

from langchain_core.tracers.context import tracing_v2_enabled


# EXTERNALISATION OF PROMPTS TO HAVE THEIR OWN VERSIONING
from shared.rag_prompts import __template__

load_dotenv(find_dotenv())

config = load_config()

app_name = config['DEFAULT']['APP_NAME']
LLM_MODEL = config['LLM']['LLM_MODEL']

topics = ["Cloud", "Security", "GenAI", "Application", "Architecture", "AWS", "Other"]


def load_sidebar():
    with st.sidebar:
        st.header("Parameters")
        st.sidebar.subheader("LangChain model provider")
        st.sidebar.checkbox("OpenAI", LLM_MODEL == "OPENAI", disabled=True)
        st.sidebar.checkbox("Mistral", LLM_MODEL == "MISTRAL", disabled=True)


@st.cache_resource(ttl="1h")
def load_doc():
    loader = SimpleDirectoryReader(input_dir=f"data/sources/pdf/", recursive=True, required_exts=[".pdf"])
    all_docs = loader.load_data()
    return all_docs


@st.cache_resource(ttl="1h")
def configure_agent(model_name, advanced_rag = None):
    all_docs = load_doc()

    ## START LLAMAINDEX
    embeddings_rag = None
    llm_rag = None
    if model_name.startswith("gpt"):
        llm_rag = OpenAI(model=model_name, temperature=0.1)
        embeddings_rag = OpenAIEmbedding()
    if model_name.startswith("mistral"):
        llm_rag = MistralAI(model=model_name, temperature=0.1)
        embeddings_rag = MistralAIEmbedding()

    #
    # Settings seems to be the preferred version to setup llm and embeddings with latest LI API
    Settings.llm = llm_rag
    Settings.embed_model = embeddings_rag

    agent_li = agent_li_factory(advanced_rag=advanced_rag, llm=llm_rag, documents=all_docs, topics=topics)

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
            name=f"Knowledge Agent (LI)",
            func=agent_li.chat,
            description=f"""Useful when you need to answer questions on {topics}. "
                        "DO NOT USE MULTI-ARGUMENTS INPUT.""",
            handle_tool_error=_handle_error,
        ),
    ]
    #
    # END LLAMAINDEX

    #
    # START LANGCHAIN
    # MODEL FOR LANGCHAIN IS DEFINE GLOBALLY IN CONF/CONFIG.INI
    llm_agent = load_model()

    agent = create_react_agent(
        llm=llm_agent,
        tools=lc_tools,
        prompt=PromptTemplate.from_template(__template__)
    )

    #
    # TODO
    # sometimes received "Parsing LLM output produced both a final answer and a parse-able action" with mistral
    # add a handle_parsing_errors, reduce the case but still appears time to time.
    agent_executor = AgentExecutor(
        agent=agent,
        tools=lc_tools,
        handle_parsing_errors="Check your output and make sure it conforms!"
                              " Do not output an action and a final answer at the same time.")
    # END LANGCHAIN
    #
    return agent_executor


def main():

    st.title("Question Answering Assistant (RAG)")

    load_sidebar()

    agent_model = st.sidebar.radio("RAG Agent LLM Provider", ["OPENAI", "MISTRAL"], index=1)

    st.sidebar.subheader("RAG Agent Model")
    # for openai only
    model_name_gpt = st.sidebar.radio("OpenAI Model", ["gpt-3.5-turbo", "gpt-4-turbo"],
                                      captions=["GPT 3.5 Turbo", "GPT 4 Turbo"],
                                      index=0, disabled=agent_model != "OPENAI")

    model_name_mistral = st.sidebar.radio("Mistral Model", ["mistral-small-latest", "mistral-medium-latest", "mistral-large-latest"],
                                          captions=["Mistral 7b", "Mixtral", "Mistral Large"],
                                          index=2, disabled=agent_model != "MISTRAL")

    model_name = None
    if agent_model == "MISTRAL":
        model_name = model_name_gpt
    elif agent_model == "OPENAI":
        model_name = model_name_mistral

    st.sidebar.subheader("RAG Agent params")
    advanced_rag = st.sidebar.radio("Advanced RAG (Llamaindex)", ["direct_query", "subquery", "automerging",
                                                                  "sentence_window"])

    st.header("RAG agent with LlamaIndex")
    agent = configure_agent(model_name, advanced_rag)

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
            with tracing_v2_enabled(project_name="Applied AI RAG Assistant", tags=["LlamaIndex", "Agent"]):
                response = agent.invoke({"input": prompt})

                answer = f"🦙: {response['output']}"
                st.write(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    load_doc()
    main()
