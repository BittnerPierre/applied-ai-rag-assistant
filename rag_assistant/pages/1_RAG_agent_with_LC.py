from json import JSONDecodeError
from typing import Union

import chromadb
import streamlit as st
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import Tool, ToolException
from streamlit.runtime.uploaded_file_manager import UploadedFile

from utils.config_loader import load_config
from utils.utilsdoc import load_doc

from utils.utilsrag_lc import agent_lc_factory

from utils.utilsllm import load_model, load_embeddings

from dotenv import load_dotenv, find_dotenv

from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import (
    StreamlitChatMessageHistory,
)

from langchain_core.tracers.context import tracing_v2_enabled

# EXTERNALISATION OF PROMPTS TO HAVE THEIR OWN VERSIONING
from shared.rag_prompts import __structured_chat_agent__, human

load_dotenv(find_dotenv())

config = load_config()

app_name = config['DEFAULT']['APP_NAME']
LLM_MODEL = config['MODEL_PROVIDER']['MODEL_PROVIDER']

topics = ["Cloud", "Security", "GenAI", "Application", "Architecture", "AWS", "Other"]

model_to_index = {
    "OPENAI": 0,
    "MISTRAL": 1,
    "BEDROCK": 2
}


def load_sidebar():
    with st.sidebar:
        st.header("Parameters")
        st.sidebar.subheader("LangChain model provider")
        st.sidebar.checkbox("OpenAI", LLM_MODEL == "OPENAI", disabled=True)
        st.sidebar.checkbox("Mistral", LLM_MODEL == "MISTRAL", disabled=True)
        st.sidebar.checkbox("Bedrock", LLM_MODEL == "BEDROCK", disabled=True)


def _load_doc(pdfs: Union[list[UploadedFile], None, UploadedFile]) -> list[Document]:
    # loader = PyPDFDirectoryLoader("data/sources/pdf/")
    # all_docs = loader.load()
    all_docs = load_doc(pdfs)
    return all_docs


def configure_agent(all_docs: list[Document], model_name, chain_type, search_type="similarity", search_kwargs=None):

    embeddings_rag = load_embeddings(model_name)
    llm_rag = load_model(model_name, temperature=0.1)

    chroma_client = chromadb.EphemeralClient()

    vectorstore = Chroma.from_documents(
        documents=all_docs,
        embedding=embeddings_rag,
        client=chroma_client,
        collection_name="RAG_LC_Agent"
    )

    # vectorstore = get_store() # embeddings_rag, collection_name="RAG_LC_Agent")
    retrieval_qa_chain = agent_lc_factory(chain_type, llm_rag, search_kwargs,
                                          search_type, vectorstore)

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
            name=f"Knowledge Agent (LC)",
            func=retrieval_qa_chain,
            description=f"""Useful when you need to answer questions on {topics}. "
                        "DO NOT USE MULTI-ARGUMENTS INPUT.""",
            handle_tool_error=_handle_error,
        ),
    ]
    ## START LANGCHAIN
    # MODEL FOR LANGCHAIN IS DEFINE GLOBALLY IN CONF/CONFIG.INI
    # defaulting to "gpt-4-turbo" because it is the only one resilient
    llm_agent = load_model("gpt-4-turbo")

    prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", __structured_chat_agent__),
                    MessagesPlaceholder("rag_chat_history", optional=True),
                    ("human", human),
                ]
            )

    # create_react_agent
    agent = create_structured_chat_agent(
        llm=llm_agent,
        tools=lc_tools,
        prompt=prompt
    )

    #
    # TODO
    # sometimes received "Parsing LLM output produced both a final answer and a parse-able action" with mistral
    # add a handle_parsing_errors, reduce the case but still appears time to time.
    agent_executor = AgentExecutor(
        agent=agent,
        tools=lc_tools,
        handle_parsing_errors="Check your output and make sure it conforms to required format!"
                              "Format is Action:```$JSON_BLOB``` then Observation"
                              " Do not output an action and a final answer at the same time.")
    ## END LANGCHAIN
    return agent_executor


def main():

    st.title("Question Answering Assistant (RAG)")

    load_sidebar()

    model_index = model_to_index[LLM_MODEL]
    agent_model = st.sidebar.radio("RAG Agent LLM Provider", ["OPENAI", "MISTRAL", "BEDROCK"], index=model_index)

    st.sidebar.subheader("RAG Agent Model")
    model_name_gpt = st.sidebar.radio("OpenAI Model", ["gpt-3.5-turbo", "gpt-4-turbo"],
                                      captions=["GPT 3.5 Turbo", "GPT 4 Turbo"],
                                      index=1, disabled=agent_model != "OPENAI")

    model_name_mistral = st.sidebar.radio("Mistral Model", ["mistral-small-latest", "mistral-medium-latest", "mistral-large-latest"],
                                          captions=["Mistral 7b", "Mixtral", "Mistral Large"],
                                          index=2, disabled=agent_model != "MISTRAL")

    model_name_bedrock = st.sidebar.radio("Bedrock Model", ["anthropic.claude-v2:1", "anthropic.claude-v2"],
                                            captions=["Claude v2.1", "Claude v2"],
                                            index=0, disabled=agent_model != "BEDROCK")

    model_name = None
    if agent_model == "MISTRAL":
        model_name = model_name_mistral
    elif agent_model == "OPENAI":
        model_name = model_name_gpt
    elif agent_model == "BEDROCK":
        model_name = model_name_bedrock

    chain_type = st.sidebar.radio("Chain type (LangChain)",
                                  ["stuff", "map_reduce", "refine", "map_rerank"])

    st.sidebar.subheader("Search params (LangChain)")
    k = st.sidebar.slider('Number of relevant chunks', 2, 10, 4, 1)

    search_type = st.sidebar.radio("Search Type", ["similarity", "mmr",
                                                    "similarity_score_threshold"])

    pdfs = st.file_uploader("Document(s) à transmettre", type=['pdf', 'txt', 'md'], accept_multiple_files=True)

    disabled = True

    docs = []
    # if st.button("Transmettre", disabled=disabled):
        # calling an internal function for adapting LC or LI Document
    docs = _load_doc(pdfs)

    if (docs is not None) and (len(docs)):
        disabled = False

    if not disabled:

        history = StreamlitChatMessageHistory(key="rag_chat_history")
        if len(history.messages) == 0:
            history.add_ai_message("What do you want to know?")

        view_messages = st.expander("View the message contents in session state")

        st.header("RAG agent with LangChain")
        agent = configure_agent(docs, model_name, chain_type, search_type, {"k": k})

        chain_with_history = RunnableWithMessageHistory(
            agent,
            lambda session_id: history,
            input_messages_key="input",
            history_messages_key="rag_chat_history",
        )

        # Display chat messages from history on app rerun
        for message in history.messages:
            with st.chat_message(message.type):
                st.markdown(message.content)

        # Accept user input
        if prompt := st.chat_input():
            # Add user message to chat history
            # Note: new messages are saved to history automatically by Langchain during run
            # st.session_state.messages.append({"role": "user", "content": prompt})
            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(prompt)

            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                # Display assistant response in chat message container
                with tracing_v2_enabled(project_name="Applied AI RAG Assistant",
                                        tags=["LangChain", "Agent"]):
                    config = {"configurable": {"session_id": "any"}}
                    response = chain_with_history.invoke(
                        input={
                            "input": prompt
                        },
                        config=config
                    )
                    answer = f"🦜: {response['output']}"
                    st.write(answer)

        # Draw the messages at the end, so newly generated ones show up immediately
        with view_messages:
            """
            Message History initialized with:
            ```python
            msgs = StreamlitChatMessageHistory(key="rag_chat_history")
            ```
    
            Contents of `st.session_state.rag_chat_history`:
            """
            view_messages.json(st.session_state.rag_chat_history)


if __name__ == "__main__":
    main()
