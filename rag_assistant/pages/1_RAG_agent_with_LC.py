from json import JSONDecodeError

import streamlit as st
from langchain.agents import AgentExecutor, create_react_agent, create_structured_chat_agent
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import Tool, ToolException

from utils.config_loader import load_config

from utils.utilsrag_lc import agent_lc_factory

from utils.utilsllm import load_model, load_embeddings

from dotenv import load_dotenv, find_dotenv

from langchain_core.tracers.context import tracing_v2_enabled

# EXTERNALISATION OF PROMPTS TO HAVE THEIR OWN VERSIONING
from shared.rag_prompts import __structured_chat_agent__

load_dotenv(find_dotenv())

config = load_config()

app_name = config['DEFAULT']['APP_NAME']
LLM_MODEL = config['LLM']['LLM_MODEL']

topics = ["Cloud", "Security", "GenAI", "Application", "Architecture", "AWS", "Other"]

model_to_index = {
    "OPENAI": 0,
    "MISTRAL": 1
}


human = '''{input}
    
    {agent_scratchpad}
    
    (reminder to respond in a JSON blob no matter what)'''



def load_sidebar():
    with st.sidebar:
        st.header("Parameters")
        st.sidebar.subheader("LangChain model provider")
        st.sidebar.checkbox("OpenAI", LLM_MODEL == "OPENAI", disabled=True)
        st.sidebar.checkbox("Mistral", LLM_MODEL == "MISTRAL", disabled=True)


@st.cache_resource(ttl="1h")
def load_doc() -> list[Document]:
    loader = PyPDFDirectoryLoader("data/sources/pdf/")
    all_docs = loader.load()
    return all_docs


@st.cache_resource(ttl="1h")
def configure_agent(model_name, chain_type=None, search_type="similarity", search_kwargs=None):
    all_docs = load_doc()

    embeddings_rag = load_embeddings(model_name)
    llm_rag = load_model(model_name, temperature=0.1)

    retrieval_qa_chain = agent_lc_factory(all_docs, chain_type, embeddings_rag, llm_rag, search_kwargs,
                                          search_type, collection_name="RAG_LC_Agent")

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
    llm_agent = load_model()
    # create_react_agent
    agent = create_structured_chat_agent(
        llm=llm_agent,
        tools=lc_tools,
        prompt=ChatPromptTemplate.from_messages(
                [
                    ("system", __structured_chat_agent__),
                    MessagesPlaceholder("chat_history", optional=True),
                    ("human", human),
                ]
            )
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
    ## END LANGCHAIN
    return agent_executor


def main():

    st.title("Question Answering Assistant (RAG)")

    load_sidebar()

    model_index = model_to_index[LLM_MODEL]
    agent_model = st.sidebar.radio("RAG Agent LLM Provider", ["OPENAI", "MISTRAL"], index=model_index)

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
        model_name = model_name_mistral
    elif agent_model == "OPENAI":
        model_name = model_name_gpt

    ## OLD STUFF WITH LANGCHAIN, COMMENTED TO FOCUS ON LLAMAINDEX AGENT
    chain_type = st.sidebar.radio("Chain type (LangChain)",
                                  ["stuff", "map_reduce", "refine", "map_rerank"])

    st.sidebar.subheader("Search params (LangChain)")
    k = st.sidebar.slider('Number of relevant chunks', 2, 10, 4, 1)

    search_type = st.sidebar.radio("Search Type", ["similarity", "mmr",
                                                    "similarity_score_threshold"])

    st.header("RAG agent with LangChain")
    agent = configure_agent(model_name, chain_type, search_type, {"k":k})

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

            ## OLD WAY without agent
            # llm = load_model()
            # embeddings = load_embeddings()
            # output = invoke(prompt, template, llm, chain_type, store, search_type, k, verbose)
            # st.write(output)
            # st.session_state.messages.append({"role": "assistant", "content": output})

            with tracing_v2_enabled(project_name="Applied AI RAG Assistant", tags=["LangChain", "Agent"]):
                chat_history = st.session_state.messages
                response = agent.invoke(
                    {
                        "input": prompt,
                        "chat_history": chat_history
                    }
                )
                answer = f"🦜: {response['output']}"
                st.write(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    load_doc()
    main()
