import streamlit as st
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_core.vectorstores import VectorStore

from utils.config_loader import load_config
from utils.utilsdoc import get_store, load_store

from utils.utilsrag import invoke

from utils.utilsllm import load_model, load_embeddings

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

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
        st.sidebar.subheader("LangChain model provider")
        st.sidebar.checkbox("OpenAI", LLM_MODEL == "OPENAI", disabled=True)
        st.sidebar.checkbox("Mistral", LLM_MODEL == "MISTRAL", disabled=True)


# @st.cache_resource(ttl="1h")
# def configure_agent(model_name, advanced_rag = None):
#
#     ## START LLAMAINDEX
#     loader = DirectoryReader("data/sources/pdf/", globs="**/*.pdf")
#     all_docs = loader.load_data()
#
#     def _handle_error(error: ToolException) -> str:
#         if error == JSONDecodeError:
#             return "Reformat in JSON and try again"
#         elif error.args[0].startswith("Too many arguments to single-input tool"):
#             return "Format in a SINGLE STRING. DO NOT USE MULTI-ARGUMENTS INPUT."
#         return (
#                 "The following errors occurred during tool execution:"
#                 + error.args[0]
#                 + "Please try another tool.")
#
#     lc_tools = [
#         Tool(
#             name=f"LlamaIndex RAG Agent",
#             func=agent_li.chat,
#             description=f"""Useful when you need to answer questions. "
#                         "DO NOT USE MULTI-ARGUMENTS INPUT.""",
#             handle_tool_error=_handle_error,
#         ),
#     ]
#     ## END LLAMAINDEX
#
#     ## START LANGCHAIN
#     # MODEL FOR LANGCHAIN IS DEFINE GLOBALLY IN CONF/CONFIG.INI
#     llm_agent = load_model()
#
#     agent = create_react_agent(
#         llm=llm_agent,
#         tools=lc_tools,
#         prompt=PromptTemplate.from_template(__template__)
#     )
#
#     agent_executor = AgentExecutor(agent=agent, tools=lc_tools, handle_parsing_errors=True)
#     ## END LANGCHAIN
#     return agent_executor


@st.cache_resource(ttl="1h")
def load_doc():
    loader = PyPDFDirectoryLoader("data/sources/pdf/")
    all_docs = loader.load()
    load_store(all_docs)
    return all_docs


def main():

    st.title("📄Chat with Doc 🤗")

    load_sidebar()

    agent_model = st.sidebar.radio("RAG Agent Provider", ["OPENAI", "MISTRAL"], index=1)

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

    # llm = load_model(model_name)
    embeddings = load_embeddings()
    llm = load_model()

    template = st.sidebar.text_area("Prompt", __template2__)

    ## OLD STUFF WITH LANGCHAIN, COMMENTED TO FOCUS ON LLAMAINDEX AGENT
    chain_type = st.sidebar.radio("Chain type (LangChain)",
                                  ["stuff", "map_reduce", "refine", "map_rerank"])

    st.sidebar.subheader("Search params (LangChain)")
    k = st.sidebar.slider('Number of relevant chunks', 1, 10, 4, 1)

    search_type = st.sidebar.radio("Search Type", ["similarity", "mmr",
                                                   "similarity_score_threshold"])

    # agent = configure_agent(model_name, advanced_rag)

    verbose = st.sidebar.checkbox("Verbose")

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

            # response = agent.invoke({"input": prompt})

            st.write(output)
            st.session_state.messages.append({"role": "assistant", "content": output})


if __name__ == "__main__":
    load_doc()
    main()
