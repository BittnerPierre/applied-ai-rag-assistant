import os
import threading
import streamlit as st
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories.streamlit import StreamlitChatMessageHistory
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from dotenv import load_dotenv, find_dotenv
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, \
    HumanMessagePromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.tracers.context import tracing_v2_enabled
import uuid
from utils.utilsdoc import get_store
from utils.config_loader import load_config
from streamlit_feedback import streamlit_feedback
import logging

from streamlit.runtime.scriptrunner import add_script_run_ctx
from streamlit.runtime.scriptrunner.script_run_context import get_script_run_ctx

from utils.utilsllm import load_model

load_dotenv(find_dotenv())
openai_api_key = os.getenv('OPENAI_API_KEY')

# set logging
logger = logging.getLogger('AI_assistant_feedback')
logger.setLevel(logging.INFO)

# Check if the directory exists, if not create it
log_dir = "./logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Create a file handler for the logger
handler = logging.FileHandler(os.path.join(log_dir, 'feedback.log'))
handler.setLevel(logging.INFO)

# Create a logging format
formatter = logging.Formatter('%(asctime)s -  %(message)s')
handler.setFormatter(formatter) # Add the formatter to the handler  

# Add the handler to the logger
logger.addHandler(handler)


config = load_config()
collection_name = config['VECTORDB']['collection_name']

def get_session_id():
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    return st.session_state.session_id

def get_chat_history(session_id):
    if "chat_histories" not in st.session_state:
        st.session_state.chat_histories = {}
    if session_id not in st.session_state.chat_histories:
        st.session_state.chat_histories[session_id] = StreamlitChatMessageHistory(key=f"chat_history_{session_id}")
    return st.session_state.chat_histories[session_id]

__template2__ = """You are an assistant designed to guide software application architect and tech lead to go through a risk assessment questionnaire for application cloud deployment. 
    The questionnaire is designed to cover various pillars essential for cloud architecture,
     including security, compliance, availability, access methods, data storage, processing, performance efficiency,
      cost optimization, and operational excellence.
      
    You will assist user to answer to the questionnaire solely based on the information that will be provided to you.

    For each question, you are to follow the "Chain of Thought" process. This means that for each user's response, you will:

    - Acknowledge the response,
    - Reflect on the implications of the choice,
    - Identify any risks associated with the selected option,
    - Suggest best practices and architecture patterns that align with the user's selection,
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
    Be concise in your answer with a professional tone. 
    You are not to perform tasks outside the scope of the questionnaire, 
    such as executing code or accessing external databases. 
    Your guidance should be solely based on the information provided by the user in the context of the questionnaire.
    
    To start the conversation, introduce yourself and give 3 domains in which you can assist user."""


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container: st.delta_generator.DeltaGenerator, ctx, initial_text: str = ""):
        self.container = container
        self.text = initial_text
        self.run_id_ignore_token = None
        self.ctx = ctx

    def on_llm_start(self, serialized: dict, prompts: list, **kwargs):
        # Workaround to prevent showing the rephrased question as output
        # THIS TRICKS DOES NOT WORK IF THERE IS A SYSTEM PROMPT
        # SO QUESTION IS SHOWED WHEN STREAMING AND THEN CLEARED with final result
        if prompts[0].startswith("Human"):
            self.run_id_ignore_token = kwargs.get("run_id")
        # adding current thread to streamlit context to be able to display streaming
        add_script_run_ctx(threading.current_thread(), self.ctx)

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if self.run_id_ignore_token == kwargs.get("run_id", False):
            return
        self.text += token
        self.container.markdown(self.text)


# Define the callback handler for printing retrieval information
class PrintRetrievalHandler(BaseCallbackHandler):
    def __init__(self, container, ctx):
        self.status = container.status("**Context Retrieval**")
        self.ctx = ctx

    def on_retriever_start(self, serialized: dict, query: str, **kwargs):
        # adding current thread to streamlit context to be able to display streaming
        add_script_run_ctx(threading.current_thread(), self.ctx)
        self.status.write(f"**Question:** {query}")
        self.status.update(label=f"**Context Retrieval:** {query}")

    def on_retriever_end(self, documents, **kwargs):
        for idx, doc in enumerate(documents):
            source = doc.metadata["filename"]
            self.status.write(f"**Document {idx} from {source}**")
            self.status.markdown(doc.page_content)
        self.status.update(state="complete")


def configure_retriever():
    vectordb = get_store()

    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 5}) # , "fetch_k": 4

    return retriever


def _submit_feedback(user_response, emoji=None):
    if user_response['score'] == '👍':
        feedback_score = '+1'
    else:
        feedback_score = '-1'
    logger.info(f"Feedback_Score: {feedback_score}, Feedback_text: {user_response['text']}")
    return user_response


st.set_page_config(page_title="Chat with Documents", page_icon="🦜")


# Configure the retriever with PDF files
retriever = configure_retriever()

# Setup memory for contextual conversation
memory = ConversationBufferMemory(return_messages=True)

# Setup LLM and QA chain
llm = load_model(streaming=True)

### Contextualize question ###
contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is.
Maintain the same language as the user question.
"""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

### Answer question ###
qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.\
Maintain the same writing style as used in the context.\
Keep the same language as the follow up question.

{context}"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_chat_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)


llm_stream = load_model(streaming=True)


general_system_template = r""" 
Given a specific context, please give a short answer to the question,
 covering the required advices in general
 
Context:
----
{context}
----
Please respond while maintaining the same writing style as used in this excerpt.
Maintain the same language as the follow up input message.
"""

# Was in the previous prompt
#  and then provide the names all of relevant (even if it relates a bit) products.

general_user_template = "Question:```{question}```"
messages = [
            SystemMessagePromptTemplate.from_template(general_system_template),
            HumanMessagePromptTemplate.from_template(general_user_template)
]
qa_prompt = ChatPromptTemplate.from_messages( messages )

qa_chain = ConversationalRetrievalChain.from_llm(
    llm_stream, retriever=retriever, memory=memory, verbose=True,
    combine_docs_chain_kwargs={'prompt': qa_prompt}
)

suggested_questions = [
    "Comment sécuriser les données sensibles ?",
    "Quelles stratégies pour une haute disponibilité ?",
    "Quels sont les mécanismes d'authentification API ?",
    "Comment assurez l'efficacité des performances ?",
]



def handle_assistant_response(user_query):
    session_id = get_session_id()
    msgs = get_chat_history(session_id)

    st.chat_message("user").write(user_query)
    with ((st.chat_message("assistant"))):
        # Retrieving the streamlit context to bind it to call back
        # in order to write in another threadcontext
        ctx = get_script_run_ctx()
        retrieval_handler = PrintRetrievalHandler(st.container(), ctx)
        # RETRIEVE THE CONTAINER TO CLEAR IT LATER to not show question twice
        e = st.empty()
        stream_handler = StreamHandler(e, ctx)
        ai_response = ""
        with tracing_v2_enabled(project_name="Chat with Docs",
                                tags=["LangChain", "Chain", "Chat History"]):
            ai_response = conversational_rag_chain.invoke(
                    input={"input": user_query},
                    config={
                        "configurable": {"session_id": session_id},
                        "callbacks": [
                            retrieval_handler,
                            stream_handler
                        ]
                    },
                )["answer"]
            # emptying container to remove initial question that is render by llm
            e.empty()
            with e.container():
                st.markdown(ai_response)
        logger.info(f"User Query: {user_query}, AI Response: {ai_response}")

def suggestion_clicked(question):
    st.session_state.user_suggested_question = question


def main():
    st.title("Chat with Documents")

    st.sidebar.title("Chat Sessions")

    session_id = get_session_id()
    chat_sessions = list(st.session_state.get("chat_histories", {}).keys())

    if st.sidebar.button("New Chat Session"):
        session_id = str(uuid.uuid4())
        st.session_state.chat_histories[session_id] = StreamlitChatMessageHistory(key=f"chat_history_{session_id}")

    selected_session = st.sidebar.radio("Select Chat Session", [session_id] + chat_sessions)

    if selected_session != session_id:
        session_id = selected_session
        st.session_state.session_id = session_id

    msgs = get_chat_history(session_id)

    if len(msgs.messages) == 0 or st.sidebar.button("Clear message history"):
        msgs.clear()

    # Display suggested questions in a 2x2 table
    col1, col2 = st.columns(2)
    for i, question in enumerate(suggested_questions, start=1):
        col = col1 if i % 2 != 0 else col2
        col.button(question, on_click=suggestion_clicked, args=[question])

    # Chat interface
    avatars = {"human": "user", "ai": "assistant"}
    msgs = get_chat_history(session_id)
    for i, msg in enumerate(msgs.messages):
        st.chat_message(avatars[msg.type]).write(msg.content)
        if (msg.type == "ai") and (i > 0):
            streamlit_feedback(feedback_type = "thumbs",
                               optional_text_label="Cette réponse vous convient-elle?",
                               key=f"feedback_{i}",
                               on_submit=lambda x: _submit_feedback(x, emoji="👍"))

    # Handle suggested questions
    if "user_suggested_question" in st.session_state:
        user_query = st.session_state.user_suggested_question
        st.session_state.pop("user_suggested_question")  # Clear the session state
        handle_assistant_response(user_query)

    # Handle user queries
    if user_query := st.chat_input(placeholder="Ask me anything!"):
        handle_assistant_response(user_query)


if __name__ == "__main__":
    main()




# def handle_assistant_response(user_query):
#     st.chat_message("user").write(user_query)
#     with ((st.chat_message("assistant"))):
#         # Retrieving the streamlit context to bind it to call back
#         # in order to write in another threadcontext
#         ctx = get_script_run_ctx()
#         retrieval_handler = PrintRetrievalHandler(st.container(), ctx)
#         # RETRIEVE THE CONTAINER TO CLEAR IT LATER to not show question twice
#         e = st.empty()
#         stream_handler = StreamHandler(e, ctx)
#         ai_response = ""
#         with tracing_v2_enabled(project_name="Chat with Docs",
#                                 tags=["LangChain", "Chain", "Chat History"]):
#             # Use the current session ID when adding messages to the chat history
#             ai_response = conversational_rag_chain.invoke(
#                     input={"input": user_query},
#                     config={
#                         "configurable": {"session_id": session_id},
#                         "callbacks": [
#                             retrieval_handler,
#                             stream_handler
#                         ]
#                     },
#                 )["answer"]
#             # emptying container to remove initial question that is render by llm
#             e.empty()
#             with e.container():
#                 st.markdown(ai_response)
#         logger.info(f"User Query: {user_query}, AI Response: {ai_response}")