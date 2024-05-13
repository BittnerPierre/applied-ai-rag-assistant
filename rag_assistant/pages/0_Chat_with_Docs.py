import os

import streamlit as st
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories.streamlit import StreamlitChatMessageHistory
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from dotenv import load_dotenv, find_dotenv
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, \
    HumanMessagePromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.tracers.context import tracing_v2_enabled

from utils.utilsdoc import get_store
from utils.config_loader import load_config
from streamlit_feedback import streamlit_feedback
import logging

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

sessionid = "abc123"

__template2__ = """You are an assistant designed to guide software application architect and tech lead to go through a risk assessment questionnaire for application cloud deployment. 
    The questionnaire is designed to cover various pillars essential for cloud architecture,
     including security, compliance, availability, access methods, data storage, processing, performance efficiency,
      cost optimization, and operational excellence.
      
    You will assist user to answer to the questionnaire solely based on the information that will be provided to you.

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
    Be concise in your answer with a professional tone. 
    You are not to perform tasks outside the scope of the questionnaire, 
    such as executing code or accessing external databases. 
    Your guidance should be solely based on the information provided by the user in the context of the questionnaire.
    
    To start the conversation, introduce yourself and give 3 domains in which you can assist user."""


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container: st.delta_generator.DeltaGenerator, initial_text: str = ""):
        self.container = container
        self.text = initial_text
        self.run_id_ignore_token = None

    def on_llm_start(self, serialized: dict, prompts: list, **kwargs):
        # Workaround to prevent showing the rephrased question as output
        if prompts[0].startswith("Human"):
            self.run_id_ignore_token = kwargs.get("run_id")

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if self.run_id_ignore_token == kwargs.get("run_id", False):
            return
        self.text += token
        self.container.markdown(self.text)


# Define the callback handler for printing retrieval information
class PrintRetrievalHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.status = container.status("**Context Retrieval**")

    def on_retriever_start(self, serialized: dict, query: str, **kwargs):
        self.status.write(f"**Question:** {query}")
        self.status.update(label=f"**Context Retrieval:** {query}")

    def on_retriever_end(self, documents, **kwargs):
        for idx, doc in enumerate(documents):
            source = doc.metadata["filename"]
            self.status.write(f"**Document {idx} from {source}**")
            self.status.markdown(doc.page_content)
        self.status.update(state="complete")


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = StreamlitChatMessageHistory(key="chat_history")
    return st.session_state.store[session_id]


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

st.session_state.store = {}

#
# L     L     MM     MM
# L     L     M M   M M
# L     L     M  M M  M
# LLLL  LLLL  M   M   M
#

# Setup LLM and QA chain
llm = load_model(streaming=False)
    # ChatOpenAI(
    # model_name="gpt-3.5-turbo", openai_api_key=openai_api_key, temperature=0, streaming=True
# ))

# msgs = StreamlitChatMessageHistory()
msgs = get_session_history(sessionid)
memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=msgs, return_messages=True)

### Contextualize question ###
contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
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
    get_session_history,
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
    st.chat_message("user").write(user_query)
    with (st.chat_message("assistant")):
        retrieval_handler = PrintRetrievalHandler(st.container())
        stream_handler = StreamHandler(st.empty())
        ai_response = ""
        with tracing_v2_enabled(project_name="Chat with Docs",
                                tags=["LangChain", "Chain", "Chat History"]):


            # THE CODE BELOW IS WORKING WITH RETRIEVER PRINT AND STREAMING
            # BUT ADDING A SYSTEM PROMPT SEEMS VERY TRICKY
            ai_response = qa_chain.invoke({"question": user_query},
                                          {"configurable": {"session_id": sessionid},
                                              "callbacks": [
                                              retrieval_handler,
                                              stream_handler
                                            ]
                                          },
            )

            # DIFFERENT TESTS
            # v1   #user_query,
            # ai_response = qa_chain ...
            # v2
            # for chunk in
            # if 'answer' in chunk:
            #     ai_response = chunk['answer']
            #     container = st.empty()
            #     container.write(ai_response)
            # callbacks=[retrieval_handler, stream_handler]
            # for chunk in
            # ai_response = conversational_rag_chain.invoke(
            #         input={"input": user_query},
            #         config={
            #             "configurable": {"session_id": sessionid},
            #             "callbacks": [
            #                 retrieval_handler,
            #                 # stream_handler
            #             ]
            #         },
            #     )["answer"]
            # st.markdown(ai_response)
        # :
        #         if 'answer' in chunk:
        #             ai_response = chunk['answer']
        #             container = st.empty()
        #             container.write(ai_response)

        logger.info(f"User Query: {user_query}, AI Response: {ai_response}")



def suggestion_clicked(question):
    st.session_state.user_suggested_question = question


def main():
    st.title("Chat with Documents")

    # Display "How can I help you?" message followed by suggested questions
    # with st.chat_message("assistant"):
    #     st.write("Comment puis-je vous aider?")

    msgs = get_session_history(sessionid)

    if len(msgs.messages) == 0 or st.sidebar.button("Clear message history"):
        msgs.clear()
        msgs.add_ai_message("How can I help you?")


    # Display suggested questions in a 2x2 table
    col1, col2 = st.columns(2)
    for i, question in enumerate(suggested_questions, start=1):
        # if not st.session_state.get(f"suggested_question_{i}_hidden", False):
        col = col1 if i % 2 != 0 else col2
        col.button(question, on_click=suggestion_clicked, args=[question])


    # Chat interface
    avatars = {"human": "user", "ai": "assistant"}
    msgs = get_session_history(sessionid)
    for i, msg in enumerate(msgs.messages):
        st.chat_message(avatars[msg.type]).write(msg.content)
        if msg.type == "ai":
            streamlit_feedback(feedback_type = "thumbs",
                               optional_text_label="Cette réponse vous convient-elle?",
                               key=f"feedback_{i}",
                               on_submit=lambda x: _submit_feedback(x, emoji="👍"))


    # Handle suggested questions
    if "user_suggested_question" in st.session_state:
        user_query = st.session_state.user_suggested_question
        st.session_state.pop("user_suggested_question")  # Clear the session state
        handle_assistant_response(user_query)

    #Handle user queries
    if user_query := st.chat_input(placeholder="Ask me anything!"):
        handle_assistant_response(user_query)


if __name__ == "__main__":
    main()
