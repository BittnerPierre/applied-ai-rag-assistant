import os
import openai
import tempfile
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv, find_dotenv
from utils.utilsdoc import get_store
from utils.config_loader import load_config
from streamlit_feedback import streamlit_feedback
import logging


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

st.set_page_config(page_title="Finaxys: Chat with Documents", page_icon="🦜")
st.title("Finaxys: Chat with Documents")


# Define paths for PDF files
pdf_files_paths = [
    "data/sources/pdf/aws/waf/AWS_Well-Architected_Framework.pdf",
    "data/sources/pdf/Questionnaire d'évaluation des risques applicatifs pour le Cloud Public.pdf",
    "data/sources/pdf/enisa/Cloud Security Guide for SMEs.pdf",
    "data/sources/pdf/aws/caf/aws-caf-for-ai.pdf",
    # Add more paths as needed
]


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


@st.cache_resource(ttl="1h")
def configure_retriever(pdf_files_paths):
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


def handle_assistant_response(user_query):
    with st.chat_message("Assistant"):
        retrieval_handler = PrintRetrievalHandler(st.container())
        stream_handler = StreamHandler(st.empty(), initial_system_prompt=__template2__)
        ai_response = qa_chain.run(user_query, callbacks=[retrieval_handler, stream_handler])
        logger.info(f"User Query: {user_query}, AI Response: {ai_response}")


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container: st.delta_generator.DeltaGenerator, initial_text: str = "", initial_system_prompt: str = ""):
        self.container = container
        self.text = initial_text
        self.initial_system_prompt = initial_system_prompt
        self.run_id_ignore_token = None

    def on_llm_start(self, serialized: dict, prompts: list, **kwargs):
        # Workaround to prevent showing the rephrased question as output
        if prompts[0].startswith("Human"):
            self.run_id_ignore_token = kwargs.get("run_id")
            user_prompt = prompts[0]  # La requête de l'utilisateur
            combined_prompt = f"{self.initial_system_prompt}\n\n{user_prompt}"
            prompts[0] = combined_prompt  # Mettre à jour la requête avec la combinaison

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
            source = os.path.basename(doc.metadata["source"])
            self.status.write(f"**Document {idx} from {source}**")
            self.status.markdown(doc.page_content)
        self.status.update(state="complete")

# Configure the retriever with PDF files
retriever = configure_retriever(pdf_files_paths)

# Setup memory for contextual conversation
msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=msgs, return_messages=True)

# Setup LLM and QA chain
load_dotenv(find_dotenv())
openai_api_key = os.getenv('OPENAI_API_KEY')
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo", openai_api_key=openai_api_key, temperature=0, streaming=True
)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm, retriever=retriever, memory=memory, verbose=True
)

suggested_questions = [
    "Comment sécuriser les données sensibles ?",
    "Quelles stratégies pour une haute disponibilité ?",
    "Quels sont les mécanismes d'authentification API ?",
    "Comment assurez l'efficacité des performances ?",
]

# Display "How can I help you?" message followed by suggested questions
with st.chat_message("assistant"):
    st.write("Comment puis-je vous aider?")

# Display suggested questions in a 2x2 table
col1, col2 = st.columns(2)
for i, question in enumerate(suggested_questions, start=1):
    if not st.session_state.get(f"suggested_question_{i}_hidden", False):
        col = col1 if i % 2 != 0 else col2
        if col.button(question):
            st.session_state.user_query = question


# Chat interface
avatars = {"human": "user", "ai": "assistant"}
for i, msg in enumerate(msgs.messages):
    st.chat_message(avatars[msg.type]).write(msg.content)
    if msg.type == "ai":
        streamlit_feedback(feedback_type = "thumbs",
                                 optional_text_label="[Optional]Est ce que cette reponse vous convient?",
                                 key=f"feedback_{i}",
                                 on_submit=lambda x: _submit_feedback(x, emoji="👍"))


# Handle suggested questions§
if "user_query" in st.session_state:
    user_query = st.session_state.user_query
    st.session_state.pop("user_query")  # Clear the session state
    st.chat_message("user").write(user_query)
    handle_assistant_response(user_query)
    st.rerun()

#Handle user queries
if user_query := st.chat_input(placeholder="Ask me anything!"):
    st.chat_message("user").write(user_query)
    handle_assistant_response(user_query)
    st.rerun()