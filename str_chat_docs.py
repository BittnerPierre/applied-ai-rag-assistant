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
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv

st.set_page_config(page_title="Finaxys: Chat with Documents", page_icon="🦜")
st.title("Finaxys: Chat with Documents")

# Display the image icon along with the app title
st.image("/Users/loicsteve/Downloads/LOGO.b42ce8d.svg", use_column_width=True) 

# Define paths for PDF files
pdf_files_paths = [
    "/Users/loicsteve/Desktop/HrFlow/Is Reinforcement Learning (not ) for Natural Language Processing.pdf",
    #"/path/to/your/pdf/file2.pdf",
    # Add more paths as needed
]


@st.cache_resource(ttl="1h")
def configure_retriever(pdf_files_paths):
    # Read documents
    docs = []
    for file_path in pdf_files_paths:
        loader = PyPDFLoader(file_path)
        docs.extend(loader.load())

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # Create embeddings and store in vectordb
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    persist_directory = "/Users/loicsteve/Desktop/LLM Engineer Roadmap/LCL Project/docs/chroma/"
    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embedding,
        persist_directory=persist_directory
    )

    # Define retriever
    retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 2, "fetch_k": 4})

    return retriever


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
        self.container = container

    def on_retriever_start(self, serialized: dict, query: str, **kwargs):
        self.container.write(f"**Question:** {query}")

    def on_retriever_end(self, documents, **kwargs):
        for idx, doc in enumerate(documents):
            source = os.path.basename(doc.metadata["source"])
            self.container.write(f"**Document {idx} from {source}**")
            self.container.markdown(doc.page_content)


# Define paths to your PDF files directly
# pdf_files_paths = [
#     "/Users/loicsteve/Desktop/HrFlow/Is Reinforcement Learning (not ) for Natural Language Processing.pdf",
#     #"/path/to/your/pdf/file2.pdf",
#     # Add more paths as needed
# ]

# Configure the retriever with PDF files
retriever = configure_retriever(pdf_files_paths)

# Setup memory for contextual conversation
msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=msgs, return_messages=True)

# Setup LLM and QA chain
load_dotenv(dotenv_path='/Users/loicsteve/Desktop/LLM Engineer Roadmap/LCL Project/key.env')
openai_api_key = os.getenv('OPENAI_API_KEY')
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo", openai_api_key=openai_api_key, temperature=0, streaming=True
)
qa_chain = ConversationalRetrievalChain.from_llm(
    llm, retriever=retriever, memory=memory, verbose=True
)

# # Add "How can I help you?" message to chat history
# if len(msgs.messages) == 0:
#     msgs.add_ai_message("How can I help you?")

# Define suggested questions
suggested_questions = [
    "What is reinforcement learning?",
    "What is reinforcement learning for human feedback?",
    "How to process reinforcement learning?",
    "What are the advantages of reinforcement learning?"
]

# Display "How can I help you?" message followed by suggested questions
with st.chat_message("assistant"):
    st.write("How can I help you?")

for i, question in enumerate(suggested_questions, start=1):
    if st.button(f"Suggested Question {i}: {question}", key=f"suggested_question_{i}"):
        # Perform action upon clicking the button (e.g., send the question to the chatbot)
        user_query = question

        with st.chat_message("assistant"):
            stream_handler = StreamHandler(st.empty())
            response = qa_chain.run(user_query, callbacks=[stream_handler])

# Chat interface
avatars = {"human": "user", "ai": "assistant"}
for msg in msgs.messages:
    st.chat_message(avatars[msg.type]).write(msg.content)

if user_query := st.chat_input(placeholder="Ask me anything!"):
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        retrieval_handler = PrintRetrievalHandler(st.container())
        stream_handler = StreamHandler(st.empty())
        response = qa_chain.run(user_query, callbacks=[retrieval_handler, stream_handler])
