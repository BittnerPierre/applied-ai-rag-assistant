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

    template = st.sidebar.text_area("Prompt", __template__)

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

            # stream = client.chat.completions.create(
            #     messages=[
            #         {"role": m["role"], "content": m["content"]}
            #         for m in st.session_state.messages
            #     ],
            #     model=st.session_state["openai_model"],
            #     stream=True,
            # )
            # response = st.write_stream(stream)
            st.write(output)
            st.session_state.messages.append({"role": "assistant", "content": output})

    # if "generated" not in st.session_state:
    #     st.session_state["generated"] = []
    #
    # if "past" not in st.session_state:
    #     st.session_state["past"] = []

    # with st.form(key="form"):
    #     user_input = st.text_input("You: ", "Hello, what do you want to know?", key="input")
    #     submit_button_pressed = st.form_submit_button("Submit to Bot")
    #
    # if submit_button_pressed:
    #
    #     # result = chain({"question": user_input})
    #     # output = f"Answer: {result['answer']}"      # \nSources: {result['sources']}
    #
    #     store: VectorStore = get_store(embeddings)
    #
    #     output = invoke(user_input, template, llm, chain_type, store, search_type, k, verbose)
    #
    #     st.session_state.past.append(user_input)
    #     st.session_state.generated.append(output)
    #
    # if st.session_state["generated"]:
    #
    #     for i in range(len(st.session_state["generated"]) - 1, -1, -1):
    #         with st.chat_message()
    #             message(st.session_state["generated"][i], key=str(i))
    #             message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")

        # requests_wrapper = RequestsWrapper(headers=headers)

        # data = sp.pydantic_form(key="personne_morale_form", model_name=PersonneMorale)
        # if data:
        #    st.json(data.json())


if __name__ == "__main__":
    main()
