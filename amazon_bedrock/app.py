import streamlit as st
from query_against_openSearch import answer_query

#Header/Title
st.title("Amazon OpenSearch Chat With Docs")

#configuring values for session state
if "messages" not in st.session_state:
    st.session_state.messages = []
#writing the message that is store in the session state
for messages in st.session.state.messages:
    with st.chat_messages(messages["role"]):
        st.markdown(messages["content"])

# Evaluating st.chat_input and determining if a question has been input
if question:= st.chat_input("Ask a question"):
    #with the user icon, write the question to the frontend 
    with st.chat_messages("user"):
        st.markdown(question)
    # append teh question and the role (user) as a message to teh session state
    st.session_state.messages.append({"role": "user", "content": question})
    #respond as the assistant with the answer to the question
    with st.chat_messages("assistant"):
        # making sure there is no messages present when generating answer
        message_placeholder = st.empty()
        #putting a spinning icon to show that the assistant is working on the answer
        with st.status("Determining the best possible answer!", expanded=False) as status:
            #passing the question into theb opensearch query function, which later invokes the LLM to generate the answer
            answer = answer_query(question)
            #writing the answer to the frontend
            message_placeholder.markdown(f"{answer}")
            #showing a completion message to the front end
            status.update(label="Questrion Answered!...", state = "completed", expanded=False)
    # appending the answer to the session state as a message
    st.session_state.messages.append({"role": "assistant", "content": answer})