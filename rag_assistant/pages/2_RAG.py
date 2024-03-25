from json import JSONDecodeError

import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
from langchain.output_parsers import PydanticOutputParser
from langchain.schema.vectorstore import VectorStore
from langchain.tools.base import ToolException
from streamlit_chat import message
from pydantic import BaseModel
from typing import Optional
import streamlit_pydantic as sp
from langchain.globals import set_verbose

from langchain.chains import create_extraction_chain_pydantic, RetrievalQA
from langchain.docstore.document import Document

from langchain.chat_models import AzureChatOpenAI, ChatOpenAI
from langchain.embeddings import AzureOpenAIEmbeddings

from langchain.agents import tool

from utils import invoke
from utils import load_doc, load_store

from langchain.agents import AgentType, Tool, initialize_agent

from src.entity.models.dilitrust import Enterprise, Societe

__template__ = """Use the following pieces of context to answer the question at the end. 
   If you don't know the answer, just say that you don't know, don't try to make up an answer. 
   Use three sentences maximum and keep the answer as concise as possible. 
   Always answer in French. 
   {context}
   Question: {question}
   Helpful Answer:"""


_EXTRACTION_TEMPLATE = """Extract and save the relevant entities mentioned \
in the following passage together with their properties.

Only extract the properties mentioned in the 'information_extraction' function.

If a property is not present and is not required in the function parameters, do not include it in the output.

Passage:
{input}"""

# Pydantic data class
class _PersonneMorale(BaseModel):
    siren: Optional[str]
    siren_formate: Optional[str]
    date_immatriculation: Optional[str]
    raison_sociale: Optional[str]
    sigle: Optional[str]
    adresse_siege: Optional[str]
    activite: Optional[str]
    forme_juridique: Optional[str]
    capital_social: Optional[str]

class _Mandataire(BaseModel):
    nom: Optional[str]
    prenom: Optional[str]
    fonction: Optional[str]
    nomination: Optional[str]
    nationalite: Optional[str]
    date_de_naissance: Optional[str]
    lieu: Optional[str]
    domicile_personnel: Optional[str]





@tool
def get_word_length(word: str) -> int:
    """Returns the length of a word."""
    return len(word)


tools = [get_word_length]

def load_model(model: str):
    if st.secrets["OPENAI_API_TYPE"] == "azure":
        llm = AzureChatOpenAI(
            openai_api_version=st.secrets["AZURE_OPENAI_API_VERSION"],
            azure_endpoint=st.secrets["AZURE_OPENAI_ENDPOINT"],
            azure_deployment=st.secrets["AZURE_OPENAI_DEPLOYMENT"],
            api_key=st.secrets["AZURE_OPENAI_API_KEY"],
        )
    else:
        llm = ChatOpenAI(model_name=model, temperature=0)

    return llm


def load_embeddings(model: str):
    if st.secrets["OPENAI_API_TYPE"] == "azure":
        embeddings = AzureOpenAIEmbeddings(
            azure_deployment=st.secrets["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"],
            openai_api_version=st.secrets["AZURE_OPENAI_API_VERSION"],
        )
    else:
        embeddings = OpenAIEmbeddings()

    return embeddings


def load_sidebar():
    with st.sidebar:
        st.header("Parameters")
        st.sidebar.checkbox("Azure", st.secrets["OPENAI_API_TYPE"] == "azure", disabled=True)



def main():
    st.title("📄Personne Morale Extractor 🤗")
    load_sidebar()


    # for openai only
    model_name = st.sidebar.radio("Model", ["gpt-3.5-turbo-1106", "gpt-4-32k-0613", "gpt-4-1106-preview", "gpt-4-0613"],
                             captions=["GPT 3.5 Turbo (16k;4k)", "GPT-4-32k (32k;4k?)", "GPT-4 Turbo (128k;4k)",
                                       "GPT-4 (8k;4k?)"], index=1, disabled=st.secrets["OPENAI_API_TYPE"] == "azure")

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
    set_verbose(verbose)

    option = st.radio("Look at", ["Extract KBIS","Chat with doc", "Extract Status", "Extract Agents"], index=None)  # , "Papper"

    # llm = load_model(model_name)
    embeddings = load_embeddings(model_name)

    # upload a your pdf file

    if option is not None:
        pdfs = st.file_uploader("Upload Doc", type='pdf', accept_multiple_files=True)

        company_name = "ENGIE IT SA"
        siren = 340793959
        dt_code = 20213

        company_name = st.text_input("Company name", company_name)

        if (pdfs is not None) and (len(pdfs)):
            docs = load_doc(pdfs)
            if option == 'Extract KBIS':
                llm = load_model(model_name)

                chain = create_extraction_chain_pydantic(pydantic_schema=Societe, llm=llm, verbose=verbose)
                extracts = chain.run(docs)

                societe: Societe = extracts[0]

                st.header("Societe")

                st.subheader("Entreprise")
                sp.pydantic_output(societe.enterprise)

                st.subheader("Identifications")
                sp.pydantic_output(societe.identification)

            elif option == 'Extract Status':

                store = load_store(docs, embeddings)

                user_input = "Extrait les informations sur les mandataires sociaux et personnes morales." \
                             "\n\nLes caractéristiques d un mandataire social sont:" \
                             "\n- nom: Optional[str]" \
                             "\n- prénom: Optional[str]" \
                             "\n- fonction: Optional[str]" \
                             "\n- nomination: Optional[str]" \
                             "\n- nationalité: Optional[str]" \
                             "\n- date de naissance: Optional[str]" \
                             "\n- lieu de naissance: Optional[str]" \
                             "\n- domicile personnel: Optional[str]" \
                             "\n\nLes caractéristiques d une personne morale sont:" \
                             "\n- numéro siren formate: [str]" \
                             "\n- date d immatriculation: [str]" \
                             "\n- raison sociale: [str]" \
                             "\n- sigle: Optionel[str]" \
                             "\n- adresse_siege: Optional[str]" \
                             "\n- activité: Optional[str]" \
                             "\n- forme_juridique: Optional[str]" \
                             "\n- capital_social: Optional[str]"

                llm = load_model(model_name)

                output = invoke(user_input, _EXTRACTION_TEMPLATE, llm, chain_type, store, search_type, k, verbose)

                docs = [
                    Document(
                        page_content=split,
                        metadata={"source": "Previous search"},
                    )
                    for split in output.split("\n\n")
                ]

                chain = create_extraction_chain_pydantic(pydantic_schema=Societe, llm=llm, verbose=verbose)
                extracts = chain.run(docs)

                if not extracts:
                        st.text("No data retrieved.")

                else:
                    societe:Societe = extracts[0]

                    st.header("Societe")

                    st.subheader("Entreprise")
                    sp.pydantic_output(societe.enterprise)

                    st.subheader("Identification")
                    sp.pydantic_output(societe.identification)

            elif option == 'Chat with doc':

                llm = load_model(model_name)

                st.header("Question Answering Assistant")

                if "generated" not in st.session_state:
                    st.session_state["generated"] = []

                if "past" not in st.session_state:
                    st.session_state["past"] = []

                with st.form(key="form"):
                    user_input = st.text_input("You: ", "Hello, what do you want to know?", key="input")
                    submit_button_pressed = st.form_submit_button("Submit to Bot")

                if submit_button_pressed:

                    # result = chain({"question": user_input})
                    # output = f"Answer: {result['answer']}"      # \nSources: {result['sources']}

                    store: VectorStore = load_store(docs, embeddings)

                    output = invoke(user_input, template, llm, chain_type, store, search_type, k, verbose)

                    st.session_state.past.append(user_input)
                    st.session_state.generated.append(output)

                if st.session_state["generated"]:

                    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
                        message(st.session_state["generated"][i], key=str(i))
                        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")

            elif option == 'Extract Agents':
                store: VectorStore = load_store(docs, embeddings)

                retriever = store.as_retriever(search_type=search_type, search_kwargs={'k': k})

                llm1 = load_model(model_name)

                corporate = RetrievalQA.from_chain_type(
                    llm=llm1, chain_type=chain_type, retriever=retriever
                )

                def _handle_error(error: ToolException) -> str:
                    if error == JSONDecodeError:
                        return "Reformat in JSON and try again"
                    elif error.args[0].startswith("Too many arguments to single-input tool"):
                        return "Format in a SINGLE STRING. DO NOT USE MULTI-ARGUMENTS INPUT."
                    return (
                            "The following errors occurred during tool execution:"
                            + error.args[0]
                            + "Please try another tool.")

                tools = [
                    Tool(
                        name="Corporate QA System",
                        func=corporate.run,
                        description="Useful when you need to answer questions about"
                                    " company, corporate, representative and shareholder. "
                                    "DO NOT USE MULTI-ARGUMENTS INPUT.",
                        handle_tool_error=_handle_error,
                    ),
                    # get_dilitrust_by_dt_code,
                    # get_dilitrust_by_usual_name,
                    # get_entreprise_by_siren
                ]

                llm3 = load_model(model_name)
                agent = initialize_agent(
                    tools, llm3, agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True
                )

                pydantic_parser = PydanticOutputParser(pydantic_object=Enterprise)
                format_instructions = pydantic_parser.get_format_instructions()
                # "\n\n{format_instructions}"

                fields_mandataire = f"""\nCharacteristics of a corporate officer are last name, first name, "
                                         " function, nomination ,"
                                         " nationality, date of birth, "
                                         " birth place, personal home."""
                fields_entity = f"""\nCharacteristics of a legal entity are "
                                         " siren number or any legal identifier, "
                                         " registration date, company name, legal name"
                                         " acronym, headquarter address, "
                                         " activity, legal form, capital social."""

                # "\n\nFirst identify the legal entity mainly concerned with Corporate QA system. "
                # "\n\nNext, identify the legal entity mainly concerned by the context. "
                # with SIREN {siren}
                # and dt_code {dt_code}
                data = agent.run(
                    f"""Your goal is to extract legal information for a corporate which usual name is {company_name}. " 
                                                     "\n\nTo do so, you'll have to identify all entities that are related to the corporate "
                                                     " such as representatives, lawyers, shareholders, auditors with Corporate QA system."
                                                     "\n\nNext you'll have to collect characteristics about each entities one by one."
                                                     "\n\nPack all collected information from each entities into a single "
                                                     " text document starting with the main legal entity. 
                                                     "\nDo not make up information. Use only data provided in your context.
                                                     "\nDo not use smart code to retrieve data with tools."
                                                     "\nIt is ok if some data is missing.""")
                # print(data)
                # "IF fields are still missing, you can look at pappers with siren to complete missing data."
                # "While doing so, identify the fields that mismatch from previous retrieval and list them."
                # "Finally, you can use dilitrust to complete missing data."
                # "Again while doing so, identify the fields that mismatch from previous retrieval and list them."

                print(data)

                # societe = Enterprise(** json.loads(data))
                # print(societe)

                extract_chain = create_extraction_chain_pydantic(pydantic_schema=Enterprise, llm=llm1, verbose=verbose)
                extracts = extract_chain.run(data)
                corporate = extracts[0]

                st.header("Société")

                st.subheader("Identification")
                sp.pydantic_output(corporate.identification)

                st.subheader("Données Générales")
                sp.pydantic_output(corporate.enterprise)

                st.subheader("Mandataires")
                sp.pydantic_output(corporate.mandataires)

        # requests_wrapper = RequestsWrapper(headers=headers)

        # data = sp.pydantic_form(key="personne_morale_form", model_name=PersonneMorale)
        # if data:
        #    st.json(data.json())


if __name__ == "__main__":
    main()
