from typing import Sequence

from llama_index.core.base.base_query_engine import BaseQueryEngine
from pydantic import BaseModel, Field

from langchain.docstore.document import Document

from langchain.schema.prompt_template import format_document

from langchain.chains.combine_documents import collapse_docs, split_list_of_docs
from functools import partial
from operator import itemgetter

from langchain.callbacks.manager import trace_as_chain_group

from langchain.utils.openai_functions import convert_pydantic_to_openai_function
from langchain.output_parsers.openai_functions import PydanticOutputFunctionsParser

from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough


# LLAMA INDEX SUITE

from llama_index.core.tools import QueryEngineTool, ToolMetadata

from llama_index.core.query_engine import SubQuestionQueryEngine

from llama_index.agent.openai import OpenAIAgent
from llama_index.core.vector_stores.types import VectorStore


# https://python.langchain.com/docs/use_cases/question_answering/
# https://python.langchain.com/docs/modules/chains/document/stuff
# https://python.langchain.com/docs/modules/chains/document/map_reduce
# https://python.langchain.com/docs/modules/chains/document/refine
# https://python.langchain.com/docs/modules/chains/document/map_rerank
def invoke(question: str, template: str, llm: ChatOpenAI, chain_type: str, vectorstore: VectorStore,
           search_type: str, k: int, verbose: bool):

    retriever = vectorstore.as_retriever(search_type=search_type, search_kwargs={'k': k})
    docs = retriever.invoke(question)
    output = None

    if verbose:
        print(docs)

    document_prompt = PromptTemplate.from_template("{page_content}")
    partial_format_document = partial(format_document, prompt=document_prompt)
    # temporary to replace with incoming question

    map_prompt = PromptTemplate.from_template(
        "Answer the user question using the context."
        "\n\nContext:\n\n{context}\n\nQuestion: {question}"
    )

    rag_prompt_custom = PromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    if chain_type == "stuff":
        print("stuff chain")

        rag_chain = (
                {
                    "context": lambda x: "\n\n".join(
                        format_document(doc, document_prompt) for doc in x["docs"]
                    ),
                    "question": itemgetter("question"),
                }
                | map_prompt
                | llm
                | StrOutputParser()
        )
        output = rag_chain.invoke({"docs": docs, "question": question})

    elif chain_type == "map_reduce":

        print("map_reduce chain")

        # PromptTemplate.from_template("Summarize this content:\n\n{context}")
        first_prompt = PromptTemplate.from_template(
            "Answer the user question using the context."
            "\n\nContext:\n\n{context}\n\nQuestion: " + question
        )
        # first_prompt = first_prompt.format_prompt(question=question)

        # The chain we'll apply to each individual document.
        map_chain = (
                {"context": partial_format_document}
                | first_prompt
                | llm
                | StrOutputParser()
        )

        # A wrapper chain to keep the original Document metadata
        map_as_doc_chain = (
                RunnableParallel({"doc": RunnablePassthrough(), "context": map_chain})
                | (
                    lambda x: Document(page_content=x["context"], metadata=x["doc"].metadata)
                )
        ).with_config(run_name="Summarize (return doc)")

        # The chain we'll repeatedly apply to collapse subsets of the documents
        # into a consolidate document until the total token size of our
        # documents is below some max size.
        collapse_chain = (
                {"context": format_docs}
                | PromptTemplate.from_template("Collapse this content:\n\n{context}")
                | llm
                | StrOutputParser()
        )

        def get_num_tokens(docs):
            return llm.get_num_tokens(format_docs(docs))

        def collapse(
                docs,
                config,
                token_max=4000,
        ):
            collapse_ct = 1
            while get_num_tokens(docs) > token_max:
                config["run_name"] = f"Collapse {collapse_ct}"
                invoke = partial(collapse_chain.invoke, config=config)
                split_docs = split_list_of_docs(docs, get_num_tokens, token_max)
                docs = [collapse_docs(_docs, invoke) for _docs in split_docs]
                collapse_ct += 1
            return docs

        # The chain we'll use to combine our individual document summaries
        # (or summaries over subset of documents if we had to collapse the map results)
        # into a final summary.
        reduce_chain = (
                {"context": format_docs}
                | PromptTemplate.from_template("Combine these answers:\n\n{context}")
                | llm
                | StrOutputParser()
        ).with_config(run_name="Reduce")

        # The final full chain
        rag_chain = (map_as_doc_chain.map() | collapse | reduce_chain).with_config(
            run_name="Map reduce"
        )

        output = rag_chain.invoke(docs, config={"max_concurrency": 5})

    elif chain_type == "map_rerank":
        print("map_reduce chain")

        # Chain to apply to each individual document. Chain
        # provides an answer to the question based on the document
        # and scores it's confidence in the answer.
        class AnswerAndScore(BaseModel):
            """Return the answer to the question and a relevance score."""

            answer: str = Field(
                description="The answer to the question, which is based ONLY on the provided context."
            )
            score: float = Field(
                description="A 0.0-1.0 relevance score, where 1.0 indicates the provided context answers the question completely and 0.0 indicates the provided context does not answer the question at all."
            )

        function = convert_pydantic_to_openai_function(AnswerAndScore)
        map_chain = (
                map_prompt
                | ChatOpenAI().bind(
            temperature=0, functions=[function], function_call={"name": "AnswerAndScore"}
        )
                | PydanticOutputFunctionsParser(pydantic_schema=AnswerAndScore)
        ).with_config(run_name="Map")

        # Final chain, which after answer and scoring based on
        # each doc return the answer with the highest score.

        def top_answer(scored_answers):
            return max(scored_answers, key=lambda x: x.score).answer

        # document_prompt = PromptTemplate.from_template("{page_content}")
        rag_chain = (
                (
                    lambda x: [
                        {
                            "context": format_document(doc, document_prompt),
                            "question": question,  # x["question"]
                        }
                        for doc in x["docs"]
                    ]
                )
                | map_chain.map()
                | top_answer
        ).with_config(run_name="Map rerank")

        output = rag_chain.invoke({"docs": docs, "question": question})

    elif chain_type == "refine":
        # first_prompt = PromptTemplate.from_template("Summarize this content:\n\n{context}")
        first_prompt = PromptTemplate.from_template(
            "Answer the user question using the context."
            "\n\nContext:\n\n{context}\n\nQuestion: " + question
        )
        document_prompt = PromptTemplate.from_template("{page_content}")
        partial_format_doc = partial(format_document, prompt=document_prompt)
        summary_chain = {"context": partial_format_doc} | first_prompt | llm | StrOutputParser()
        refine_prompt = PromptTemplate.from_template(
            "Answer the user question."
            "\n\nHere's your first summary: {prev_response}. "
            "\n\nNow add to it based on the following context: {context}\n\nQuestion: " + question
        )
        refine_chain = (
                {
                    "prev_response": itemgetter("prev_response"),
                    "context": lambda x: partial_format_doc(x["doc"]),
                }
                | refine_prompt
                | llm
                | StrOutputParser()
        )

        def refine_loop(docs):
            with trace_as_chain_group("refine loop", inputs={"input": docs}) as manager:
                summary = summary_chain.invoke(
                    docs[0], config={"callbacks": manager, "run_name": "initial summary"}
                )
                for i, doc in enumerate(docs[1:]):
                    summary = refine_chain.invoke(
                        {"prev_response": summary, "doc": doc},
                        config={"callbacks": manager, "run_name": f"refine {i}"},
                    )
                manager.on_chain_end({"output": summary})
            return summary

        output = refine_loop(docs)
    return output

from llama_index.core import ServiceContext, VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.indices.loading import load_index_from_storage
import os


def build_sentence_window_index(
    documents, llm, embed_model="local:BAAI/bge-small-en-v1.5", save_dir="sentence_index"
):
    # create the sentence window node parser w/ default settings
    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=3,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )
    sentence_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model,
        node_parser=node_parser,
    )
    if not os.path.exists(save_dir):
        sentence_index = VectorStoreIndex.from_documents(
            documents, service_context=sentence_context
        )
        sentence_index.storage_context.persist(persist_dir=save_dir)
    else:
        sentence_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=save_dir),
            service_context=sentence_context,
        )

    return sentence_index


def get_sentence_window_query_engine(
    sentence_index,
    similarity_top_k=6,
    rerank_top_n=2,
):
    # define postprocessors
    postproc = MetadataReplacementPostProcessor(target_metadata_key="window")
    rerank = SentenceTransformerRerank(
        top_n=rerank_top_n, model="BAAI/bge-reranker-base"
    )

    sentence_window_engine = sentence_index.as_query_engine(
        similarity_top_k=similarity_top_k, node_postprocessors=[postproc, rerank]
    )
    return sentence_window_engine


from llama_index.core.node_parser import HierarchicalNodeParser

from llama_index.core.node_parser import get_leaf_nodes
from llama_index.core import StorageContext
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.query_engine import RetrieverQueryEngine


def build_automerging_index(
    documents,
    llm,
    embed_model="local:BAAI/bge-small-en-v1.5",
    save_dir="merging_index",
    chunk_sizes=None,
):
    chunk_sizes = chunk_sizes or [2048, 512, 128]
    node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=chunk_sizes)
    nodes = node_parser.get_nodes_from_documents(documents)
    leaf_nodes = get_leaf_nodes(nodes)
    merging_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model,
    )
    storage_context = StorageContext.from_defaults()
    storage_context.docstore.add_documents(nodes)

    if not os.path.exists(save_dir):
        automerging_index = VectorStoreIndex(
            leaf_nodes, storage_context=storage_context, service_context=merging_context
        )
        automerging_index.storage_context.persist(persist_dir=save_dir)
    else:
        automerging_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=save_dir),
            service_context=merging_context,
        )
    return automerging_index


def get_automerging_query_engine(
    automerging_index,
    similarity_top_k=12,
    rerank_top_n=2,
):
    base_retriever = automerging_index.as_retriever(similarity_top_k=similarity_top_k)
    retriever = AutoMergingRetriever(
        base_retriever, automerging_index.storage_context, verbose=True
    )

    rerank = SentenceTransformerRerank(
        top_n=rerank_top_n, model="BAAI/bge-reranker-base"
    )

    auto_merging_engine = RetrieverQueryEngine.from_args(
        retriever, node_postprocessors=[rerank]
    )

    return auto_merging_engine


def create_automerging_engine(llm, documents: Sequence[Document], embed_model: str):

    automerging_index = build_automerging_index(
        documents,
        llm,
        embed_model=embed_model,
        save_dir="merging_index"
    )

    automerging_query_engine = get_automerging_query_engine(
        automerging_index,
    )

    return automerging_query_engine


def create_automerging_agent(llm, documents: Sequence[Document], name:str, description: str, embed_model: str = "local:BAAI/bge-small-en-v1.5", query_engine:  BaseQueryEngine = None):

    if not query_engine:
        query_engine = create_automerging_engine(llm, documents, embed_model)

    agent_li = create_lli_agent(name, description, query_engine)
    return agent_li


def create_sentence_window_engine(llm, documents: Sequence[Document], embed_model: str):
    sentence_index = build_sentence_window_index(
        documents,
        llm,
        embed_model=embed_model,
        save_dir="sentence_index"
    )
    sentence_window_engine = get_sentence_window_query_engine(sentence_index)
    return sentence_window_engine


def create_sentence_window_agent(llm, documents: Sequence[Document], name:str, description: str, embed_model: str = "local:BAAI/bge-small-en-v1.5", query_engine: BaseQueryEngine = None):

    if query_engine is None:
        query_engine = create_sentence_window_engine(llm, documents, embed_model)

    agent_li = create_lli_agent(name, description, query_engine)

    return agent_li


def infer_topic_from_list(doc_name, topics):
    # Normalize the document name to lower case for case-insensitive matching
    doc_name_lower = doc_name.lower()
    for topic in topics:
        # Check if the topic is in the document name
        if topic.lower() in doc_name_lower:
            return topic
    return "Other"  # Default topic if no matches found


def create_subquery_engine(llm, topics: list[str], documents: Sequence[Document], embed_model: str):
    doc_set = {topic: [] for topic in topics}
    all_docs = []
    for doc in documents:
        topic = infer_topic_from_list(doc.metadata['file_path'], topics)
        doc_set[topic].append(doc)
        all_docs.append(doc)

    Settings.chunk_size = 512
    Settings.chunk_overlap = 64
    Settings.llm = llm
    Settings.embed_model = embed_model
    index_set = {}
    for topic in topics:
        # chroma_collection = db.get_or_create_collection(f"RAG_{topic}")
        # vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        # storage_context = StorageContext.from_defaults(vector_store=vector_store)
        storage_context = StorageContext.from_defaults()
        cur_index = VectorStoreIndex.from_documents(
            doc_set[topic],
            storage_context=storage_context,
        )
        index_set[topic] = cur_index
        storage_context.persist(persist_dir=f"./storage/{topic}")

    index_set = {}
    for topic in topics:
        storage_context = StorageContext.from_defaults(
            persist_dir=f"./storage/{topic}"
        )
        cur_index = load_index_from_storage(
            storage_context,
        )
        index_set[topic] = cur_index

    individual_query_engine_tools = [
        QueryEngineTool(
            query_engine=index_set[topic].as_query_engine(),
            metadata=ToolMetadata(
                name=f"vector_index_{topic}",
                description=f"useful for when you want to answer queries about {topic}",
            ),
        )
        for topic in topics
    ]

    # now I want to do the same with a list of BaseTool
    query_engine = SubQuestionQueryEngine.from_defaults(
        query_engine_tools=individual_query_engine_tools,
        llm=llm,
    )
    return query_engine


def create_subquery_agent(llm, topics: list[str], documents: Sequence[Document], name: str, description: str, embed_model: str = None, query_engine: BaseQueryEngine = None): #/-documents: Sequence[Document],  "local:BAAI/bge-small-en-v1.5"

    if query_engine is None:
        query_engine = create_subquery_engine(llm, topics, documents, embed_model)

    agent_li = create_lli_agent(name, description, query_engine)
    return agent_li


def create_direct_query_engine(llm, documents: Sequence[Document], embed_model: str):
    service_context = ServiceContext.from_defaults(
        llm=llm, embed_model=embed_model
    )
    index = VectorStoreIndex.from_documents(documents,
                                            service_context=service_context)
    query_engine = index.as_query_engine()
    return query_engine


def create_direct_query_agent(llm, documents: Sequence[Document], name:str, description: str, embed_model: str = "local:BAAI/bge-small-en-v1.5", query_engine: BaseQueryEngine = None):

    if query_engine is None:
        query_engine = create_direct_query_engine(llm, documents, embed_model)

    agent_li = create_lli_agent(name, description, query_engine)

    return agent_li


def create_lli_agent(name:str, description: str, query_engine: BaseQueryEngine):

    query_engine_tool = QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name=name,
            description=description,
        ),
    )
    agent_li = OpenAIAgent.from_tools(tools=[query_engine_tool], verbose=True)

    return agent_li