# applied-ai-rag-assistant
Assistant RAG Advanced with Streamlit, Langchain, LlamaIndex and ChromaDB

Initially forked from https://github.com/langchain-ai/streamlit-agent/ `chat_with_documents.py`

Apps feature LangChain 🤝 Streamlit integrations such as the
[Callback integration](https://python.langchain.com/docs/modules/callbacks/integrations/streamlit) and
[StreamlitChatMessageHistory](https://python.langchain.com/docs/integrations/memory/streamlit_chat_message_history).


## Setup

This project uses [Poetry](https://python-poetry.org/) for dependency management.

```shell
# Create Python environment
$ poetry install

# Install git pre-commit hooks
$ poetry shell
$ pre-commit install
```

## Running

```shell
# Run mrkl_demo.py or another app the same way
$ streamlit run streamlit_agent/chat_with_documents.py
```