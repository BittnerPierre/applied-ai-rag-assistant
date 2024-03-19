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

# Running with Docker

This project includes `Dockerfile` to run the app in Docker container. In order to optimise the Docker Image is optimised for size and building time with cache techniques.

To generate Image with `DOCKER_BUILDKIT`, follow below command

```DOCKER_BUILDKIT=1 docker build --target=runtime . -t applied-ai-rag-assistant:latest```

1. Run the docker container directly

``docker run -d --name langchain-streamlit-agent -p 8051:8051 applied-ai-rag-assistant:latest ``

2. Run the docker container using docker-compose (Recommended)

Edit the Command in `docker-compose` with target streamlit app

``docker-compose up``
