# applied-ai-rag-assistant
Assistant RAG Advanced with Streamlit, Langchain, LlamaIndex and ChromaDB

Initially forked from https://github.com/langchain-ai/streamlit-agent/ `chat_with_documents.py`

Apps feature LangChain 🤝 Streamlit integrations such as the
[Callback integration](https://python.langchain.com/docs/modules/callbacks/integrations/streamlit) and
[StreamlitChatMessageHistory](https://python.langchain.com/docs/integrations/memory/streamlit_chat_message_history).

Now we have added Mistral La Plateforme, Bedrock, llamaindex and langchain agent for advanced RAG, model vision on RAG with anthropic claude.

## Setup

This project uses [Poetry](https://python-poetry.org/) for dependency management.

```shell
# Create Python environment
$ poetry install

# Install git pre-commit hooks
$ poetry shell
$ pre-commit install
```

### Note on package dependencies
For now, we are not forcing package's version in poetry and try to upgrade as fast as we can. :)
As we are using a lot of new and young "GENAI" component that have not finalized their interface,
application and tests tends to break a lot and often especially they are not testing evolution with each other.

Main packages are:
- Langchain (LLM Orchestration and agent)
- LlamaIndex (RAG)
- Streamlit (UX)
- TruLens (Testing)
- Chroma (Vector Store)
- OpenAI (LLM)
- MistralAI (LLM)
- boto3 (for bedrock and AWS integration)

## Running

### Environment variables
The project expects some environment variables to be setup in order to run.
Some are mandatory for running and some are only needed if you want to run on a specific platform.

The project currently supports the following platforms: OPENAI, AZURE, MISTRAL, BEDROCK (AWS).

We recommend to add the variables in a .env file within the directory path outside the project directory to avoid any accidental commit.
Your home directory is fine.

Here are the variables:

```shell
OPENAI_API_KEY=<YOUR_OPENAI_API_KEY>
MISTRAL_API_KEY=<YOUR_MISTRAL_API_KEY>
AZURE_OPENAI_API_KEY=<YOUR_AZURE_OPENAI_API_KEY>
HF_TOKEN=<YOUR_HUGGING_FACE_TOKEN>
LANGCHAIN_TRACING_V2=<true or false>
LANGCHAIN_API_KEY=<YOUR_LANGCHAIN_API_KEY>
```

### MISTRAL PLATFORM
If you want to use MISTRAL PLATFORM, you need a MISTRAL_API_KEY and a HF_TOKEN.
HF_TOKEN is required to download the embeddings from hugging face.
It is done automatically but you need to have the HF_TOKEN and to have granted access on the model page on hugging face.
https://huggingface.co/mistralai/Mixtral-8x7B-v0.1


### LANGSMITH and LLM OBSERVABILITY
We are using LANGSMITH for LLM Observability. 
Langsmith requires LANGCHAIN_TRACING_V2 and a LANGCHAIN_API_KEY.

You can stop tracing with 'LANGCHAIN_TRACING_V2=false'.
Oddly 'LANGCHAIN_API_KEY' is still required even if you set 'LANGCHAIN_TRACING_V2' to false. 
But you can put anything in it, the variable should only exist.

LANGSMITH is free for personal use with a quota limit of 5k traces per month. 
It is very useful so I recommend it to you.

https://smith.langchain.com/

### AWS BEDROCK
If you want to use Bedrock (AWS), you can define your credential in $HOME/.aws/credentials directory
We use eu-west-3 and eu-central-1 for  claude anthropic, mistral large and titan embeddings within bedrock.
Adapt it to your own needs. Beware that models are not consistently deployed within AWS region. 


### MODEL VISION
We are starting to add model vision support in our assistant.
For now, we are only supporting CLAUDE 3 vision with BEDROCK AWS.


## Config
Most parameters like model name, region, etc. can be modified in conf/config.ini for all model providers.


## Testing
We use pytest and Trulens to evaluate the assistant (RAG Triad). 

For RAG testing, we are using OpenAI as provider for trulens feedback function so you need at least openai api key to make it work. 
But you can adapt it for your own purpose.

Tests in tests/utils/ directory use Mistral Large through 'La Platforme' so you'll MISTRAL_API_KEY.
Tests in tests/rag/ use bedrock (AWS) and openai GPT. So you'll need OPENAI_API_KEY and AWS credentials.


```shell
# Run mrkl_demo.py or another app the same way
$ streamlit run streamlit_agent/0_Chat_with_Docs.py
```

# Running with Docker (OLD)

This project includes `Dockerfile` to run the app in Docker container. In order to optimise the Docker Image is optimised for size and building time with cache techniques.

To generate Image with `DOCKER_BUILDKIT`, follow below command

```DOCKER_BUILDKIT=1 docker build --target=runtime . -t applied-ai-rag-assistant:latest```

1. Run the docker container directly

``docker run -d --name applied-ai-rag-assistant -p 8051:8051 applied-ai-rag-assistant:latest ``

2. Run the docker container using docker-compose (Recommended)

Edit the Command in `docker-compose` with target streamlit app

``docker-compose up``

## Run the app with Docker

Build the image:
```sh
docker build -t ai_assistant .
```

Then run a container from the image we just created :
```sh
docker run -p 80:80 -e OPENAI_API_KEY="secret_value" ai_assistant
```
Replace secret_value with your openai key. 

The application should run on http://localhost:80/

## Run the app on AWS

Install AWS CLI : https://docs.aws.amazon.com/fr_fr/cli/latest/userguide/getting-started-install.html
Install Docker : https://docs.docker.com/engine/install/

Build and push the Docker image on the AWS Elastic Container Registry (ECR) with AWS CLI:
```sh
docker build -t ai_assistant .
aws configure set aws_access_key_id "access-key"
aws configure set aws_secret_access_key "secret-access-key"
aws ecr get-login-password --region eu-west-1 | docker login --username AWS --password-stdin 441525731509.dkr.ecr.eu-west-1.amazonaws.com
docker tag ai_assistant:latest 441525731509.dkr.ecr.eu-west-1.amazonaws.com/ai_assistant:latest
docker push 441525731509.dkr.ecr.eu-west-1.amazonaws.com/ai_assistant:latest
```
Replace access-key and secret-access-key with valid AWS credentials that will be used to push to the ECR. The AWS user must have the correct rights to push image on the ECR.

Once the image pushed on the ECR, go to the terraform directory with (make sure to meet the basic requirements in AWS so that the terraform files works (see AWS resources requirements)) :
```sh
cd terraform
```
and run:
```sh
export AWS_ACCESS_KEY="access-key"
export AWS_SECRET_ACCESS_KEY="secret-access-key"
terraform init
terraform plan
terraform apply
```
Terraform is needed : https://developer.hashicorp.com/terraform/tutorials/aws-get-started/install-cli
Replace access-key and secret-access-key with valid AWS credentials that will be used to create the resources. The account must have all the necessary rights to create and access the resources needed.

You will find the ECS cluster here : https://eu-west-1.console.aws.amazon.com/ecs/v2/clusters?region=eu-west-1

To redeploy the service with the latest version of the application, go to : https://eu-west-1.console.aws.amazon.com/ecs/v2/clusters/ai_assistant/services?region=eu-west-1.
Select your running service in the services list
Click on Update.
Check Force new Deployment, and click on Update.
The latest version of the image will be deployed with the new service.