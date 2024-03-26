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