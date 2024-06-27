from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.embeddings.bedrock import BedrockEmbeddings


def get_embedding_function():
    # embeddings = BedrockEmbeddings(
    #     credentials_profile_name="default", region_name="us-east-1"
    # )
    embeddings = OllamaEmbeddings(model="nomic-embed-text") # it permits to run open source LLMs locally
    return embeddings

# to load model locally on cmd -> ollama pull llama2    (mistral)
# to run model locally on cmd -> ollama run llama2
# use ollama -> ollama serve

