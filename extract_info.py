import argparse
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama

from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """

The following text is extracted from a scientific document about skin care. 
Your task is to identify and extract information related to the ingredient '{ingredient}'. 
For this ingredient, list all biological mechanisms, active molecules, and target biological systems or cells mentioned in the text. 
Structure the extracted information in a JSON format as shown below:

Text:
{text}

JSON output:
{{
    "Ingredient": "{ingredient}",
    "Biological Mechanisms": [
        {{
            "Mechanism": "<mechanism>",
            "Active Molecules": ["<molecule1>", "<molecule2>", ...],
            "Targets": ["<target1>", "<target2>", ...]
        }},
        ...
    ]
}}
"""


def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("ingredient", type=str, help="The ingredient.")
    args = parser.parse_args()
    ingredient = args.ingredient
    query_rag(ingredient)
    
def query_rag(ingredient: str):
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(ingredient, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(text=context_text, ingredient=ingredient) 
    
    