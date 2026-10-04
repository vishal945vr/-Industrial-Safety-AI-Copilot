# llm  libarys 
import os
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings


from dotenv import load_dotenv
load_dotenv()
#llm 
qroq_llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)
# embedding model
embeddings_huggingface = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)



def get_llm():

    return ChatGroq(

        model="openai/gpt-oss-120b",

        temperature=0,

        max_retries=2
    )


# llm
def llm():
    return qroq_llm

# embedding
def embeddings():
    return embeddings_huggingface

#  llm invoke 

def llm_invoke(question):
    groq_llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0
    )

    response = groq_llm.invoke(question)

    return response.content

