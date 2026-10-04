from langchain_community.vectorstores import FAISS
from .model import embeddings


def load_vectorstore():
    embed = embeddings()

    vectorstore = FAISS.load_local(
        "vectorstore",
        embed,
        allow_dangerous_deserialization=True
    )

    return vectorstore


def retrieverby_vst():
    # Load FAISS vector database
    vectorstore_local = load_vectorstore()

    # Create retriever
    retriever = vectorstore_local.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    return retriever


def retrieve_context(question):
    retriever = retrieverby_vst()

    documents = retriever.invoke(question)

    return documents