from langchain_community.retrievers import BM25Retriever
from .vectordatabase import load_vectorstore


def hybrid_sercher():

    # Load FAISS vector database
    vectorstore = load_vectorstore()
    vst_rter = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    documents = list(vectorstore.docstore._dict.values())
    Bh25 = BM25Retriever.from_documents(documents)
    Bh25.k = 1

    return Bh25, vst_rter


def retrieve_hybrid_info(question):

    Bh25, vst_rter = hybrid_sercher()

    # BM25 results
    bm25_docs = Bh25.invoke(question)

    # Vector results
    vector_docs = vst_rter.invoke(question)

    

    combined_documents = []

    # BM25 = 40%
    for doc in bm25_docs:
        if doc not in combined_documents:
            combined_documents.append(doc)

    # Vector = 60%
    for doc in vector_docs:
        if doc not in combined_documents:
            combined_documents.append(doc)

    return combined_documents




    