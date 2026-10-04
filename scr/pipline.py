
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from rag_operation import save_vectorstore,load_embeddings,chunks_splitter,create_vectorstore,data_loader
from pathlib import Path

# file s path 
data_files=Path("data").glob("*.pdf")
for f in data_files:
    print(f)

# Complete RAG Data Pipeline



def create_rag_database(pdf_path):

    # Load embedding model
    embedding = load_embeddings()

    # Load PDF
    documents = data_loader(pdf_path)

    print(f"Loaded documents: {len(documents)}")

    # Split PDF into chunks
    chunks = chunks_splitter(documents)

    print(f"Created chunks: {len(chunks)}")

    # Create FAISS vector database
    vector_db = create_vectorstore(
        chunks,
        embedding
    )

    # Save vector database
    save_vectorstore(
        vector_db,
        "vectorstore"
    )

    return vector_db


# --------------------------------------------------
# 7. Main Function
# --------------------------------------------------

def rag_pipeline():

    pdf_path = Path(r"D:\rag_project\data")

    vector_db = create_rag_database(pdf_path)

    print("RAG vector database created successfully")




if __name__ == "__main__":
    rag_pipeline()
