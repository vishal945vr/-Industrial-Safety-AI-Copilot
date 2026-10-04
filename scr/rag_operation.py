from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from model import embeddings
from langchain_community.vectorstores import FAISS
from pathlib import Path
# laod the embedding  model 

# 1. Load Embedding Model


def load_embeddings():

    embed = embeddings()

    return embed



# 2. Load PDF

def data_loader(folder_path):

    documents = []

    pdf_files = list(Path(folder_path).glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in: {folder_path}"
        )

    for pdf_file in pdf_files:

        print(f"Loading PDF: {pdf_file}")

        loader = PyPDFLoader(str(pdf_file))

        docs = loader.load()

        documents.extend(docs)

    print(f"Total PDF files: {len(pdf_files)}")
    print(f"Total pages loaded: {len(documents)}")

    return documents



# 3. Split Documents into Chunks


def chunks_splitter(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(documents)

    return chunks



# 4. Create FAISS Vector Database


def create_vectorstore(chunks, embedding):

    vector_db = FAISS.from_documents(
        documents=chunks,
        embedding=embedding
    )

    return vector_db



# 5. Save FAISS Vector Database


def save_vectorstore(vector_db, path="vectorstore"):

    vector_db.save_local(path)

    print(f"Vector database saved at: {path}")


