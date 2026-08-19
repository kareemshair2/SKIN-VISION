import os
from langchain_chroma import Chroma
from src.config import VECTOR_DB_DIR
from src.embeddings import get_embedding_model
from src.text_splitter import process_and_split_documents

def initialize_vector_db():
    """
    Initializes the Chroma vector database, processes raw documents,
    generates embeddings, and stores them locally with their metadata.
    """
    # 1. Get the embedding model
    embeddings = get_embedding_model()
    
    # 2. Process and split the raw text documents
    chunks = process_and_split_documents()
    
    if not chunks:
        print("No chunks to store in Vector DB.")
        return None

    # 3. Extract texts, metadatas, and ids for ChromaDB
    texts = [chunk["content"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]
    ids = [chunk["chunk_id"] for chunk in chunks]

    # 4. Create and persist the Chroma vector database
    print(f"Storing {len(texts)} chunks into Vector DB at {VECTOR_DB_DIR}...")
    vector_db = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        ids=ids,
        persist_directory=VECTOR_DB_DIR
    )
    
    print("Vector database initialized and persisted successfully.")
    return vector_db

def load_existing_vector_db():
    """
    Loads an already persisted Chroma vector database from disk.
    """
    embeddings = get_embedding_model()
    if os.path.exists(VECTOR_DB_DIR) and os.listdir(VECTOR_DB_DIR):
        vector_db = Chroma(
            persist_directory=VECTOR_DB_DIR,
            embedding_function=embeddings
        )
        print("Existing Vector DB loaded successfully.")
        return vector_db
    else:
        print("No existing Vector DB found. Creating a new one...")
        return initialize_vector_db()

if __name__ == "__main__":
    # Test loading or initializing the vector database
    db = load_existing_vector_db()