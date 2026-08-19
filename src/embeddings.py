from langchain_huggingface import HuggingFaceEmbeddings

def get_embedding_model():
    """
    Initializes and returns the embedding model.
    Using HuggingFace embeddings for local, fast, and cost-effective vector generation.
    """
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Configure model parameters if needed (e.g., running on CPU)
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}
    
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    
    print("Embedding model initialized successfully.")
    return embeddings

if __name__ == "__main__":
    # Test embedding generation
    emb_model = get_embedding_model()
    sample_vector = emb_model.embed_query("Test medical document embedding")
    print(f"Embedding vector dimension: {len(sample_vector)}")