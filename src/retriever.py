import os
from typing import List, Dict, Any
from src.vector_db import load_existing_vector_db

def get_relevant_context(query: str, k: int = 4) -> List[Dict[str, Any]]:
    """
    Searches the vector database for the top-k most relevant text chunks 
    matching the user query across all available diseases. 
    Extracts content, similarity scores, and metadata including image paths.
    
    Args:
        query (str): The search query or symptom description provided by the user.
        k (int): Number of top relevant chunks to retrieve. Default is 4.
        
    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing retrieved text, metadata, and scores.
    """
    # Load the persisted Chroma vector database
    vector_db = load_existing_vector_db()
    
    if not vector_db:
        print("Error: Vector database is empty or could not be loaded.")
        return []

    try:
        # Perform similarity search with scores to evaluate relevance
        results = vector_db.similarity_search_with_score(query, k=k)
        
        retrieved_data = []
        for doc, score in results:
            chunk_info = {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "similarity_score": float(score)
            }
            retrieved_data.append(chunk_info)
            
        print(f"Successfully retrieved {len(retrieved_data)} relevant context chunks for the query.")
        return retrieved_data
        
    except Exception as e:
        print(f"An error occurred during retrieval: {e}")
        return []

def get_disease_specific_context(disease_id: str, k: int = 3) -> List[Dict[str, Any]]:
    """
    Retrieves chunks specifically filtered by a target disease identifier 
    (e.g., 'disease_1', 'disease_2', 'disease_3', 'disease_4').
    
    Args:
        disease_id (str): The specific disease identifier to filter by.
        k (int): Number of chunks to retrieve.
        
    Returns:
        List[Dict[str, Any]]: Filtered list of chunks related to the specific disease.
    """
    vector_db = load_existing_vector_db()
    
    if not vector_db:
        print("Error: Vector database is empty.")
        return []

    try:
        # Using metadata filtering to target a specific disease among the 4 diseases
        results = vector_db.similarity_search(
            query=disease_id, 
            k=k, 
            filter={"disease_id": disease_id}
        )
        
        retrieved_data = []
        for doc in results:
            chunk_info = {
                "content": doc.page_content,
                "metadata": doc.metadata
            }
            retrieved_data.append(chunk_info)
            
        print(f"Retrieved {len(retrieved_data)} chunks for disease ID: {disease_id}")
        return retrieved_data
        
    except Exception as e:
        print(f"An error occurred during filtered retrieval: {e}")
        return []

if __name__ == "__main__":
    # Test the retriever module with a multi-disease query
    print("Testing general retriever...")
    test_query = "أعراض الأمراض الجلدية وطرق العلاج"
    contexts = get_relevant_context(test_query, k=4)
    
    for idx, item in enumerate(contexts):
        print(f"\n--- Context Item {idx + 1} ---")
        print(f"Disease ID: {item['metadata'].get('disease_id')}")
        print(f"Source File: {item['metadata'].get('source_file')}")
        print(f"Image Path: {item['metadata'].get('image_path')}")
        print(f"Content Preview: {item['content'][:150]}...")