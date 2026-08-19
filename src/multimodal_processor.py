import os
import base64
from typing import Optional

def encode_image_to_base64(image_path: str) -> Optional[str]:
    """
    Encodes an image file into a base64 string format required by some LLM APIs 
    (like OpenAI and Anthropic).
    
    Args:
        image_path (str): The local file path of the image.
        
    Returns:
        Optional[str]: Base64 encoded string of the image, or None if failed.
    """
    if not image_path or not os.path.exists(image_path):
        print(f"Error: Image path does not exist: {image_path}")
        return None
        
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
            return encoded_string
    except Exception as e:
        print(f"Error encoding image to base64: {e}")
        return None

def prepare_multimodal_payload(image_path: Optional[str], context_chunks: list, user_query: str) -> dict:
    """
    Prepares the unified multimodal payload combining user query, 
    retrieved textual medical context, and the image reference.
    
    Args:
        image_path (Optional[str]): Path to the patient's uploaded image.
        context_chunks (list): Retrieved text chunks from the retriever.
        user_query (str): The user's specific question or description.
        
    Returns:
        dict: Formatted payload containing context and image details for the generator.
    """
    # Combine all retrieved text chunks into a single reference context
    combined_context = "\n\n".join([chunk["content"] for chunk in context_chunks])
    
    payload = {
        "user_query": user_query,
        "medical_context": combined_context,
        "image_path": image_path,
        "image_base64": encode_image_to_base64(image_path) if image_path else None
    }
    
    print("Multimodal payload prepared successfully.")
    return payload

if __name__ == "__main__":
    # Test the multimodal processor with a dummy path
    test_image = "images/disease_1/eczema_sample_1.jpg" # Adjust based on your folders
    if os.path.exists(test_image):
        b64_test = encode_image_to_base64(test_image)
        print(f"Image encoded successfully. Length: {len(b64_test) if b64_test else 0}")
    else:
        print("Test image path not found, skipping local test.")