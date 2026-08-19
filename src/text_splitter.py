import os
import glob
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import RAW_DOCS_DIR, IMAGES_DIR, CHUNK_SIZE, CHUNK_OVERLAP

def process_and_split_documents():
    """
    Reads raw text files from the documents directory, splits them into manageable chunks,
    and attaches relevant image paths and metadata to each chunk.
    """
    chunked_data = []
    
    # Initialize the text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len
    )
    
    # Find all text files in the raw-documents directory
    txt_files = glob.glob(os.path.join(RAW_DOCS_DIR, "*.txt"))
    
    if not txt_files:
        print(f"Warning: No text files found in {RAW_DOCS_DIR}")
        return chunked_data

    for file_path in txt_files:
        # Extract disease identifier (e.g., 'disease_1' from 'disease_1.txt')
        file_name = os.path.basename(file_path)
        disease_id = os.path.splitext(file_name)[0]
        
        # Read the content of the text file
        with open(file_path, "r", encoding="utf-8") as file:
            text_content = file.read()
            
        # Split the text into chunks
        chunks = text_splitter.split_text(text_content)
        
        # Define the corresponding image directory path for this disease
        disease_image_dir = os.path.join(IMAGES_DIR, disease_id)
        
        # Find images inside the disease folder if they exist
        available_images = []
        if os.path.exists(disease_image_dir):
            available_images = glob.glob(os.path.join(disease_image_dir, "*.*"))
        
        # Select the primary image path if available
        primary_image = available_images[0] if available_images else None

        # Create structured chunks with metadata
        for i, chunk in enumerate(chunks):
            chunk_record = {
                "chunk_id": f"{disease_id}_chunk_{i+1}",
                "disease_id": disease_id,
                "content": chunk,
                "metadata": {
                    "source_file": file_name,
                    "image_path": primary_image,
                    "chunk_index": i
                }
            }
            chunked_data.append(chunk_record)
            
    print(f"Successfully processed {len(txt_files)} files into {len(chunked_data)} chunks.")
    return chunked_data

if __name__ == "__main__":
    # Test the text splitter independently
    data = process_and_split_documents()
    for sample in data[:2]:  # Print first two chunks as a sample
        print(sample)