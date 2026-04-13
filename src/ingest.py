import os
from datasets import load_dataset
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm
import torch

def build_vector_db():
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print("Loading ArXiv dataset...")
    dataset = load_dataset("CShorten/ML-ArXiv-Papers", split="train")
    
    print("Initializing BGE embeddings on CUDA...")
    embeddings = HuggingFaceEmbeddings(
        model_name="./models/bge-large-en-v1.5", 
        model_kwargs={'device': device}, 
        encode_kwargs={'normalize_embeddings': True}
    )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False,
    )

    persist_directory = os.path.join(os.path.dirname(__file__), "..", "data", "chroma_db")
    os.makedirs(persist_directory, exist_ok=True)
    
    vectorstore = Chroma(
        collection_name="arxiv_corpus",
        embedding_function=embeddings,
        persist_directory=persist_directory
    )

    batch_size = 2000 
    docs_to_insert = []
    
    print(f"Processing, chunking, and embedding records...")
    
    
    for i in tqdm(range(len(dataset))):
        record = dataset[i]
        
        metadata = {
            "title": str(record.get("title", "Unknown Title")),
            "id": str(record.get("Unnamed: 0.1", str(i)))
        }
        
        page_content = str(record.get("abstract", ""))
        
        if page_content.strip():
            chunks = text_splitter.split_text(page_content)
            
            for chunk in chunks:
                doc = Document(page_content=chunk, metadata=metadata)
                docs_to_insert.append(doc)
        
        if len(docs_to_insert) >= batch_size:
            vectorstore.add_documents(docs_to_insert)
            docs_to_insert = []
            
    if docs_to_insert:
        vectorstore.add_documents(docs_to_insert)

    print(f"\nSuccess! Vector database built with chunked documents at: {persist_directory}")

if __name__ == "__main__":
    build_vector_db()