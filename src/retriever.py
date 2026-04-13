import os
from datasets import load_dataset
import torch
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document 

class ArXivHybridRetriever:
    def __init__(self):
        if torch.cuda.is_available():
            self.device = 'cuda'
        elif torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'
            
        print(f"Hardware detected. Running on: {self.device.upper()}")
        print("Loading local embeddings...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="./models/bge-large-en-v1.5", 
            model_kwargs={'device': self.device},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        persist_directory = os.path.join(os.path.dirname(__file__), "..", "data", "chroma_db")
        
        print("Connecting to ChromaDB...")
        self.vectorstore = Chroma(
            collection_name="arxiv_corpus",
            embedding_function=self.embeddings,
            persist_directory=persist_directory
        )
        self.vector_retriever = self.vectorstore.as_retriever(search_kwargs={"k": 15})

        print("Building BM25 Index from Dataset (Including Titles!)...")
        raw_dataset = load_dataset("CShorten/ML-ArXiv-Papers", split="train")
        
        bm25_docs = []
        for item in raw_dataset:
            abstract = str(item.get("abstract", ""))
            title = str(item.get("title", "Unknown Title"))
            
            if abstract.strip():
                combined_text = f"Title: {title}\nAbstract: {abstract}"
                bm25_docs.append(Document(page_content=combined_text, metadata={"title": title}))
        
        self.bm25_retriever = BM25Retriever.from_documents(bm25_docs)
        self.bm25_retriever.k = 15
        print("Configuring Hybrid Ensemble...")
        self.hybrid_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, self.vector_retriever],
            weights=[0.5, 0.5]
        )
        
        print("Loading Cross-Encoder Re-ranker on CUDA...")
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device=self.device)

    def retrieve_and_rerank(self, query, top_k=5):
        print(f"\n[Retriever] Executing Hybrid Search for: '{query}'")
        
        initial_docs = self.hybrid_retriever.invoke(query)
        
        if not initial_docs:
            return []

        pairs = [[query, doc.page_content] for doc in initial_docs]
        scores = self.cross_encoder.predict(pairs)
        
        scored_docs = sorted(zip(initial_docs, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, score in scored_docs[:top_k]]

if __name__ == "__main__":
    retriever = ArXivHybridRetriever()
    test_query = "Attention Is All You Need"
    results = retriever.retrieve_and_rerank(test_query, top_k=3)
    
    for i, doc in enumerate(results, 1):
        print(f"\nRank {i}: {doc.metadata.get('title')} \n{doc.page_content[:200]}...")