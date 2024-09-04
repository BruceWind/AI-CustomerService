import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict

class RAGEngine:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.documents = []

    def add_documents(self, documents: List[str]):
        """Add documents to the RAG engine."""
        self.documents.extend(documents)
        embeddings = self.model.encode(documents)
        
        if self.index is None:
            self.index = faiss.IndexFlatL2(embeddings.shape[1])
        
        self.index.add(np.array(embeddings).astype('float32'))

    def query(self, question: str, k: int = 3) -> List[Dict[str, str]]:
        """Query the RAG engine to retrieve relevant documents."""
        question_embedding = self.model.encode([question])
        D, I = self.index.search(np.array(question_embedding).astype('float32'), k)
        
        results = []
        for i in I[0]:
            results.append({
                "content": self.documents[i],
                "score": float(D[0][i])  # Convert numpy float to Python float
            })
        
        return results

# Create a global instance of the RAG engine
rag_engine = RAGEngine()

# Example usage:
if __name__ == "__main__":
    # Add some sample documents
    sample_docs = [
        "The quick brown fox jumps over the lazy dog.",
        "Python is a high-level programming language.",
        "Machine learning is a subset of artificial intelligence.",
    ]
    rag_engine.add_documents(sample_docs)

    # Query the RAG engine
    query = "What is Python?"
    results = rag_engine.query(query)
    
    print(f"Query: {query}")
    for i, result in enumerate(results, 1):
        print(f"Result {i}:")
        print(f"Content: {result['content']}")
        print(f"Score: {result['score']}")
        print()