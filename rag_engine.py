import faiss
import numpy as np
from typing import List, Dict
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
import tiktoken

# Load environment variables
load_dotenv()

class RAGEngine:
    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE")
        )
        self.index = None
        self.documents = []
        self.encoding = tiktoken.get_encoding("cl100k_base")

    def split_text(self, text: str, max_tokens: int = 500) -> List[str]:
        tokens = self.encoding.encode(text)
        chunks = []
        for i in range(0, len(tokens), max_tokens):
            chunk = self.encoding.decode(tokens[i:i + max_tokens])
            chunks.append(chunk)
        return chunks

    async def add_documents(self, documents: List[str]):
        all_chunks = []
        for doc in documents:
            chunks = self.split_text(doc)
            all_chunks.extend(chunks)
        
        self.documents.extend(all_chunks)
        
        # Process embeddings in batches
        batch_size = 100  # Adjust as needed
        all_embeddings = []
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i+batch_size]
            embeddings = await self._get_embeddings(batch)
            all_embeddings.extend(embeddings)
        
        # Recreate the index with all embeddings
        self.index = faiss.IndexFlatL2(len(all_embeddings[0]))
        self.index.add(np.array(all_embeddings).astype('float32'))

    async def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        response = await self.client.embeddings.create(
            model="text-embedding-3-small",
            input=texts
        )
        return [embedding.embedding for embedding in response.data]

    async def get_embedding(self, text: str) -> List[float]:
        embeddings = await self._get_embeddings([text])
        return embeddings[0]

    async def query(self, question: str, k: int = 3) -> List[Dict[str, str]]:
        if not self.documents:
            return []  # Return empty list if no documents are added yet
        
        question_embedding = await self.get_embedding(question)
        D, I = self.index.search(np.array([question_embedding]).astype('float32'), min(k, len(self.documents)))
        
        results = []
        for i, idx in enumerate(I[0]):
            if 0 <= idx < len(self.documents):
                results.append({
                    "content": self.documents[idx],
                    "score": float(D[0][i])
                })
        return results

# Example usage:
if __name__ == "__main__":
    import asyncio

    async def main():
        # Add some sample documents
        sample_docs = [
            "The quick brown fox jumps over the lazy dog.",
            "Python is a high-level programming language.",
            "Machine learning is a subset of artificial intelligence.",
        ]
        await rag_engine.add_documents(sample_docs)

        # Query the RAG engine
        query = "What is Python?"
        results = await rag_engine.query(query)
        
        print(f"Query: {query}")
        for i, result in enumerate(results, 1):
            print(f"Result {i}:")
            print(f"Content: {result['content']}")
            print(f"Score: {result['score']}")
            print()

    asyncio.run(main())