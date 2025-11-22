import faiss
import numpy as np

class FAISSVectorStore:
    def __init__(self, dimension=768):
        self.index = faiss.IndexFlatIP(dimension)
        self.documents = []
    
    def add_documents(self, documents, embeddings):
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.documents.extend(documents)
    
    def search(self, query_embedding, top_k=5):
        faiss.normalize_L2(query_embedding.reshape(1, -1))
        scores, indices = self.index.search(query_embedding.reshape(1, -1), top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):
                results.append({
                    'text': self.documents[idx],
                    'score': float(score)
                })
        return results
