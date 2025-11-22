import numpy as np

class SemanticCache:
    def __init__(self, embedder, threshold=0.95):
        self.embedder = embedder
        self.threshold = threshold
        self.cache = {}
    
    def get(self, query):
        query_emb = self.embedder.encode(query)
        
        for cached_emb_tuple, response in self.cache.items():
            cached_emb = np.array(cached_emb_tuple)
            similarity = np.dot(query_emb, cached_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(cached_emb))
            
            if similarity > self.threshold:
                return response
        return None
    
    def set(self, query, response):
        query_emb = self.embedder.encode(query)
        self.cache[tuple(query_emb)] = response
