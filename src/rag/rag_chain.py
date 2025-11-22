class RAGPipeline:
    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever
    
    def generate(self, query):
        # Retrieve
        docs = self.retriever.retrieve(query)
        context = "\n\n".join([d['text'] for d in docs])
        
        # Generate
        prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        response = self.llm.generate(prompt)
        
        return response
