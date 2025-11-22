from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

class RAGQueryRequest(BaseModel):
    query: str
    top_k: int = 5

@router.post("/query")
async def rag_query(request: RAGQueryRequest):
    # Mock RAG query
    return {
        "answer": f"RAG answer for: {request.query}",
        "sources": ["doc1", "doc2"]
    }
