from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7

@router.post("/generate")
async def generate(request: GenerateRequest):
    # Mock inference
    return {"text": f"Generated response for: {request.prompt}"}
