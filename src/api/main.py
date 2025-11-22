from fastapi import FastAPI
from src.api.routes import training, inference, rag

app = FastAPI(title="LLM Fine-Tuning Platform")

app.include_router(training.router, prefix="/training", tags=["Training"])
app.include_router(inference.router, prefix="/inference", tags=["Inference"])
app.include_router(rag.router, prefix="/rag", tags=["RAG"])

@app.get("/")
async def root():
    return {"message": "Welcome to the LLM Fine-Tuning Platform API"}
