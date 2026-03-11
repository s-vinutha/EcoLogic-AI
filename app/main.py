from fastapi import FastAPI
from app.services.classifier import generate_tags
from app.database.vector_store import get_retriever # We'll define this next

app = FastAPI(title="EcoLogic AI")

@app.post("/analyze-product")
async def analyze(name: str):
    # RAG: Get relevant categories first
    retriever = get_retriever()
    relevant_docs = retriever.get_relevant_documents(name)
    context = " ".join([d.page_content for d in relevant_docs])
    
    # AI: Generate the structured tags
    return generate_tags(name, context)