# main.py - sahi order

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from dotenv import load_dotenv

from models import ChatRequest, ChatResponse, Recommendation
from catalog import get_vectorstore
from agent import run_agent

load_dotenv()

vectorstore = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global vectorstore
    mistral_key = os.getenv("MISTRAL_API_KEY")
    if not mistral_key:
        raise RuntimeError("MISTRAL_API_KEY .env mein set nahi hai!")
    print("SHL Catalog ChromaDB mein load ho raha hai...")
    vectorstore = get_vectorstore(mistral_key)
    print("Server ready!")
    yield
    print("Server band ho raha hai.")

# PEHLE app banta hai
app = FastAPI(
    title="SHL Assessment Recommender",
    lifespan=lifespan,
)

# BAAD MEIN routes aate hain
@app.get("/", response_class=HTMLResponse)
def index():
    with open("index.html") as f:
        return f.read()

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    global vectorstore
    if vectorstore is None:
        raise HTTPException(status_code=503, detail="Catalog load nahi hua.")
    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    if not messages:
        raise HTTPException(status_code=400, detail="Messages empty nahi hone chahiye.")
    if len(messages) > 8:
        messages = messages[-8:]
    mistral_key = os.getenv("MISTRAL_API_KEY")
    result = run_agent(messages, vectorstore, mistral_key)
    recommendations = None
    if result["recommendations"]:
        recommendations = [
            Recommendation(name=r["name"], url=r["url"], test_type=r["test_type"])
            for r in result["recommendations"]
        ]
    return ChatResponse(
        reply=result["reply"],
        recommendations=recommendations,
        end_of_conversation=result["end_of_conversation"],
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
