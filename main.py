# main.py
# FastAPI server — yahi run hoga
# GET /health → server ready hai ya nahi
# POST /chat → agent se baat karo

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv

from models import ChatRequest, ChatResponse, Recommendation
from catalog import get_vectorstore
from agent import run_agent

# .env file se API keys load karo
load_dotenv()

from fastapi.responses import HTMLResponse

@app.get("/", response_class=HTMLResponse)
def index():
    with open("index.html") as f:
        return f.read()

# Global vectorstore — server start hone pe ek baar banta hai
vectorstore = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Server start hone pe catalog load karo"""
    global vectorstore
    
    mistral_key = os.getenv("MISTRAL_API_KEY")
    if not mistral_key:
        raise RuntimeError("MISTRAL_API_KEY .env mein set nahi hai!")
    
    print("SHL Catalog ChromaDB mein load ho raha hai...")
    vectorstore = get_vectorstore(mistral_key)
    print("Server ready!")
    
    yield  # Server yahan run karta hai
    
    # Cleanup (optional)
    print("Server band ho raha hai.")


# FastAPI app
app = FastAPI(
    title="SHL Assessment Recommender",
    description="Conversational agent for recommending SHL assessments",
    lifespan=lifespan,
)


@app.get("/health")
def health_check():
    """
    Readiness check.
    Evaluator pehle yahan call karta hai — 200 aana chahiye.
    """
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Main chat endpoint.
    
    Request: {messages: [{role, content}, ...]}
    Response: {reply, recommendations, end_of_conversation}
    """
    global vectorstore
    
    if vectorstore is None:
        raise HTTPException(status_code=503, detail="Catalog abhi load nahi hua, thodi der mein try karo.")
    
    # Messages ko dict format mein convert karo
    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    
    # Validation: Empty messages nahi hone chahiye
    if not messages:
        raise HTTPException(status_code=400, detail="Messages empty nahi hone chahiye.")
    
    # Turn limit check — assignment mein max 8 turns hai
    if len(messages) > 8:
        messages = messages[-8:]  # Last 8 hi rakho
    
    # Agent run karo
    mistral_key = os.getenv("MISTRAL_API_KEY")
    result = run_agent(messages, vectorstore, mistral_key)
    
    # Recommendations format karo
    recommendations = None
    if result["recommendations"]:
        recommendations = [
            Recommendation(
                name=r["name"],
                url=r["url"],
                test_type=r["test_type"],
            )
            for r in result["recommendations"]
        ]
    
    return ChatResponse(
        reply=result["reply"],
        recommendations=recommendations,
        end_of_conversation=result["end_of_conversation"],
    )


# Local development ke liye
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
