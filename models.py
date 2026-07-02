# models.py
# Request aur Response ka shape define karta hai
# Assignment ke API spec ke according EXACT same schema chahiye

from pydantic import BaseModel
from typing import Optional


class Message(BaseModel):
    """Ek single message — user ya assistant ka"""
    role: str   # "user" ya "assistant"
    content: str


class ChatRequest(BaseModel):
    """POST /chat ko jo bheja jaata hai"""
    messages: list[Message]


class Recommendation(BaseModel):
    """Ek SHL assessment recommendation"""
    name: str        # Assessment ka naam
    url: str         # SHL catalog URL
    test_type: str   # "K", "P", "A", "B", "S", etc.


class ChatResponse(BaseModel):
    """POST /chat se jo wapas aata hai"""
    reply: str                              # Agent ka text response
    recommendations: Optional[list[Recommendation]] = None  # Empty jab clarifying
    end_of_conversation: bool = False       # True jab user confirm kar le