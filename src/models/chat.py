from typing import List, Optional
from pydantic import BaseModel

class ChatRequest(BaseModel):
    question: str
    thread_id: str

class ChatResponse(BaseModel):
    answer: str
    thread_id: str 