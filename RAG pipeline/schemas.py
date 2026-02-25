# app/schemas.py
from pydantic import BaseModel

class QuestionRequest(BaseModel):
    question: str
