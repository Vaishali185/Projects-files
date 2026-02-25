# app/main.py
from fastapi import FastAPI
from schemas import QuestionRequest
from rag import ask_question

app = FastAPI(title="Zero-Cost AI RAG Assistant")

@app.post("/ask")
def ask(req: QuestionRequest):
    return ask_question(req.question)

@app.get("/health")
def health():
    return {"status": "ok"}
