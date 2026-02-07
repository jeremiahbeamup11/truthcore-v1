from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv
import os

load_dotenv()  # Load variables from .env file
app = FastAPI(title="TruthCore v1", version="1.0")

class AnalyzeRequest(BaseModel):
    url: str

class Claim(BaseModel):
    text: str
    confidence: float  # 0.0 to 1.0
    explanation: str

class AnalyzeResponse(BaseModel):
    url: str
    claims: List[Claim]
    overall_confidence: float
    status: str

@app.get("/")
def read_root():
    return {"message": "TruthCore v1 API is running! 🚀"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/analyze", response_model=AnalyzeResponse)
def analyze_article(request: AnalyzeRequest):
    print("XAI_API_KEY loaded:", bool(os.getenv("XAI_API_KEY")))
    sample_claims = [
        Claim(
            text="The event happened on January 15th, 2025.",
            confidence=0.92,
            explanation="Multiple reliable sources confirm the date."
        ),
        Claim(
            text="The president made a specific statement about the economy.",
            confidence=0.65,
            explanation="Statement confirmed by official transcript, but context is debated."
        ),
        Claim(
            text="Unverified claim about a celebrity scandal.",
            confidence=0.28,
            explanation="Only tabloid sources report this; no primary evidence found."
        )
    ]

    return AnalyzeResponse(
        url=request.url,
        claims=sample_claims,
        overall_confidence=0.68,
        status="success"
    )