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
async def analyze_article(request: AnalyzeRequest):
    # Load API keys
    xai_api_key = os.getenv("XAI_API_KEY")
    serpi_api_key = os.getenv("SERPI_API_KEY")
    
    # Step 1: Fetch article content
    import requests
    from bs4 import BeautifulSoup
    try:
        response = requests.get(request.url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        # Extract main text (simple heuristic - improve later)
        article_text = ' '.join([p.text for p in soup.find_all('p')])
        article_text = article_text[:4000]  # Limit for API (adjust as needed)
    except Exception as e:
        return AnalyzeResponse(
            url=request.url,
            claims=[],
            overall_confidence=0.0,
            status=f"error: Failed to fetch article - {str(e)}"
        )
    
    # Step 2: Use xAI API to extract claims
    import httpx
    xai_url = "https://api.x.ai/v1/chat/completions"  # Official xAI API endpoint
    prompt = f"""
    Extract verifiable claims from this article text. For each claim:
    - Text: The exact claim statement
    - Confidence: A float 0.0-1.0 estimating truth probability based on your knowledge
    - Explanation: Brief reasoning and sources if possible
    
    Article: {article_text}
    """
    
    headers = {
        "Authorization": f"Bearer {xai_api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "grok-beta",  # Or your preferred model
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,  # Lower for factual accuracy
        "max_tokens": 1500
    }
    
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(xai_url, headers=headers, json=data)
            resp.raise_for_status()
            result = resp.json()
            claims_raw = result['choices'][0]['message']['content']
            
            # Parse the xAI response into Claim objects (simple split - improve later)
            claims = []
            for line in claims_raw.split('\n\n'):
                if line.strip():
                    parts = line.split('\n')
                    if len(parts) >= 3:
                        claims.append(Claim(
                            text=parts[0].replace("Text: ", ""),
                            confidence=float(parts[1].replace("Confidence: ", "")),
                            explanation=parts[2].replace("Explanation: ", "")
                        ))
            
            overall_confidence = sum(c.confidence for c in claims) / len(claims) if claims else 0.0
            
            return AnalyzeResponse(
                url=request.url,
                claims=claims,
                overall_confidence=overall_confidence,
                status="success"
            )
    except Exception as e:
        return AnalyzeResponse(
            url=request.url,
            claims=[],
            overall_confidence=0.0,
            status=f"error: xAI API failed - {str(e)}"
        )