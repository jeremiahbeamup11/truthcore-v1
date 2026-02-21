from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv
import os
from xai_sdk import Client
from xai_sdk.chat import user, system  # Important: Import these wrappers

load_dotenv()  # Load variables from .env file

app = FastAPI(title="TruthCore v1", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Keep for local testing
        "https://truthcore-frontend.vercel.app",  # Add for live Vercel frontend
        # Add any other origins if needed, e.g., custom domain later
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
        article_text = ' '.join([p.text for p in soup.find_all('p')])
        article_text = article_text[:4000]  # Limit for API
    except Exception as e:
        return AnalyzeResponse(
            url=request.url,
            claims=[],
            overall_confidence=0.0,
            status=f"error: Failed to fetch article - {str(e)}"
        )
    
    # Step 2: Use xAI SDK for initial claim extraction (focus on core assertions)
    try:
        client = Client(
            api_key=xai_api_key,
            timeout=3600
        )
        
        chat = client.chat.create(
            model="grok-4",
            store_messages=False
        )
        
        chat.append(system("You are a precise fact-extraction AI. Extract verifiable claims, focusing on the core assertion (e.g., the action or statement) rather than peripheral details like titles or dates. If a detail seems outdated or minor, flag it but don't lower confidence for the main claim. Always output at least 3 claims if possible, even if low confidence. Use JSON array format strictly. Complete the JSON fully."))
        
        prompt = f"""
Extract at least 3 verifiable claims from this article text. For each:
- "text": The exact claim statement
- "confidence": A float 0.0-1.0 estimating truth probability based on your knowledge (focus on core, flag minor issues)
- "explanation": Brief reasoning and sources if possible; suggest search terms for verification if confidence low

Output ONLY a JSON array of objects. No other text.

Article: {article_text}
        """
        chat.append(user(prompt))
        
        response = chat.sample()
        claims_raw = response.content[0]
        
        # Parse as JSON
        import json
        try:
            claims_data = json.loads(claims_raw)
            claims = []
            for item in claims_data:
                claims.append(Claim(
                    text=item["text"],
                    confidence=item["confidence"],
                    explanation=item["explanation"]
                ))
        except json.JSONDecodeError:
            claims = []
        
        # Step 3: Use Serpi AI for up-to-date verification (cross-check each claim)
        if claims:
            for claim in claims:
                if claim.confidence < 0.5:  # Only search low-confidence claims to save costs
                    try:
                        serpi_url = "https://serpapi.com/search"  # Serpi API endpoint
                        params = {
                            "q": claim.text,  # Search the claim directly for verification
                            "api_key": serpi_api_key,
                            "num": 5  # Top 5 results for evidence
                        }
                        serpi_resp = requests.get(serpi_url, params=params)
                        serpi_resp.raise_for_status()
                        serpi_results = serpi_resp.json()
                        # Simple evidence aggregation (improve later)
                        evidence = [result['snippet'] for result in serpi_results.get('organic_results', [])[:3]]
                        claim.explanation += f" (Verified evidence: {'; '.join(evidence)})"
                        # Boost confidence if evidence supports
                        if any("confirm" in snippet.lower() or "true" in snippet.lower() for snippet in evidence):
                            claim.confidence = min(1.0, claim.confidence + 0.3)  # Boost by 30%
                    except Exception as search_e:
                        claim.explanation += f" (Search verification failed: {str(search_e)})"
        
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