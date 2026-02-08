from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv
import os
from xai_sdk import Client
from xai_sdk.chat import user, system  # Important: Import these wrappers

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
        article_text = article_text[:4000]  # Limit for API
        print("=== ARTICLE TEXT SNIPPET ===")
        print(article_text[:300])
        print("=== END ARTICLE ===")
    except Exception as e:
        return AnalyzeResponse(
            url=request.url,
            claims=[],
            overall_confidence=0.0,
            status=f"error: Failed to fetch article - {str(e)}"
        )
    
    # Step 2: Use xAI SDK for claims extraction
    try:
        client = Client(
            api_key=xai_api_key,
            timeout=3600  # Longer timeout for reasoning
        )
        
        # Create chat session
        chat = client.chat.create(
            model="grok-4",  # Switch to a more advanced model
            store_messages=False
        )
        
        # Append system and user messages using wrappers
        chat.append(system("You are a precise fact-extraction AI. Always output at least 3 claims if possible, even if low confidence. Use JSON array format strictly. Complete the JSON fully."))
        
        prompt = f"""
Extract at least 3 verifiable claims from this article text. Output ONLY a JSON array of objects, each with:
- "text": The exact claim statement
- "confidence": A float 0.0-1.0 estimating truth probability based on your knowledge
- "explanation": Brief reasoning and sources if possible

Example output: 
[{{"text": "Example claim 1", "confidence": 0.95, "explanation": "Reasoning 1"}}, {{"text": "Example claim 2", "confidence": 0.8, "explanation": "Reasoning 2"}}]

No other text, no introduction, just the JSON array. Make sure to close the array properly.

Article: {article_text}
        """
        chat.append(user(prompt))
        
        # Get response using stream to collect full content
        claims_raw = ""
        for response, chunk in chat.stream():
            if chunk.content:
                claims_raw += chunk.content
        print("=== RAW XAI OUTPUT ===")
        print(repr(claims_raw))  # Full raw
        print("=== END RAW ===")
        
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
        except json.JSONDecodeError as parse_e:
            print(f"JSON Parse Error: {str(parse_e)}")  # Debug in terminal
            claims = []
        
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