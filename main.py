from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv
import os
import json
import glob
import re

load_dotenv()

app = FastAPI(title="TruthCore v1", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*", "http://localhost:3000", "https://truthcore-frontend.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====================== MODELS ======================

class AnalyzeRequest(BaseModel):
    url: str

class Claim(BaseModel):
    text: str
    verdict: str           # "true", "false", "misleading", "unverified"
    confidence: float
    explanation: str
    sources: List[str] = []

class AnalyzeResponse(BaseModel):
    url: str
    transcript: Optional[str] = None
    claims: List[Claim]
    overall_confidence: float
    status: str

# ====================== HELPERS ======================

def get_cookie_file(env_var: str, filename: str) -> str | None:
    """
    Reads base64-encoded cookie content from an environment variable,
    decodes it, writes to a temp file, and returns the path.
    Falls back to local paths in development.
    """
    import base64
    content = os.getenv(env_var)
    if content:
        path = f'/tmp/{filename}'
        try:
            # Try base64 decode first
            decoded = base64.b64decode(content).decode('utf-8')
        except Exception:
            # Fall back to raw content if not base64
            decoded = content
        with open(path, 'w') as f:
            f.write(decoded)
        return path

    # Local dev fallbacks
    local_paths = {
        'tiktok_cookies.txt': '/Users/jeremiahmaysmei/Downloads/cookies.txt',
        'instagram_cookies.txt': '/Users/jeremiahmaysmei/Downloads/cookies (i).txt',
    }
    local = local_paths.get(filename)
    if local and os.path.exists(local):
        return local

    return None

def detect_platform(url: str) -> str:
    if "tiktok.com" in url:
        return "tiktok"
    elif "instagram.com" in url or "instagr.am" in url:
        return "instagram"
    elif "youtube.com" in url or "youtu.be" in url:
        return "youtube"
    return "unknown"

def download_audio(url: str) -> str:
    """
    Downloads audio from TikTok, Instagram, or YouTube.
    Returns the path to the downloaded audio file.
    """
    import yt_dlp

    platform = detect_platform(url)
    output_template = '/tmp/truthcore_audio.%(ext)s'

    # Clean up any previous downloads
    for f in glob.glob('/tmp/truthcore_audio.*'):
        os.remove(f)

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_template,
        'quiet': False,
        'no_warnings': False,
        'extract_flat': False,
        # Convert to mp3 so whisper always gets a consistent format
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '128',
        }],
    }

    # TikTok and Instagram both need cookies + browser impersonation
    if platform == "tiktok":
        from yt_dlp.networking.impersonate import ImpersonateTarget
        cookie_file = get_cookie_file('TIKTOK_COOKIES', 'tiktok_cookies.txt')
        if cookie_file:
            ydl_opts['cookiefile'] = cookie_file
        ydl_opts['impersonate'] = ImpersonateTarget('chrome')
    elif platform == "instagram":
        from yt_dlp.networking.impersonate import ImpersonateTarget
        cookie_file = get_cookie_file('INSTAGRAM_COOKIES', 'instagram_cookies.txt')
        if cookie_file:
            ydl_opts['cookiefile'] = cookie_file
        ydl_opts['impersonate'] = ImpersonateTarget('chrome')

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    # Find the downloaded file (postprocessor converts to mp3)
    audio_file = '/tmp/truthcore_audio.mp3'
    if not os.path.exists(audio_file):
        # Fallback: find whatever was downloaded
        files = glob.glob('/tmp/truthcore_audio.*')
        if not files:
            raise FileNotFoundError("No audio file found after download")
        audio_file = files[0]

    return audio_file

def transcribe_audio(audio_file: str) -> str:
    """Transcribes audio file using AssemblyAI."""
    import assemblyai as aai

    aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")
    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(audio_file)

    if transcript.status == aai.TranscriptStatus.error:
        raise Exception(f"AssemblyAI transcription error: {transcript.error}")

    return transcript.text.strip()

def call_grok(api_key: str, system_prompt: str, user_prompt: str, model: str = "grok-3") -> str:
    """
    Calls xAI Grok and returns the text response.
    Uses the REST API directly to avoid SDK truncation issues.
    """
    import requests

    response = requests.post(
        "https://api.x.ai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": 4096,
            "temperature": 0.1,  # Low temp for consistent JSON
        },
        timeout=120,
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()

def extract_json_from_response(raw: str) -> list:
    """
    Robustly extracts a JSON array from a model response,
    even if there's extra text or markdown around it.
    """
    # Remove markdown code blocks if present
    raw = re.sub(r'```(?:json)?', '', raw).strip()

    # Try to find a JSON array in the response
    match = re.search(r'\[.*\]', raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Try parsing the whole thing
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass

    return []

def extract_claims_from_transcript(transcript: str, api_key: str) -> list[dict]:
    """
    Uses Grok to extract verifiable factual claims from transcript.
    Returns list of dicts with 'text' key.
    """
    system_prompt = """You are a precise claim-extraction AI. Your ONLY job is to identify specific, verifiable factual claims from a video transcript.

Return ONLY a valid JSON array. No markdown, no explanation, no preamble — just the raw JSON array.

Each item in the array must have exactly this format:
{"text": "The specific claim made in the video"}

Focus on: statistics, named entities, historical claims, medical/scientific claims, political claims. Skip opinions."""

    user_prompt = f"""Extract all verifiable factual claims from this transcript. Return ONLY a JSON array.

Transcript:
{transcript}

Return format: [{{"text": "claim here"}}, {{"text": "another claim"}}]"""

    raw = call_grok(api_key, system_prompt, user_prompt)
    print("Raw claims extraction response:", repr(raw[:200]))
    return extract_json_from_response(raw)

def fact_check_claims(claims_text: list[str], api_key: str) -> list[dict]:
    """
    Uses Grok (with live search) to fact-check each extracted claim.
    Returns list of dicts with verdict, confidence, explanation, sources.
    """
    if not claims_text:
        return []

    claims_formatted = "\n".join([f"{i+1}. {c}" for i, c in enumerate(claims_text)])

    system_prompt = """You are a professional fact-checker with access to current information. 

For each claim, determine if it is TRUE, FALSE, MISLEADING, or UNVERIFIED.
- TRUE: Claim is accurate and verifiable
- FALSE: Claim is demonstrably incorrect  
- MISLEADING: Claim has some truth but omits key context or is framed deceptively
- UNVERIFIED: Cannot be confirmed or denied with available information

Return ONLY a valid JSON array. No markdown, no explanation, no preamble.

Each item must have exactly this format:
{
  "text": "the original claim",
  "verdict": "true" | "false" | "misleading" | "unverified",
  "confidence": 0.0-1.0,
  "explanation": "2-3 sentences explaining your verdict",
  "sources": ["source description 1", "source description 2"]
}"""

    user_prompt = f"""Fact-check each of these claims from a social media video. Return ONLY a JSON array.

Claims to fact-check:
{claims_formatted}"""

    raw = call_grok(api_key, system_prompt, user_prompt, model="grok-3")
    print("Raw fact-check response:", repr(raw[:300]))
    return extract_json_from_response(raw)

# ====================== ROUTES ======================

@app.get("/")
def read_root():
    return {"message": "TruthCore v1 API is running! 🚀"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

# ====================== TEXT ARTICLE ANALYSIS ======================

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_article(request: AnalyzeRequest):
    xai_api_key = os.getenv("XAI_API_KEY")

    import requests as req
    from bs4 import BeautifulSoup

    try:
        response = req.get(request.url, timeout=10, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        article_text = ' '.join([p.text for p in soup.find_all('p')])[:5000]
    except Exception as e:
        return AnalyzeResponse(url=request.url, claims=[], overall_confidence=0.0, status=f"error: Failed to fetch article - {str(e)}")

    if not article_text.strip():
        return AnalyzeResponse(url=request.url, claims=[], overall_confidence=0.0, status="error: No readable text found in article")

    try:
        raw_claims = extract_claims_from_transcript(article_text, xai_api_key)
        claims_text = [c.get("text", "") for c in raw_claims if c.get("text")]

        fact_checked = fact_check_claims(claims_text, xai_api_key)

        claims = []
        for item in fact_checked:
            claims.append(Claim(
                text=item.get("text", ""),
                verdict=item.get("verdict", "unverified"),
                confidence=float(item.get("confidence", 0.5)),
                explanation=item.get("explanation", ""),
                sources=item.get("sources", []),
            ))

        overall_confidence = sum(c.confidence for c in claims) / len(claims) if claims else 0.0
        return AnalyzeResponse(url=request.url, claims=claims, overall_confidence=overall_confidence, status="success")

    except Exception as e:
        return AnalyzeResponse(url=request.url, claims=[], overall_confidence=0.0, status=f"error: {str(e)}")

# ====================== VIDEO ANALYSIS ======================

@app.post("/analyze_video", response_model=AnalyzeResponse)
async def analyze_video(request: AnalyzeRequest):
    print(f"\n=== Starting video analysis ===")
    print(f"URL: {request.url}")
    print(f"Platform: {detect_platform(request.url)}")

    xai_api_key = os.getenv("XAI_API_KEY")
    if not xai_api_key:
        return AnalyzeResponse(url=request.url, claims=[], overall_confidence=0.0, status="error: XAI_API_KEY not set")

    # Step 1: Download audio
    print("\n[Step 1] Downloading audio...")
    audio_file = None
    try:
        audio_file = download_audio(request.url)
        print(f"Download complete. File: {audio_file} ({os.path.getsize(audio_file)} bytes)")
    except Exception as e:
        print(f"Download error: {e}")
        return AnalyzeResponse(url=request.url, claims=[], overall_confidence=0.0, status=f"error: Download failed - {str(e)}")

    # Step 2: Transcribe
    print("\n[Step 2] Transcribing audio...")
    try:
        transcript = transcribe_audio(audio_file)
        print(f"Transcription complete. Length: {len(transcript)} chars")
        print(f"Preview: {transcript[:150]}...")
    except Exception as e:
        print(f"Transcription error: {e}")
        return AnalyzeResponse(url=request.url, claims=[], overall_confidence=0.0, status=f"error: Transcription failed - {str(e)}")
    finally:
        if audio_file and os.path.exists(audio_file):
            os.remove(audio_file)

    if not transcript or len(transcript) < 10:
        return AnalyzeResponse(
            url=request.url,
            transcript=transcript,
            claims=[],
            overall_confidence=0.0,
            status="error: Transcript too short — video may have no speech"
        )

    # Step 3: Extract claims
    print("\n[Step 3] Extracting claims from transcript...")
    try:
        raw_claims = extract_claims_from_transcript(transcript, xai_api_key)
        claims_text = [c.get("text", "") for c in raw_claims if c.get("text")]
        print(f"Extracted {len(claims_text)} claims")
        for i, c in enumerate(claims_text):
            print(f"  {i+1}. {c}")
    except Exception as e:
        print(f"Claim extraction error: {e}")
        return AnalyzeResponse(
            url=request.url,
            transcript=transcript,
            claims=[],
            overall_confidence=0.0,
            status=f"error: Claim extraction failed - {str(e)}"
        )

    if not claims_text:
        return AnalyzeResponse(
            url=request.url,
            transcript=transcript,
            claims=[],
            overall_confidence=0.0,
            status="error: No verifiable claims found in transcript"
        )

    # Step 4: Fact-check claims
    print("\n[Step 4] Fact-checking claims...")
    try:
        fact_checked = fact_check_claims(claims_text, xai_api_key)
        print(f"Fact-checked {len(fact_checked)} claims")
    except Exception as e:
        print(f"Fact-check error: {e}")
        return AnalyzeResponse(
            url=request.url,
            transcript=transcript,
            claims=[],
            overall_confidence=0.0,
            status=f"error: Fact-checking failed - {str(e)}"
        )

    # Build final response
    claims = []
    for item in fact_checked:
        claims.append(Claim(
            text=item.get("text", ""),
            verdict=item.get("verdict", "unverified"),
            confidence=float(item.get("confidence", 0.5)),
            explanation=item.get("explanation", ""),
            sources=item.get("sources", []),
        ))

    overall_confidence = sum(c.confidence for c in claims) / len(claims) if claims else 0.0

    print(f"\n=== Analysis complete ===")
    print(f"Claims: {len(claims)}, Overall confidence: {overall_confidence:.2f}")

    return AnalyzeResponse(
        url=request.url,
        transcript=transcript,
        claims=claims,
        overall_confidence=overall_confidence,
        status="success"
    )
