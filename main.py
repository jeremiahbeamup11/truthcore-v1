from fastapi import FastAPI, HTTPException, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from pydantic import BaseModel, validator
from typing import List, Optional
from dotenv import load_dotenv
import os
import json
import glob
import re
import stripe

load_dotenv()

stripe.api_key = os.getenv("STRIPE_SECRET_KEY")

# ====================== RATE LIMITER ======================

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Veracity v1", version="1.0")
app.state.limiter = limiter

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"detail": "You're analyzing too fast. Please wait a minute and try again."}
    )

# ====================== CORS ======================

ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "https://truthcore-frontend.vercel.app",
    "https://truthcore.ai",
    "https://www.truthcore.ai",
]

VERCEL_PROJECT = os.getenv("VERCEL_PROJECT_NAME", "truthcore-frontend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

# ====================== MODELS ======================

ALLOWED_DOMAINS = [
    "tiktok.com", "vm.tiktok.com",
    "instagram.com", "instagr.am",
    "youtube.com", "youtu.be",
    "www.youtube.com", "www.tiktok.com", "www.instagram.com",
    "x.com", "www.x.com",
]

FREE_MAX_CLAIMS = 5
PRO_MAX_CLAIMS = 15
FREE_MONTHLY_LIMIT = 10
PRO_MONTHLY_LIMIT = 100

class AnalyzeRequest(BaseModel):
    url: str
    user_id: Optional[str] = None

    @validator("url")
    def validate_url(cls, v):
        v = v.strip()
        if not v.startswith("http://") and not v.startswith("https://"):
            raise ValueError("URL must start with http:// or https://")
        return v

class CheckoutRequest(BaseModel):
    user_id: str
    email: str
    plan: str  # "monthly" or "annual"

class Claim(BaseModel):
    text: str
    verdict: str
    confidence: float
    explanation: str
    sources: List[str] = []

class AnalyzeResponse(BaseModel):
    url: str
    transcript: Optional[str] = None
    claims: List[Claim]
    overall_confidence: float
    status: str
    plan: Optional[str] = "free"

# ====================== HELPERS ======================

USER_FRIENDLY_ERRORS = {
    "download": "We couldn't download that content. It may be private, deleted, or geo-restricted.",
    "transcription": "We couldn't transcribe the audio. The video may not have speech, or it's too short.",
    "extraction": "We had trouble reading the claims from this content. Please try again.",
    "factcheck": "Fact-checking hit an issue. Please try again in a moment.",
    "short": "This video doesn't appear to have enough spoken content to fact-check.",
    "no_claims": "No verifiable factual claims were found in this content.",
    "config": "Server configuration error. Please contact support.",
    "limit": "You've reached your monthly analysis limit. Upgrade to Pro for 100 analyses per month.",
    "unsupported": "That platform isn't supported yet. Try TikTok, Instagram, YouTube, or X.",
}

def supabase_request(method: str, table: str, data: dict = None, params: dict = None):
    """Make a direct REST call to Supabase — no supabase package needed."""
    import requests as req
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_KEY")
    if not url or not key:
        print("Supabase env vars missing")
        return None
    try:
        headers = {
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Prefer": "return=representation",
        }
        endpoint = f"{url}/rest/v1/{table}"
        if method == "GET":
            resp = req.get(endpoint, headers=headers, params=params, timeout=5)
        elif method == "POST":
            resp = req.post(endpoint, headers=headers, json=data, timeout=5)
        elif method == "PATCH":
            resp = req.patch(endpoint, headers=headers, json=data, params=params, timeout=5)
        else:
            return None
        if resp.status_code in [200, 201]:
            return resp.json()
        print(f"Supabase error {resp.status_code}: {resp.text}")
        return None
    except Exception as e:
        print(f"Supabase request error: {e}")
        return None


def verify_token(authorization: str | None) -> str | None:
    """Verify Supabase JWT token and return user_id, or None if invalid."""
    if not authorization or not authorization.startswith("Bearer "):
        return None
    token = authorization.replace("Bearer ", "").strip()
    try:
        import requests as req
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_KEY")
        if not url or not key:
            return None
        resp = req.get(
            f"{url}/auth/v1/user",
            headers={
                "apikey": key,
                "Authorization": f"Bearer {token}",
            },
            timeout=5,
        )
        if resp.status_code == 200:
            data = resp.json()
            return data.get("id")
        return None
    except Exception as e:
        print(f"Token verification error: {e}")
        return None

def sanitize_url(url: str) -> str:
    """Sanitize URL to prevent injection attacks."""
    import urllib.parse
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ["http", "https"]:
        raise ValueError("Only http and https URLs are allowed")
    hostname = parsed.hostname or ""
    blocked = ["localhost", "127.0.0.1", "0.0.0.0", "169.254", "192.168", "10.", "172."]
    if any(hostname == b.rstrip(".") or hostname.startswith(b) for b in blocked):
        raise ValueError("Internal URLs are not allowed")
    if parsed.username or parsed.password:
        raise ValueError("URLs with credentials are not allowed")
    return urllib.parse.urlunparse((
        parsed.scheme, parsed.netloc, parsed.path,
        parsed.params, parsed.query, ""
    ))

# ====================== FIX #6: Domain allowlist for social media endpoints ======================

def is_social_media_domain(url: str) -> bool:
    """Check if URL belongs to a supported social media platform."""
    import urllib.parse
    parsed = urllib.parse.urlparse(url)
    hostname = (parsed.hostname or "").lower()
    return any(hostname == d or hostname.endswith(f".{d}") for d in ALLOWED_DOMAINS)

# ====================== FIX #4: Safe ISO date parsing ======================

def _parse_iso(date_str: str):
    """Parse ISO date string safely, handling both +00:00 and Z formats."""
    from datetime import datetime
    if date_str.endswith("Z"):
        date_str = date_str[:-1] + "+00:00"
    return datetime.fromisoformat(date_str)

# ====================== PLAN / USAGE HELPERS ======================

def get_user_plan(user_id: str) -> str:
    """Returns 'pro' or 'free' for a given user."""
    if not user_id:
        return "free"
    try:
        from datetime import datetime, timezone
        result = supabase_request("GET", "subscriptions", params={
            "select": "plan,status,current_period_end,trial_end",
            "user_id": f"eq.{user_id}",
            "limit": "1",
        })
        if result and len(result) > 0:
            sub = result[0]
            plan = sub.get("plan", "free")
            status = sub.get("status", "inactive")
            period_end = sub.get("current_period_end")
            trial_end = sub.get("trial_end")
            now = datetime.now(timezone.utc)
            # FIX #4: proper datetime comparison
            if trial_end and _parse_iso(trial_end) > now:
                return "pro"
            if plan == "pro" and status == "active" and period_end and _parse_iso(period_end) > now:
                return "pro"
        return "free"
    except Exception as e:
        print(f"get_user_plan error: {e}")
        return "free"

# FIX #3: Split into check_usage (before analysis) and increment_usage (after success)

def check_usage(user_id: str, plan: str) -> bool:
    """Returns True if user is within their limit, False if limit reached."""
    if not user_id:
        return True

    limit = PRO_MONTHLY_LIMIT if plan == "pro" else FREE_MONTHLY_LIMIT

    try:
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)

        result = supabase_request("GET", "usage", params={
            "select": "*",
            "user_id": f"eq.{user_id}",
            "limit": "1",
        })

        if not result or len(result) == 0:
            return True  # no record yet, they're fine

        usage = result[0]
        reset_date = usage.get("reset_date")
        count = usage.get("analyses_this_month", 0)

        # FIX #4: proper datetime comparison
        if reset_date and now > _parse_iso(reset_date):
            return True

        if count >= limit:
            print(f"User {user_id} hit limit: {count}/{limit}")
            return False

        return True

    except Exception as e:
        print(f"check_usage error: {e}")
        return True  # fail open


def increment_usage(user_id: str):
    """Increment usage counter AFTER a successful analysis."""
    if not user_id:
        return

    try:
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)

        result = supabase_request("GET", "usage", params={
            "select": "*",
            "user_id": f"eq.{user_id}",
            "limit": "1",
        })

        if not result or len(result) == 0:
            supabase_request("POST", "usage", data={
                "user_id": user_id,
                "analyses_this_month": 1,
                "reset_date": _next_month(now),
            })
            print(f"Created usage record for {user_id}")
            return

        usage = result[0]
        reset_date = usage.get("reset_date")
        count = usage.get("analyses_this_month", 0)

        # FIX #4: proper datetime comparison
        if reset_date and now > _parse_iso(reset_date):
            supabase_request("PATCH", "usage", data={
                "analyses_this_month": 1,
                "reset_date": _next_month(now),
            }, params={"user_id": f"eq.{user_id}"})
        else:
            supabase_request("PATCH", "usage", data={
                "analyses_this_month": count + 1,
            }, params={"user_id": f"eq.{user_id}"})
            print(f"Usage incremented for {user_id}: {count + 1}")

    except Exception as e:
        print(f"increment_usage error: {e}")

def _next_month(dt) -> str:
    from datetime import timezone
    import calendar
    year = dt.year + (1 if dt.month == 12 else 0)
    month = 1 if dt.month == 12 else dt.month + 1
    day = min(dt.day, calendar.monthrange(year, month)[1])
    from datetime import datetime
    return datetime(year, month, day, tzinfo=timezone.utc).isoformat()

# ====================== FIX #7: Efficient subscription count ======================

def get_subscription_count() -> int:
    """Get total subscription count using Supabase HEAD request."""
    import requests as req
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_KEY")
    if not url or not key:
        return 9999  # fail safe, no trial
    try:
        resp = req.head(
            f"{url}/rest/v1/subscriptions",
            headers={
                "apikey": key,
                "Authorization": f"Bearer {key}",
                "Prefer": "count=exact",
            },
            params={"select": "id"},
            timeout=5,
        )
        # Supabase returns count in content-range header: "0-N/total"
        content_range = resp.headers.get("content-range", "")
        if "/" in content_range:
            total = content_range.split("/")[-1]
            if total != "*":
                return int(total)
        return 9999  # can't determine, fail safe
    except Exception:
        return 9999

# ====================== OTHER HELPERS ======================

def get_cookie_file(env_var: str, filename: str) -> str | None:
    import base64
    content = os.getenv(env_var)
    if content:
        path = f'/tmp/{filename}'
        try:
            decoded = base64.b64decode(content).decode('utf-8')
        except Exception:
            decoded = content
        with open(path, 'w') as f:
            f.write(decoded)
        return path

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
    elif "x.com" in url:
        return "x"
    return "unknown"

def cleanup_audio():
    for f in glob.glob('/tmp/truthcore_audio.*'):
        try:
            os.remove(f)
            print(f"Cleaned up: {f}")
        except Exception as e:
            print(f"Warning: could not remove {f}: {e}")

def download_audio(url: str) -> str:
    import yt_dlp

    platform = detect_platform(url)
    output_template = '/tmp/truthcore_audio.%(ext)s'

    cleanup_audio()

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_template,
        'quiet': True,
        'no_warnings': True,
        'extract_flat': False,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '128',
        }],
    }

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
    elif platform == "x":
        from yt_dlp.networking.impersonate import ImpersonateTarget
        cookie_file = get_cookie_file('X_COOKIES', 'x_cookies.txt')
        if cookie_file:
            ydl_opts['cookiefile'] = cookie_file
        ydl_opts['impersonate'] = ImpersonateTarget('chrome')

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except Exception as e:
        raise Exception(f"yt-dlp failed: {str(e)}")

    audio_file = '/tmp/truthcore_audio.mp3'
    if not os.path.exists(audio_file):
        files = glob.glob('/tmp/truthcore_audio.*')
        if not files:
            raise FileNotFoundError("No audio file produced after download")
        audio_file = files[0]

    if os.path.getsize(audio_file) == 0:
        raise Exception("Downloaded audio file is empty")

    return audio_file

def transcribe_audio(audio_file: str) -> str:
    import assemblyai as aai

    aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")
    config = aai.TranscriptionConfig(speech_models=[aai.SpeechModel.universal])
    transcriber = aai.Transcriber(config=config)
    transcript = transcriber.transcribe(audio_file)

    if transcript.status == aai.TranscriptStatus.error:
        raise Exception(f"Transcription error: {transcript.error}")

    return transcript.text.strip()

def get_tweet_id_from_url(url: str):
    match = re.search(r'(?:twitter|x)\.com/\w+/status/(\d+)', url)
    if match:
        return match.group(1)
    return None

def fetch_tweet_via_api(tweet_id: str):
    import requests as req
    bearer_token = os.getenv("X_BEARER_TOKEN")
    if not bearer_token:
        return None
    try:
        response = req.get(
            f"https://api.twitter.com/2/tweets/{tweet_id}",
            headers={"Authorization": f"Bearer {bearer_token}"},
            params={
                "tweet.fields": "text,attachments,entities",
                "expansions": "attachments.media_keys",
                "media.fields": "type,duration_ms",
            },
            timeout=10,
        )
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        print(f"X API request error: {e}")
        return None

def extract_x_content(url: str) -> tuple[str, str | None]:
    import yt_dlp

    caption = ""
    transcript = None
    has_video = False

    tweet_id = get_tweet_id_from_url(url)
    if tweet_id:
        data = fetch_tweet_via_api(tweet_id)
        if data and "data" in data:
            caption = data["data"].get("text", "")
            if "includes" in data and "media" in data["includes"]:
                for media in data["includes"]["media"]:
                    if media.get("type") in ["video", "animated_gif"]:
                        has_video = True
                        break

    if has_video:
        try:
            ydl_opts = {
                "format": "bestaudio/best",
                "outtmpl": "/tmp/truthcore_audio.%(ext)s",
                "quiet": True,
                "no_warnings": True,
                "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "128"}],
            }
            try:
                from yt_dlp.networking.impersonate import ImpersonateTarget
                ydl_opts["impersonate"] = ImpersonateTarget("chrome")
                cookie_file = get_cookie_file("X_COOKIES", "x_cookies.txt")
                if cookie_file:
                    ydl_opts["cookiefile"] = cookie_file
            except Exception:
                pass

            cleanup_audio()
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])

            audio_file = "/tmp/truthcore_audio.mp3"
            if not os.path.exists(audio_file):
                files = glob.glob("/tmp/truthcore_audio.*")
                audio_file = files[0] if files else None

            if audio_file and os.path.getsize(audio_file) > 0:
                transcript = transcribe_audio(audio_file)
                os.remove(audio_file)
        except Exception as e:
            print(f"X video transcription failed (non-fatal): {e}")
            transcript = None

    parts = []
    if caption.strip():
        parts.append(f"Post caption: {caption.strip()}")
    if transcript and transcript.strip():
        parts.append(f"Video speech: {transcript.strip()}")

    combined = "\n\n".join(parts)

    if not combined.strip():
        raise Exception("No content could be extracted from this X post")

    return combined, transcript

# ====================== IN-MEMORY CACHE (FIX #2: keyed by plan) ======================
_analysis_cache: dict = {}

def get_cached_analysis(url: str, plan: str = "free"):
    cache_key = f"{plan}:{url}"
    if cache_key in _analysis_cache:
        print(f"Cache hit for URL: {url} (plan: {plan})")
        return _analysis_cache[cache_key]
    return None

def save_analysis_to_cache(url: str, transcript, claims: list, overall_confidence: float, platform: str, plan: str = "free"):
    cache_key = f"{plan}:{url}"
    _analysis_cache[cache_key] = AnalyzeResponse(
        url=url,
        transcript=transcript,
        claims=claims,
        overall_confidence=overall_confidence,
        status="success",
        plan=plan,
    )
    print(f"Saved to in-memory cache: {url} (plan: {plan})")

# ====================== PERPLEXITY ======================

def call_perplexity(api_key: str, system_prompt: str, user_prompt: str, model: str = "sonar") -> str:
    import requests

    response = requests.post(
        "https://api.perplexity.ai/chat/completions",
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
            "temperature": 0,
        },
        timeout=120,
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()

def extract_json_from_response(raw: str) -> list:
    raw = re.sub(r'```(?:json)?', '', raw).strip()
    match = re.search(r'\[.*\]', raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass
    return []

def extract_claims_from_transcript(transcript: str, api_key: str, max_claims: int = 9) -> list[dict]:
    system_prompt = f"""You are a precise claim-extraction AI. Your ONLY job is to identify specific, verifiable factual claims from content.

Return ONLY a valid JSON array. No markdown, no explanation, no preamble.

Extract the {max_claims} most verifiable factual claims. Prioritize claims that:
- Reference well-known people, organizations, or events that can be searched online
- Contain specific statistics, numbers, or dates that can be checked
- Make assertions about historical events, science, health, or politics
- Would appear in news articles or official records

Avoid:
- Vague statements without specific verifiable facts
- Claims about completely unknown private individuals with no searchable context
- Pure opinions or predictions
- Claims so obscure they cannot be verified by any web search

Each item must have exactly this format:
{{"text": "The specific claim made in the content"}}"""

    user_prompt = f"""Extract up to {max_claims} verifiable factual claims from this content. Return ONLY a JSON array.

Content:
{transcript}

Return format: [{{"text": "claim here"}}, {{"text": "another claim"}}]"""

    raw = call_perplexity(api_key, system_prompt, user_prompt)
    print("Raw claims extraction response:", repr(raw[:200]))
    return extract_json_from_response(raw)[:max_claims]

def fact_check_claims(claims_text: list[str], api_key: str, model: str = "sonar") -> list[dict]:
    if not claims_text:
        return []

    claims_formatted = "\n".join([f"{i+1}. {c}" for i, c in enumerate(claims_text)])

    system_prompt = """You are an elite professional fact-checker with access to real-time web search. Your job is to aggressively verify every claim using live web searches.

CRITICAL: Search the web before evaluating each claim. Do not rely on training data alone. Search for the specific claim, named entities, dates, statistics, and related context.

For each claim, determine if it is TRUE, FALSE, MISLEADING, or UNVERIFIED.
- TRUE: Claim is accurate and supported by credible sources you found
- FALSE: Claim is demonstrably incorrect based on evidence you found
- MISLEADING: Claim has some truth but omits key context, exaggerates, or is framed deceptively
- UNVERIFIED: LAST RESORT ONLY. Use this only when you have aggressively searched and found zero relevant information. If you find even partial evidence, use TRUE, FALSE, or MISLEADING instead.

Before marking anything UNVERIFIED you must:
1. Search for the specific claim using multiple search queries
2. Search for the named people, organizations, dates, or statistics mentioned
3. Search for related news or context around the topic
4. Only mark UNVERIFIED if absolutely no relevant information exists anywhere

Most claims CAN be verified or refuted with proper searching. Minimize UNVERIFIED verdicts.

Confidence scoring:
- 0.9-1.0: Multiple authoritative sources confirm
- 0.7-0.89: Strong evidence with minor uncertainty
- 0.5-0.69: Some evidence but notable gaps
- 0.3-0.49: Weak or conflicting evidence
- 0.1-0.29: Minimal basis found after thorough search

Return ONLY a valid JSON array. No markdown, no explanation, no preamble.

Each item must have exactly this format:
{
  "text": "the original claim",
  "verdict": "true" | "false" | "misleading" | "unverified",
  "confidence": 0.0-1.0,
  "explanation": "2-3 sentences explaining your verdict with specific evidence you found",
  "sources": ["https://actual-url-of-source.com", "https://second-source-url.com"]
}"""

    user_prompt = f"""Fact-check each of these claims. Return ONLY a JSON array.

Claims to fact-check:
{claims_formatted}"""

    raw = call_perplexity(api_key, system_prompt, user_prompt, model=model)
    print("Raw fact-check response:", repr(raw[:300]))
    return extract_json_from_response(raw)


def run_analysis(content: str, api_key: str, plan: str) -> list[Claim]:
    """Run full analysis pipeline with plan-aware limits and model selection."""
    max_claims = PRO_MAX_CLAIMS if plan == "pro" else FREE_MAX_CLAIMS
    model = "sonar-pro" if plan == "pro" else "sonar"

    raw_claims = extract_claims_from_transcript(content, api_key, max_claims=max_claims)
    claims_text = [c.get("text", "") for c in raw_claims if c.get("text")]

    if not claims_text:
        return []

    fact_checked = fact_check_claims(claims_text, api_key, model=model)

    return [Claim(
        text=item.get("text", ""),
        verdict=item.get("verdict", "unverified"),
        confidence=float(item.get("confidence", 0.5)),
        explanation=item.get("explanation", ""),
        sources=item.get("sources", []),
    ) for item in fact_checked]

# ====================== ROUTES ======================

@app.get("/")
def read_root():
    return {"message": "Veracity API is running"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/user/plan")
async def get_plan(user_id: str):
    plan = get_user_plan(user_id)
    return {"plan": plan}

# ====================== STRIPE CHECKOUT (FIX #7: efficient count) ======================

@app.post("/create-checkout-session")
async def create_checkout_session(body: CheckoutRequest):
    try:
        price_id = (
            os.getenv("STRIPE_MONTHLY_PRICE_ID")
            if body.plan == "monthly"
            else os.getenv("STRIPE_YEARLY_PRICE_ID")
        )

        if not price_id:
            raise HTTPException(status_code=500, detail="Stripe price not configured")

        # Check if customer already exists
        stripe_customer_id = None
        result = supabase_request("GET", "subscriptions", params={
            "select": "stripe_customer_id",
            "user_id": f"eq.{body.user_id}",
            "limit": "1",
        })
        if result and len(result) > 0 and result[0].get("stripe_customer_id"):
            stripe_customer_id = result[0]["stripe_customer_id"]

        # Create or reuse Stripe customer
        if not stripe_customer_id:
            customer = stripe.Customer.create(
                email=body.email,
                metadata={"user_id": body.user_id}
            )
            stripe_customer_id = customer.id

        # FIX #7: Use efficient count instead of fetching 1000 rows
        trial_days = None
        if get_subscription_count() < 200:
            trial_days = 14

        session_params = {
            "customer": stripe_customer_id,
            "payment_method_types": ["card"],
            "line_items": [{"price": price_id, "quantity": 1}],
            "mode": "subscription",
            "success_url": "https://truthcore.ai/success?session_id={CHECKOUT_SESSION_ID}",
            "cancel_url": "https://truthcore.ai/pricing",
            "metadata": {"user_id": body.user_id},
        }

        if trial_days:
            session_params["subscription_data"] = {"trial_period_days": trial_days}

        session = stripe.checkout.Session.create(**session_params)
        return {"url": session.url}

    except stripe.StripeError as e:
        raise HTTPException(status_code=400, detail=str(e))

# ====================== STRIPE WEBHOOK (FIX: reject if secret missing) ======================

@app.post("/webhook/stripe")
async def stripe_webhook(request: Request):
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")
    webhook_secret = os.getenv("STRIPE_WEBHOOK_SECRET")

    # FIX: Never accept unverified webhooks in production
    if not webhook_secret:
        print("CRITICAL: STRIPE_WEBHOOK_SECRET is not set — rejecting webhook")
        raise HTTPException(status_code=500, detail="Webhook not configured")

    try:
        event = stripe.Webhook.construct_event(payload, sig_header, webhook_secret)
    except stripe.SignatureVerificationError:
        raise HTTPException(status_code=400, detail="Invalid signature")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    event_type = event["type"]
    print(f"Stripe webhook: {event_type}")

    if event_type in ["customer.subscription.created", "customer.subscription.updated"]:
        sub = event["data"]["object"]
        customer_id = sub["customer"]
        status = sub["status"]
        plan = "pro" if status in ["active", "trialing"] else "free"
        period_end = sub.get("current_period_end")
        trial_end = sub.get("trial_end")

        from datetime import datetime, timezone
        period_end_iso = datetime.fromtimestamp(period_end, tz=timezone.utc).isoformat() if period_end else None
        trial_end_iso = datetime.fromtimestamp(trial_end, tz=timezone.utc).isoformat() if trial_end else None

        customer = stripe.Customer.retrieve(customer_id)
        user_id = customer.get("metadata", {}).get("user_id")

        if user_id:
            # Try update first, then insert
            existing = supabase_request("GET", "subscriptions", params={
                "user_id": f"eq.{user_id}", "limit": "1"
            })
            if existing and len(existing) > 0:
                supabase_request("PATCH", "subscriptions", data={
                    "stripe_customer_id": customer_id,
                    "stripe_subscription_id": sub["id"],
                    "plan": plan,
                    "status": status,
                    "current_period_end": period_end_iso,
                    "trial_end": trial_end_iso,
                }, params={"user_id": f"eq.{user_id}"})
            else:
                supabase_request("POST", "subscriptions", data={
                    "user_id": user_id,
                    "stripe_customer_id": customer_id,
                    "stripe_subscription_id": sub["id"],
                    "plan": plan,
                    "status": status,
                    "current_period_end": period_end_iso,
                    "trial_end": trial_end_iso,
                })
            print(f"Updated subscription for user {user_id}: {plan}")

    elif event_type == "customer.subscription.deleted":
        sub = event["data"]["object"]
        customer_id = sub["customer"]
        customer = stripe.Customer.retrieve(customer_id)
        user_id = customer.get("metadata", {}).get("user_id")

        if user_id:
            supabase_request("PATCH", "subscriptions", data={
                "plan": "free",
                "status": "canceled",
            }, params={"user_id": f"eq.{user_id}"})
            print(f"Canceled subscription for user {user_id}")

    return {"status": "ok"}

# ====================== TEXT ARTICLE ANALYSIS (FIX #3, #5) ======================

@app.post("/analyze", response_model=AnalyzeResponse)
@limiter.limit("10/minute")
async def analyze_article(request: Request, body: AnalyzeRequest):
    perplexity_api_key = os.getenv("PERPLEXITY_API_KEY")
    if not perplexity_api_key:
        raise HTTPException(status_code=500, detail=USER_FRIENDLY_ERRORS["config"])
    try:
        body.url = sanitize_url(body.url)
    except ValueError as e:
        return AnalyzeResponse(url=body.url, claims=[], overall_confidence=0.0, status=f"error: {str(e)}", plan="free")

    authorization = request.headers.get("Authorization")
    user_id = verify_token(authorization)
    plan = get_user_plan(user_id) if user_id else "free"

    # FIX #5: Check cache BEFORE usage
    cached = get_cached_analysis(body.url, plan)
    if cached:
        return cached

    # FIX #3: Only check usage (don't increment yet)
    if user_id:
        allowed = check_usage(user_id, plan)
        if not allowed:
            return AnalyzeResponse(url=body.url, claims=[], overall_confidence=0.0,
                                   status=f"error: {USER_FRIENDLY_ERRORS['limit']}", plan=plan)

    import requests as req
    from bs4 import BeautifulSoup

    try:
        response = req.get(body.url, timeout=10, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        article_text = ' '.join([p.text for p in soup.find_all('p')])[:5000]
    except Exception:
        return AnalyzeResponse(url=body.url, claims=[], overall_confidence=0.0,
                               status="error: We couldn't fetch that article. It may be behind a paywall or unavailable.", plan=plan)

    if not article_text.strip():
        return AnalyzeResponse(url=body.url, claims=[], overall_confidence=0.0,
                               status="error: No readable text found in that article.", plan=plan)

    try:
        claims = run_analysis(article_text, perplexity_api_key, plan)
        overall_confidence = sum(c.confidence for c in claims) / len(claims) if claims else 0.0
        save_analysis_to_cache(body.url, None, claims, overall_confidence, "article", plan)
        # FIX #3: Increment usage only AFTER successful analysis
        if user_id:
            increment_usage(user_id)
        return AnalyzeResponse(url=body.url, claims=claims, overall_confidence=overall_confidence, status="success", plan=plan)
    except Exception:
        return AnalyzeResponse(url=body.url, claims=[], overall_confidence=0.0,
                               status="error: Something went wrong during analysis. Please try again.", plan=plan)

# ====================== VIDEO ANALYSIS (FIX #3, #5, #6) ======================

@app.post("/analyze_video", response_model=AnalyzeResponse)
@limiter.limit("5/minute")
async def analyze_video(request: Request, body: AnalyzeRequest):
    print(f"\n=== Starting video analysis ===")
    print(f"Platform: {detect_platform(body.url)}")

    perplexity_api_key = os.getenv("PERPLEXITY_API_KEY")
    if not perplexity_api_key:
        raise HTTPException(status_code=500, detail=USER_FRIENDLY_ERRORS["config"])
    try:
        body.url = sanitize_url(body.url)
    except ValueError as e:
        return AnalyzeResponse(url=body.url, claims=[], overall_confidence=0.0, status=f"error: {str(e)}", plan="free")

    # FIX #6: Enforce domain allowlist for social media endpoints
    if not is_social_media_domain(body.url):
        return AnalyzeResponse(url=body.url, claims=[], overall_confidence=0.0,
                               status=f"error: {USER_FRIENDLY_ERRORS['unsupported']}", plan="free")

    authorization = request.headers.get("Authorization")
    user_id = verify_token(authorization)
    plan = get_user_plan(user_id) if user_id else "free"

    # FIX #5: Check cache BEFORE usage
    cached = get_cached_analysis(body.url, plan)
    if cached:
        return cached

    # FIX #3: Only check usage (don't increment yet)
    if user_id:
        allowed = check_usage(user_id, plan)
        if not allowed:
            return AnalyzeResponse(url=body.url, claims=[], overall_confidence=0.0,
                                   status=f"error: {USER_FRIENDLY_ERRORS['limit']}", plan=plan)

    print("\n[Step 1] Downloading audio...")
    audio_file = None
    try:
        audio_file = download_audio(body.url)
        print(f"Download complete. File: {audio_file} ({os.path.getsize(audio_file)} bytes)")
    except Exception as e:
        print(f"Download error: {e}")
        return AnalyzeResponse(url=body.url, claims=[], overall_confidence=0.0,
                               status=f"error: {USER_FRIENDLY_ERRORS['download']}", plan=plan)

    print("\n[Step 2] Transcribing audio...")
    try:
        transcript = transcribe_audio(audio_file)
        print(f"Transcription complete. Length: {len(transcript)} chars")
    except Exception as e:
        print(f"Transcription error: {e}")
        return AnalyzeResponse(url=body.url, claims=[], overall_confidence=0.0,
                               status=f"error: {USER_FRIENDLY_ERRORS['transcription']}", plan=plan)
    finally:
        if audio_file and os.path.exists(audio_file):
            os.remove(audio_file)

    if not transcript or len(transcript) < 10:
        return AnalyzeResponse(url=body.url, transcript=transcript, claims=[],
                               overall_confidence=0.0, status=f"error: {USER_FRIENDLY_ERRORS['short']}", plan=plan)

    print("\n[Step 3] Running analysis...")
    try:
        claims = run_analysis(transcript, perplexity_api_key, plan)
        overall_confidence = sum(c.confidence for c in claims) / len(claims) if claims else 0.0
        print(f"\n=== Analysis complete. Claims: {len(claims)}, Confidence: {overall_confidence:.2f}, Plan: {plan} ===")
        save_analysis_to_cache(body.url, transcript, claims, overall_confidence, detect_platform(body.url), plan)
        # FIX #3: Increment usage only AFTER successful analysis
        if user_id:
            increment_usage(user_id)
        return AnalyzeResponse(url=body.url, transcript=transcript, claims=claims,
                               overall_confidence=overall_confidence, status="success", plan=plan)
    except Exception as e:
        print(f"Analysis error: {e}")
        return AnalyzeResponse(url=body.url, transcript=transcript, claims=[], overall_confidence=0.0,
                               status=f"error: {USER_FRIENDLY_ERRORS['factcheck']}", plan=plan)

# ====================== X POST ANALYSIS (FIX #3, #5, #6) ======================

@app.post("/analyze_x", response_model=AnalyzeResponse)
@limiter.limit("5/minute")
async def analyze_x_post(request: Request, body: AnalyzeRequest):
    print(f"\n=== Starting X post analysis ===")

    perplexity_api_key = os.getenv("PERPLEXITY_API_KEY")
    if not perplexity_api_key:
        raise HTTPException(status_code=500, detail=USER_FRIENDLY_ERRORS["config"])
    try:
        body.url = sanitize_url(body.url)
    except ValueError as e:
        return AnalyzeResponse(url=body.url, claims=[], overall_confidence=0.0, status=f"error: {str(e)}", plan="free")

    # FIX #6: Enforce domain allowlist for social media endpoints
    if not is_social_media_domain(body.url):
        return AnalyzeResponse(url=body.url, claims=[], overall_confidence=0.0,
                               status=f"error: {USER_FRIENDLY_ERRORS['unsupported']}", plan="free")

    authorization = request.headers.get("Authorization")
    user_id = verify_token(authorization)
    plan = get_user_plan(user_id) if user_id else "free"

    # FIX #5: Check cache BEFORE usage
    cached = get_cached_analysis(body.url, plan)
    if cached:
        return cached

    # FIX #3: Only check usage (don't increment yet)
    if user_id:
        allowed = check_usage(user_id, plan)
        if not allowed:
            return AnalyzeResponse(url=body.url, claims=[], overall_confidence=0.0,
                                   status=f"error: {USER_FRIENDLY_ERRORS['limit']}", plan=plan)

    print("\n[Step 1] Extracting X post content...")
    try:
        combined_text, transcript = extract_x_content(body.url)
        print(f"Combined content length: {len(combined_text)} chars")
    except Exception as e:
        print(f"X extraction error: {e}")
        return AnalyzeResponse(url=body.url, claims=[], overall_confidence=0.0,
                               status=f"error: {USER_FRIENDLY_ERRORS['download']}", plan=plan)

    if not combined_text.strip():
        return AnalyzeResponse(url=body.url, claims=[], overall_confidence=0.0,
                               status=f"error: {USER_FRIENDLY_ERRORS['no_claims']}", plan=plan)

    print("\n[Step 2] Running analysis...")
    try:
        claims = run_analysis(combined_text, perplexity_api_key, plan)
        overall_confidence = sum(c.confidence for c in claims) / len(claims) if claims else 0.0
        print(f"\n=== X analysis complete. Claims: {len(claims)}, Confidence: {overall_confidence:.2f}, Plan: {plan} ===")
        save_analysis_to_cache(body.url, transcript, claims, overall_confidence, "x", plan)
        # FIX #3: Increment usage only AFTER successful analysis
        if user_id:
            increment_usage(user_id)
        return AnalyzeResponse(url=body.url, transcript=transcript, claims=claims,
                               overall_confidence=overall_confidence, status="success", plan=plan)
    except Exception as e:
        print(f"Analysis error: {e}")
        return AnalyzeResponse(url=body.url, claims=[], overall_confidence=0.0,
                               status=f"error: {USER_FRIENDLY_ERRORS['factcheck']}", plan=plan)