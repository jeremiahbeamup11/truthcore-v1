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
        try:
            from urllib.parse import urlparse
            domain = urlparse(v).netloc.lower().lstrip("www.")
            allowed = [d.lstrip("www.") for d in ALLOWED_DOMAINS]
            if not any(domain.endswith(d) for d in allowed):
                raise ValueError("Only TikTok, Instagram, YouTube, and X URLs are supported.")
        except ValueError:
            raise
        except Exception:
            raise ValueError("Invalid URL format")
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
}

def get_supabase_client():
    try:
        from supabase import create_client
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_KEY")
        if url and key:
            return create_client(url, key)
    except Exception as e:
        print(f"Supabase client error: {e}")
    return None

def get_user_plan(user_id: str) -> str:
    """Returns 'pro' or 'free' for a given user."""
    if not user_id:
        return "free"
    try:
        sb = get_supabase_client()
        if not sb:
            return "free"
        from datetime import datetime, timezone
        result = sb.table("subscriptions").select("plan,status,current_period_end,trial_end").eq("user_id", user_id).limit(1).execute()
        if result.data:
            sub = result.data[0]
            plan = sub.get("plan", "free")
            status = sub.get("status", "inactive")
            period_end = sub.get("current_period_end")
            trial_end = sub.get("trial_end")
            now = datetime.now(timezone.utc).isoformat()
            # Check if trial is active
            if trial_end and trial_end > now:
                return "pro"
            # Check if subscription is active
            if plan == "pro" and status == "active" and period_end and period_end > now:
                return "pro"
        return "free"
    except Exception as e:
        print(f"get_user_plan error: {e}")
        return "free"

def check_and_increment_usage(user_id: str, plan: str) -> bool:
    """
    Returns True if user is allowed to analyze, False if limit reached.
    Increments usage counter.
    """
    if not user_id:
        return True  # unauthenticated users handled by rate limiter

    limit = PRO_MONTHLY_LIMIT if plan == "pro" else FREE_MONTHLY_LIMIT

    try:
        sb = get_supabase_client()
        if not sb:
            return True  # fail open if supabase unavailable

        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)

        result = sb.table("usage").select("*").eq("user_id", user_id).limit(1).execute()

        if not result.data:
            # First analysis — create usage record
            sb.table("usage").insert({
                "user_id": user_id,
                "analyses_this_month": 1,
                "reset_date": (now.replace(day=1) + __import__('dateutil.relativedelta', fromlist=['relativedelta']).relativedelta(months=1)).isoformat() if False else _next_month(now),
            }).execute()
            return True

        usage = result.data[0]
        reset_date = usage.get("reset_date")
        count = usage.get("analyses_this_month", 0)

        # Check if we need to reset the counter
        if reset_date and now.isoformat() > reset_date:
            sb.table("usage").update({
                "analyses_this_month": 1,
                "reset_date": _next_month(now),
            }).eq("user_id", user_id).execute()
            return True

        if count >= limit:
            return False

        # Increment counter
        sb.table("usage").update({
            "analyses_this_month": count + 1,
        }).eq("user_id", user_id).execute()
        return True

    except Exception as e:
        print(f"check_and_increment_usage error: {e}")
        return True  # fail open

def _next_month(dt) -> str:
    from datetime import timezone
    import calendar
    year = dt.year + (1 if dt.month == 12 else 0)
    month = 1 if dt.month == 12 else dt.month + 1
    day = min(dt.day, calendar.monthrange(year, month)[1])
    from datetime import datetime
    return datetime(year, month, day, tzinfo=timezone.utc).isoformat()

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

# ====================== IN-MEMORY CACHE ======================
_analysis_cache: dict = {}

def get_cached_analysis(url: str):
    if url in _analysis_cache:
        print(f"Cache hit for URL: {url}")
        return _analysis_cache[url]
    return None

def save_analysis_to_cache(url: str, transcript, claims: list, overall_confidence: float, platform: str, plan: str = "free"):
    _analysis_cache[url] = AnalyzeResponse(
        url=url,
        transcript=transcript,
        claims=claims,
        overall_confidence=overall_confidence,
        status="success",
        plan=plan,
    )
    print(f"Saved to in-memory cache: {url}")

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

Return ONLY a valid JSON array. No markdown, no explanation, no preamble — just the raw JSON array.

Extract ALL verifiable factual claims present in the content without omitting any. If there are more than {max_claims}, extract the {max_claims} most specific and checkable ones. Be exhaustive — do not selectively pick claims, find every one that exists.

Each item must have exactly this format:
{{"text": "The specific claim made in the content"}}

Focus on: statistics, named entities, historical claims, medical/scientific claims, political claims. Skip opinions and subjective statements.

Reject any claim that is vague, generic, or cannot be verified with a web search. Each claim must contain at least one specific fact: a number, a name, a date, a location, or a direct assertion that can be confirmed or denied."""

    user_prompt = f"""Extract up to {max_claims} verifiable factual claims from this content. Return ONLY a JSON array.

Content:
{transcript}

Return format: [{{"text": "claim here"}}, {{"text": "another claim"}}]"""

    raw = call_perplexity(api_key, system_prompt, user_prompt)
    print("Raw claims extraction response:", repr(raw[:200]))
    claims = extract_json_from_response(raw)
    return claims[:max_claims]

def fact_check_claims(claims_text: list[str], api_key: str, model: str = "sonar") -> list[dict]:
    if not claims_text:
        return []

    claims_formatted = "\n".join([f"{i+1}. {c}" for i, c in enumerate(claims_text)])

    system_prompt = """You are a professional fact-checker with access to current information.

Search the web for current information before evaluating each claim. Do not rely on training data alone.

For each claim, determine if it is TRUE, FALSE, MISLEADING, or UNVERIFIED.
- TRUE: Claim is accurate and verifiable with high confidence
- FALSE: Claim is demonstrably incorrect
- MISLEADING: Claim has some truth but omits key context or is framed deceptively
- UNVERIFIED: Cannot be confirmed or denied with available information

Confidence scoring guide:
- 0.9-1.0: Widely documented, multiple authoritative sources
- 0.7-0.89: Well supported but minor uncertainty
- 0.5-0.69: Some evidence but significant gaps
- 0.3-0.49: Weak or conflicting evidence
- 0.1-0.29: Very little basis, mostly speculation

Return ONLY a valid JSON array. No markdown, no explanation, no preamble.

Each item must have exactly this format:
{
  "text": "the original claim",
  "verdict": "true" | "false" | "misleading" | "unverified",
  "confidence": 0.0-1.0,
  "explanation": "2-3 sentences explaining your verdict with specific reasoning",
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

# ====================== STRIPE CHECKOUT ======================

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
        sb = get_supabase_client()
        stripe_customer_id = None
        if sb:
            result = sb.table("subscriptions").select("stripe_customer_id").eq("user_id", body.user_id).limit(1).execute()
            if result.data and result.data[0].get("stripe_customer_id"):
                stripe_customer_id = result.data[0]["stripe_customer_id"]

        # Create or reuse Stripe customer
        if not stripe_customer_id:
            customer = stripe.Customer.create(
                email=body.email,
                metadata={"user_id": body.user_id}
            )
            stripe_customer_id = customer.id

        # Determine trial days — first 200 users get 14-day trial
        trial_days = None
        if sb:
            count_result = sb.table("subscriptions").select("id", count="exact").execute()
            if count_result.count is not None and count_result.count < 200:
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

# ====================== STRIPE WEBHOOK ======================

@app.post("/webhook/stripe")
async def stripe_webhook(request: Request):
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")
    webhook_secret = os.getenv("STRIPE_WEBHOOK_SECRET")

    try:
        if webhook_secret:
            event = stripe.Webhook.construct_event(payload, sig_header, webhook_secret)
        else:
            event = json.loads(payload)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    sb = get_supabase_client()
    if not sb:
        return {"status": "ok"}

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

        # Get user_id from customer metadata
        customer = stripe.Customer.retrieve(customer_id)
        user_id = customer.get("metadata", {}).get("user_id")

        if user_id:
            sb.table("subscriptions").upsert({
                "user_id": user_id,
                "stripe_customer_id": customer_id,
                "stripe_subscription_id": sub["id"],
                "plan": plan,
                "status": status,
                "current_period_end": period_end_iso,
                "trial_end": trial_end_iso,
            }, on_conflict="user_id").execute()
            print(f"Updated subscription for user {user_id}: {plan}")

    elif event_type == "customer.subscription.deleted":
        sub = event["data"]["object"]
        customer_id = sub["customer"]
        customer = stripe.Customer.retrieve(customer_id)
        user_id = customer.get("metadata", {}).get("user_id")

        if user_id:
            sb.table("subscriptions").update({
                "plan": "free",
                "status": "canceled",
            }).eq("user_id", user_id).execute()
            print(f"Canceled subscription for user {user_id}")

    return {"status": "ok"}

# ====================== TEXT ARTICLE ANALYSIS ======================

@app.post("/analyze", response_model=AnalyzeResponse)
@limiter.limit("10/minute")
async def analyze_article(request: Request, body: AnalyzeRequest):
    perplexity_api_key = os.getenv("PERPLEXITY_API_KEY")
    if not perplexity_api_key:
        raise HTTPException(status_code=500, detail=USER_FRIENDLY_ERRORS["config"])

    plan = get_user_plan(body.user_id) if body.user_id else "free"

    if body.user_id:
        allowed = check_and_increment_usage(body.user_id, plan)
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

    cached = get_cached_analysis(body.url)
    if cached:
        return cached

    try:
        claims = run_analysis(article_text, perplexity_api_key, plan)
        overall_confidence = sum(c.confidence for c in claims) / len(claims) if claims else 0.0
        save_analysis_to_cache(body.url, None, claims, overall_confidence, "article", plan)
        return AnalyzeResponse(url=body.url, claims=claims, overall_confidence=overall_confidence, status="success", plan=plan)
    except Exception:
        return AnalyzeResponse(url=body.url, claims=[], overall_confidence=0.0,
                               status="error: Something went wrong during analysis. Please try again.", plan=plan)

# ====================== VIDEO ANALYSIS ======================

@app.post("/analyze_video", response_model=AnalyzeResponse)
@limiter.limit("5/minute")
async def analyze_video(request: Request, body: AnalyzeRequest):
    print(f"\n=== Starting video analysis ===")
    print(f"Platform: {detect_platform(body.url)}")

    perplexity_api_key = os.getenv("PERPLEXITY_API_KEY")
    if not perplexity_api_key:
        raise HTTPException(status_code=500, detail=USER_FRIENDLY_ERRORS["config"])

    plan = get_user_plan(body.user_id) if body.user_id else "free"

    if body.user_id:
        allowed = check_and_increment_usage(body.user_id, plan)
        if not allowed:
            return AnalyzeResponse(url=body.url, claims=[], overall_confidence=0.0,
                                   status=f"error: {USER_FRIENDLY_ERRORS['limit']}", plan=plan)

    cached = get_cached_analysis(body.url)
    if cached:
        return cached

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
        return AnalyzeResponse(url=body.url, transcript=transcript, claims=claims,
                               overall_confidence=overall_confidence, status="success", plan=plan)
    except Exception as e:
        print(f"Analysis error: {e}")
        return AnalyzeResponse(url=body.url, transcript=transcript, claims=[], overall_confidence=0.0,
                               status=f"error: {USER_FRIENDLY_ERRORS['factcheck']}", plan=plan)

# ====================== X POST ANALYSIS ======================

@app.post("/analyze_x", response_model=AnalyzeResponse)
@limiter.limit("5/minute")
async def analyze_x_post(request: Request, body: AnalyzeRequest):
    print(f"\n=== Starting X post analysis ===")

    perplexity_api_key = os.getenv("PERPLEXITY_API_KEY")
    if not perplexity_api_key:
        raise HTTPException(status_code=500, detail=USER_FRIENDLY_ERRORS["config"])

    plan = get_user_plan(body.user_id) if body.user_id else "free"

    if body.user_id:
        allowed = check_and_increment_usage(body.user_id, plan)
        if not allowed:
            return AnalyzeResponse(url=body.url, claims=[], overall_confidence=0.0,
                                   status=f"error: {USER_FRIENDLY_ERRORS['limit']}", plan=plan)

    cached = get_cached_analysis(body.url)
    if cached:
        return cached

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
        return AnalyzeResponse(url=body.url, transcript=transcript, claims=claims,
                               overall_confidence=overall_confidence, status="success", plan=plan)
    except Exception as e:
        print(f"Analysis error: {e}")
        return AnalyzeResponse(url=body.url, claims=[], overall_confidence=0.0,
                               status=f"error: {USER_FRIENDLY_ERRORS['factcheck']}", plan=plan)
