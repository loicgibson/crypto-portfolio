"""
Grok/X sentiment analysis — fetches X/Twitter sentiment for a list of crypto symbols.
Uses xAI Responses API (/v1/responses) with x_search tool.
Results are cached in app_state for 5 minutes.
"""
import json
import re
import time

import requests

from .config import GROK_API_KEY, GROK_MODEL
from .storage import app_get_state, app_set_state

_CACHE_TTL_SECS = 300  # 5 min
_API_URL = "https://api.x.ai/v1/responses"


def fetch_sentiment(symbols: list[str]) -> dict[str, dict]:
    """
    Return {SYMBOL: {"score": "bullish|bearish|neutral", "spike": bool, "summary": str}}.
    Cached for 5 min. Returns {} silently if GROK_API_KEY not configured or on any error.
    """
    if not GROK_API_KEY or not symbols:
        return {}

    cached_raw = app_get_state("grok_sentiment_cache")
    cached_ts  = app_get_state("grok_sentiment_ts")
    now = time.time()

    if cached_raw and cached_ts:
        try:
            if now - float(cached_ts) < _CACHE_TTL_SECS:
                return json.loads(cached_raw)
        except Exception:
            pass

    try:
        result = _call_grok(symbols)
    except Exception:
        return {}

    app_set_state("grok_sentiment_cache", json.dumps(result))
    app_set_state("grok_sentiment_ts",    str(now))
    return result


def _call_grok(symbols: list[str]) -> dict[str, dict]:
    symbols_str = ", ".join(symbols)
    prompt = (
        f"Search X/Twitter right now for real-time sentiment on these cryptocurrencies: {symbols_str}.\n"
        "For each symbol analyse the current discussion and return a JSON object.\n"
        "Fields per symbol:\n"
        '  "score"  : "bullish", "bearish", or "neutral"\n'
        '  "spike"  : true if there is unusual mention or hype volume in the last 2 hours\n'
        '  "summary": one sentence max — what is being said right now\n\n'
        "Return ONLY valid JSON, no markdown, no other text:\n"
        '{"BTC": {"score": "bullish", "spike": false, "summary": "..."}, "ETH": {...}, ...}'
    )

    payload = {
        "model": GROK_MODEL,
        "input": [{"role": "user", "content": prompt}],
        "tools": [{"type": "x_search"}],
    }

    resp = requests.post(
        _API_URL,
        headers={
            "Authorization": f"Bearer {GROK_API_KEY}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()

    # Extract text from the output array
    content = ""
    for item in data.get("output", []):
        if item.get("type") == "message":
            for part in item.get("content", []):
                if part.get("type") == "output_text":
                    content += part.get("text", "")

    match = re.search(r"\{.*\}", content, re.DOTALL)
    if match:
        parsed = json.loads(match.group())
        return {k.upper(): v for k, v in parsed.items() if isinstance(v, dict)}
    return {}
