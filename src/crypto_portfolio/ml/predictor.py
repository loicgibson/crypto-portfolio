"""
Prediction on fresh data — multi-timeframe.

For each symbol:
  1. Load the saved model bundle from disk
  2. Fetch recent 1h / 4h / 15m klines (Binance API → SQLite fallback)
  3. Compute multi-TF features for the most recent 1h bar
  4. Return P(max_gain_4h >= 5%) and metadata
"""
import sqlite3
import time

import numpy as np

from ..binance import get_recent_klines
from ..config import DB_PATH, ML_INTERVAL
from .features import (FEATURE_COLS, add_funding_features, add_market_context,
                        compute_features, klines_to_df)
from .trainer import load_model

# Candle counts to fetch per interval (generous margins for lookback periods)
_LIMITS = {
    "1h":  300,   # needs ~200 for MA200
    "4h":  150,   # needs ~99 for MA99 at 4h
    "15m": 500,   # needs ~99 for MA99 at 15m
}

_btc_cache: dict[str, tuple[float, object]] = {}
_BTC_CACHE_TTL = 300  # seconds


def _get_btc_df(interval: str = "1h"):
    cached = _btc_cache.get(interval)
    if cached and time.time() - cached[0] < _BTC_CACHE_TTL:
        return cached[1]
    rows = _get_fresh_klines("BTC", interval, _LIMITS.get(interval, 300))
    df   = klines_to_df(rows) if rows else None
    _btc_cache[interval] = (time.time(), df)
    return df


def _load_recent_from_db(symbol: str, interval: str, limit: int):
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT open_time, open, high, low, close, volume, close_time, "
            "quote_volume, num_trades FROM klines "
            "WHERE symbol=? AND interval=? ORDER BY open_time DESC LIMIT ?",
            (symbol.upper(), interval, limit),
        ).fetchall()
    return list(reversed(rows))


def _get_fresh_klines(symbol: str, interval: str, limit: int | None = None):
    if limit is None:
        limit = _LIMITS.get(interval, 300)
    try:
        return get_recent_klines(symbol, interval, limit=limit)
    except Exception:
        return _load_recent_from_db(symbol, interval, limit)


def predict_symbol(symbol: str, interval: str = ML_INTERVAL) -> dict:
    """
    Compute ML probability for the latest 1h bar of `symbol`.

    Returns a dict with:
      symbol, ml_prob (float 0-1), model_name, ap, horizon, threshold,
      has_model (bool), error (str | None)
    """
    sym    = symbol.upper()
    bundle = load_model(sym, interval)
    if bundle is None:
        return {"symbol": sym, "has_model": False, "ml_prob": None,
                "model_name": None, "ap": None, "error": "no model"}

    saved_cols = bundle.get("feature_cols", FEATURE_COLS)
    if list(saved_cols) != list(FEATURE_COLS):
        return {"symbol": sym, "has_model": True, "ml_prob": None,
                "model_name": bundle["model_name"],
                "ap": bundle["test_metrics"].get("ap"),
                "error": "feature mismatch — retrain with ml-train"}

    # Fetch all three intervals
    rows_1h  = _get_fresh_klines(sym, "1h")
    rows_4h  = _get_fresh_klines(sym, "4h")
    rows_15m = _get_fresh_klines(sym, "15m")

    if not rows_1h:
        return {"symbol": sym, "has_model": True, "ml_prob": None,
                "model_name": bundle["model_name"],
                "ap": bundle["test_metrics"].get("ap"),
                "error": "no fresh 1h data"}

    btc_df     = _get_btc_df("1h") if sym != "BTC" else None
    from ..storage import get_funding_df
    funding_df = get_funding_df(sym)

    df_1h  = klines_to_df(rows_1h)
    df_4h  = klines_to_df(rows_4h)  if rows_4h  else None
    df_15m = klines_to_df(rows_15m) if rows_15m else None

    df = compute_features(df_1h, df_4h=df_4h, df_15m=df_15m)
    df = add_market_context(df, btc_df, is_btc=(sym == "BTC"))
    df = add_funding_features(df, funding_df)

    if df.empty:
        return {"symbol": sym, "has_model": True, "ml_prob": None,
                "model_name": bundle["model_name"],
                "ap": bundle["test_metrics"].get("ap"),
                "error": "not enough candles for features"}

    last_row = df[FEATURE_COLS].fillna(0).iloc[[-1]].values.astype("float32")

    ap = bundle["test_metrics"].get("ap", 0.0)

    try:
        ml_prob = float(bundle["model"].predict_proba(last_row)[0, 1])
    except Exception as exc:
        return {"symbol": sym, "has_model": True, "ml_prob": None,
                "model_name": bundle["model_name"],
                "ap": ap,
                "error": str(exc)}

    # Suppress probability for low-quality models — brain shouldn't trust them
    MIN_AP = 0.35
    return {
        "symbol":     sym,
        "has_model":  True,
        "ml_prob":    round(ml_prob, 4) if ap >= MIN_AP else None,
        "model_name": bundle["model_name"],
        "ap":         round(ap, 4),
        "horizon":    bundle.get("horizon"),
        "threshold":  bundle.get("threshold"),
        "trained_at": bundle.get("trained_at"),
        "error":      None if ap >= MIN_AP else f"AP={ap:.3f}<{MIN_AP} (model too weak)",
    }


def predict_batch(symbols: list[str],
                  interval: str = ML_INTERVAL) -> dict[str, dict]:
    results = {}
    for sym in symbols:
        results[sym] = predict_symbol(sym, interval)
    return results
