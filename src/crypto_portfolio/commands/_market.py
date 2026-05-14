"""
Shared market data utilities — used by sim_cmd and live_cmd.

Indicator builders, pump pre-filters, exchange info cache, portfolio display,
and shared cycle logic via PortfolioBackend.
"""
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable
from zoneinfo import ZoneInfo

from ..binance import get_all_tickers_24h, get_earn_aprs, get_prices, get_recent_klines
from ..config import (DAILY_STOP_PCT, DAILY_TARGET_PCT, GROK_API_KEY, HARD_TAKE_PROFIT_PCT,
                      ML_INTERVAL, QUOTE_CURRENCY, STOP_LOSS_PCT, STOP_LOSS_PCT_TIER2)
from ..sentiment import fetch_sentiment
from ..display import console
from ..indicators import STABLECOINS, atr, bollinger, macd, rsi_series, sma, stochastic  # noqa: F401
from ..log_util import (log_brain_hold, log_classic_buy, log_classic_sell,
                        log_pump_buy, log_pump_candidates, log_pump_sell)
from ..metrics.compute import compute_context, compute_metrics
from ..ml.predictor import predict_symbol
from ..storage import (app_get_state, app_set_state, get_excluded, get_funding_df,
                       get_trading_symbols, set_inactive_symbols, set_trading_symbols)

_TZ = ZoneInfo("Europe/Paris")
_CLASSIC_CYCLE_HOURS     = 24
_EXCHANGE_INFO_TTL_HOURS = 24
_COOLDOWN_HOURS          = 2.0   # min hours between sell and re-buy of same symbol
_EARLY_STOP_HOURS        = 1.5   # exit if held >= 1.5h AND pnl <= -1.5%
_EARLY_STOP_PCT          = -1.5
_STAGNANT_HOURS          = 2.0   # exit if held >= 2h AND pnl < 0%

RESERVE_CANDIDATES: list[str] = ["BTC", "ETH", "SOL", "BNB"]


def _get_risk_scale(recent_pnls: list) -> tuple[float, int]:
    """Reduce capital when on a losing streak. Returns (multiplier, consecutive_losses)."""
    consecutive = 0
    for pnl in recent_pnls:  # most-recent first
        if pnl < 0:
            consecutive += 1
        else:
            break
    if consecutive >= 5:
        return 0.5, consecutive
    if consecutive >= 3:
        return 0.7, consecutive
    return 1.0, consecutive


def _get_day_balance(backend: "PortfolioBackend", total_value: float) -> float:
    """Return today's opening balance, resetting daily at Paris midnight."""
    today = datetime.now(_TZ).strftime("%Y-%m-%d")
    if backend.get_state("day_start_date") != today:
        backend.set_state("day_start_date",    today)
        backend.set_state("day_start_balance", str(round(total_value, 2)))
        return total_value
    stored = backend.get_state("day_start_balance")
    return float(stored) if stored else total_value


_TIER2_SYMBOLS: list[str] = [
    # Existing
    "HYPER", "KAT", "GIGGLE", "ENSO", "BCH", "TNSR", "NOM", "STO",
    "SOMI", "BANANAS31", "SPK", "ZBT", "BROCCOLI714", "PARTI", "AVNT",
    "BIO", "ORCA", "PNUT", "MUBARAK", "DOLO", "MMT", "SAPIEN", "RESOLV",
    "GUN", "TST", "ZKP", "ORDI", "AIXBT", "BERA", "DASH", "TUT", "XPL",
    "SNX", "NEIRO", "BARD",
    # Ajouts — avg range/j >= 10%
    "ENJ", "APE", "RED", "BLUR", "SOLV", "BOME", "API3", "1000SATS",
    "RARE", "DYM", "DYDX", "TOWNS", "ONT", "NOT", "DOGS", "LUNC",
    "PIXEL", "TREE", "币安人生",
    # Ajouts — avg range/j 8-10%
    "LDO", "CETUS", "WAL", "ACT", "ILV", "SAGA", "HOLO", "PENGU",
    "PENDLE", "ARKM", "TURBO", "SKL", "TRB", "ZEC", "FLUX", "LISTA",
    "COMP",
]


def _now_iso() -> str:
    return datetime.now(_TZ).isoformat(timespec="seconds")


@dataclass
class PortfolioBackend:
    """Abstracts data-access layer — lets sim and live share all cycle logic."""
    label: str                  # display label: "Sim" or "LIVE"
    get_usdc: Callable[[], float]
    get_holdings: Callable[[], list]
    get_transactions: Callable  # (limit: int) -> list[dict]
    get_state: Callable         # (key: str, default: str|None=None) -> str|None
    set_state: Callable         # (key: str, value: str) -> None
    add_cycle: Callable         # (ts, usdc, total, n_actions, summary) -> None
    execute: Callable           # (actions, context, now, dry_run) -> list[dict]


# ── Exchange info cache ───────────────────────────────────────────────────────

def _refresh_inactive_if_stale() -> None:
    from ..binance import get_usdc_pairs_by_status
    updated_at = app_get_state("exchange_info_updated_at")
    if updated_at is not None and get_trading_symbols():
        age = datetime.now(timezone.utc) - datetime.fromisoformat(updated_at)
        if age.total_seconds() < _EXCHANGE_INFO_TTL_HOURS * 3600:
            return
    trading, inactive = get_usdc_pairs_by_status()
    now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
    set_trading_symbols(set(trading), now_iso)
    set_inactive_symbols(inactive, now_iso)
    app_set_state("exchange_info_updated_at", now_iso)
    console.print(f"[dim]Statut de trading rafraîchi ({len(trading)} actifs, {len(inactive)} inactifs).[/]")


# ── Indicator helpers ─────────────────────────────────────────────────────────

def _raw_indicators(klines: list) -> dict:
    closes  = [float(k[4]) for k in klines]
    volumes = [float(k[5]) for k in klines]
    highs   = [float(k[2]) for k in klines]
    lows    = [float(k[3]) for k in klines]

    if len(closes) < 30:
        return {}

    rsi_vals = rsi_series(closes)
    rsi_now  = round(rsi_vals[-1], 1)
    rsi_prev = rsi_vals[-6] if len(rsi_vals) >= 6 else rsi_now

    _, _, histogram = macd(closes)
    macd_h      = round(histogram[-1], 8) if histogram else 0.0
    macd_prev_h = histogram[-2] if len(histogram) >= 2 else macd_h

    bb_upper, bb_mid, bb_lower = bollinger(closes)
    if bb_upper and bb_lower and bb_upper != bb_lower:
        bb_pct = round((closes[-1] - bb_lower) / (bb_upper - bb_lower), 3)
        bb_bw  = round((bb_upper - bb_lower) / bb_mid, 4) if bb_mid else None
    else:
        bb_pct, bb_bw = None, None

    stoch_k, stoch_d = stochastic(highs, lows, closes)
    ma20 = sma(closes, 20)
    ma50 = sma(closes, 50) if len(closes) >= 50 else None

    vol_avg   = sum(volumes[-10:]) / 10 if len(volumes) >= 10 else 0
    vol_ratio = round(volumes[-1] / vol_avg, 2) if vol_avg > 0 else None

    atr_val = atr(highs, lows, closes)
    atr_pct = round(atr_val / closes[-1] * 100, 2) if closes[-1] > 0 else None

    return {
        "rsi":        rsi_now,
        "rsi_trend":  round(rsi_now - rsi_prev, 1),
        "macd_hist":  macd_h,
        "macd_dir":   "up" if macd_h > macd_prev_h else "down",
        "bb_pct":     bb_pct,
        "bb_bw":      bb_bw,
        "stoch_k":    round(stoch_k, 1),
        "stoch_d":    round(stoch_d, 1),
        "vol_ratio":  vol_ratio,
        "above_ma20": bool(ma20 and closes[-1] > ma20),
        "above_ma50": bool(ma50 and closes[-1] > ma50),
        "atr_pct":    atr_pct,
    }


def _condensed_candles(klines: list, n: int = 12) -> list:
    recent  = klines[-n:]
    vol_avg = sum(float(k[5]) for k in klines[-20:]) / min(20, len(klines)) if klines else 1.0
    return [
        [round(float(k[1]), 6), round(float(k[4]), 6),
         round(float(k[5]) / vol_avg, 2) if vol_avg > 0 else 1.0]
        for k in recent
    ]


def _funding_summary(symbol: str) -> dict | None:
    try:
        df = get_funding_df(symbol)
        if df.empty:
            return None
        rates = df.tail(21)["rate"].tolist()
        avg   = sum(rates) / len(rates)
        return {
            "avg_7d_pct": round(avg * 100, 4),
            "last_pct":   round(rates[-1] * 100, 4),
            "trend":      "positive" if avg > 0.0001 else "negative" if avg < -0.0001 else "neutral",
        }
    except Exception:
        return None


def _has_pump_exit_signal(pos: dict) -> bool:
    pnl_pct    = pos.get("pnl_pct") or 0
    held_hours = pos.get("held_hours")
    m          = pos.get("metrics") or {}
    candles    = pos.get("candles_1h") or []

    rsi       = m.get("rsi_14")      or 50
    vol_ratio = m.get("volume_ratio") or 1.0

    change_1h = None
    if len(candles) >= 2 and candles[-2][1] > 0:
        change_1h = (candles[-1][1] / candles[-2][1] - 1) * 100

    if pnl_pct >= HARD_TAKE_PROFIT_PCT:                               return True
    if pnl_pct <= -STOP_LOSS_PCT_TIER2:                               return True
    if rsi > 85 and vol_ratio < 1.0:                                 return True
    if change_1h is not None and change_1h < -2.5 and pnl_pct > 5:  return True
    if held_hours is not None and held_hours > 6 and pnl_pct < 1:   return True
    return False


def _has_pump_entry_signal(c: dict) -> bool:
    m = c.get("metrics") or {}

    # Hard filters
    if (c.get("change_24h") or 0) > 50:
        return False
    rsi_val   = m.get("rsi_14")        or 50
    rsi_trend = m.get("rsi_trend_val") or 0
    stoch_k   = m.get("stoch_k")       or 0
    if rsi_val > 90 and rsi_trend < 5:   return False
    if stoch_k  > 95 and rsi_trend < 10: return False

    # Primary signals
    vol_ratio     = m.get("volume_ratio")   or 0
    vol_spike_15m = c.get("vol_spike_15m")  or 0
    change_1h     = c.get("change_1h")      or 0
    change_3h     = c.get("change_3h")      or 0

    primary = (
        (vol_ratio > 2.0 and change_1h > 0)
        or (change_1h > 2.0 and change_3h > 4.0 and vol_ratio > 1.5)
        or (rsi_trend > 8 and rsi_val < 55)
        or (vol_spike_15m > 3.0 and change_1h > 0)
    )
    if not primary:
        return False

    # Confirmations
    above_ma25  = (m.get("price_distance_ma25_pct") or -1) > 0
    macd_up     = m.get("macd_hist_direction") in ("strengthening", "flipping")
    consec_bull = (
        m.get("consecutive_green", 0) >= 2
        and m.get("volume_trend_5") in ("rising", "stable")
    )
    return above_ma25 or macd_up or consec_bull


def _pump_filter_reason(c: dict) -> str:
    """Return a short human-readable reason why a candidate failed _has_pump_entry_signal."""
    m          = c.get("metrics") or {}
    change_24h = c.get("change_24h") or 0
    rsi        = m.get("rsi_14")        or 50
    rsi_trend  = m.get("rsi_trend_val") or 0
    stoch_k    = m.get("stoch_k")       or 0

    if change_24h > 50:
        return f"change_24h={change_24h:.0f}%>50"
    if rsi > 90 and rsi_trend < 5:
        return f"RSI={rsi:.0f}>90+exhausted"
    if stoch_k > 95 and rsi_trend < 10:
        return f"stoch={stoch_k:.0f}>95+rsi_trend={rsi_trend:.1f}<10"

    vol_ratio     = m.get("volume_ratio")   or 0
    vol_spike_15m = c.get("vol_spike_15m")  or 0
    change_1h     = c.get("change_1h")      or 0
    change_3h     = c.get("change_3h")      or 0

    p1 = vol_ratio > 2.0 and change_1h > 0
    p2 = change_1h > 2.0 and change_3h > 4.0 and vol_ratio > 1.5
    p3 = rsi_trend > 8 and rsi < 55
    p4 = vol_spike_15m > 3.0 and change_1h > 0

    if not (p1 or p2 or p3 or p4):
        return (f"no_primary  vol={vol_ratio:.1f}  v15m={vol_spike_15m:.1f}"
                f"  c1h={change_1h:+.1f}%  c3h={change_3h:+.1f}%  rsi_trend={rsi_trend:.1f}")

    above_ma25  = (m.get("price_distance_ma25_pct") or -1) > 0
    macd_up     = m.get("macd_hist_direction") in ("strengthening", "flipping")
    consec_bull = m.get("consecutive_green", 0) >= 2 and m.get("volume_trend_5") in ("rising", "stable")
    return (f"no_confirm  ma25={'T' if above_ma25 else 'F'}"
            f"  macd={'up' if macd_up else 'dn'}"
            f"  consec_bull={'T' if consec_bull else 'F'}")


def _print_portfolio_snapshot(
    holdings: list,
    prices: dict[str, float],
    usdc: float,
    initial: float,
) -> None:
    from rich import box
    from rich.table import Table

    table = Table(box=box.SIMPLE, show_header=True)
    table.add_column("Actif",       style="bold", min_width=8)
    table.add_column("Prix actuel", justify="right")
    table.add_column("Valeur USDC", justify="right")
    table.add_column("P&L %",       justify="right")

    total_crypto = 0.0
    for h in holdings:
        price   = prices.get(h.symbol, h.avg_buy_price)
        val     = h.quantity * price
        pnl_pct = ((price / h.avg_buy_price) - 1) * 100 if h.avg_buy_price > 0 and price > 0 else 0.0
        total_crypto += val
        c = "green" if pnl_pct >= 0 else "red"
        table.add_row(
            h.symbol,
            f"{price:.4g}",
            f"{val:.2f}",
            f"[{c}]{pnl_pct:+.1f}%[/]",
        )

    total = usdc + total_crypto
    perf  = (total / initial - 1) * 100 if initial > 0 else 0.0
    pc    = "green" if perf >= 0 else "red"
    if holdings:
        table.add_section()
    table.add_row("[bold]USDC[/]",  "", f"[bold]{usdc:.2f}[/]", "")
    table.add_row("[bold]TOTAL[/]", "", f"[bold]{total:.2f}[/]", f"[bold {pc}]{perf:+.1f}%[/]")

    console.print(table)


# ── Shared context helpers ────────────────────────────────────────────────────

def _extract_recent_trades(txs: list[dict], limit: int = 10) -> tuple[list[dict], dict[str, str]]:
    """Return (recent_trades_list, last_buy_timestamp_by_symbol)."""
    recent_trades = [
        {"ts": tx["timestamp"][:16], "action": tx["tx_type"],
         "symbol": tx["symbol"], "price": round(tx["price"], 6)}
        for tx in txs[:limit]
    ]
    last_buy_ts: dict[str, str] = {}
    for tx in txs:
        if tx["tx_type"] == "BUY" and tx["symbol"] not in last_buy_ts:
            last_buy_ts[tx["symbol"]] = tx["timestamp"]
    return recent_trades, last_buy_ts


def _build_positions(
    holdings: list,
    prices: dict,
    last_buy_ts: dict[str, str],
    interval: str,
    ml_interval: str,
) -> list[dict]:
    """Build position dicts with full compute_metrics + ML."""
    now_utc = datetime.now(timezone.utc)
    positions = []
    for h in holdings:
        price   = prices.get(h.symbol, 0.0)
        pnl_pct = ((price / h.avg_buy_price) - 1) * 100 if h.avg_buy_price > 0 and price > 0 else 0.0
        held_h: float | None = None
        if h.symbol in last_buy_ts:
            try:
                bought_dt = datetime.fromisoformat(last_buy_ts[h.symbol])
                held_h = round((now_utc - bought_dt).total_seconds() / 3600, 1)
            except Exception:
                pass
        pos = {
            "symbol":        h.symbol,
            "quantity":      round(h.quantity, 8),
            "avg_buy_price": round(h.avg_buy_price, 6),
            "current_price": round(price, 6),
            "value_usdc":    round(h.quantity * price, 2),
            "pnl_pct":       round(pnl_pct, 2),
            "held_hours":    held_h,
            "metrics":       None,
            "context_1h":    None,
            "candles_1h":    None,
            "change_1h":     None,
            "change_3h":     None,
            "funding":       None,
            "ml_prob_up":    None,
            "ml_ap":         None,
        }
        try:
            klines = get_recent_klines(h.symbol, interval, limit=100)
            closes = [float(k[4]) for k in klines]
            pos["metrics"]    = compute_metrics(klines)
            pos["context_1h"] = compute_context(klines)
            pos["candles_1h"] = _condensed_candles(klines, n=8)
            pos["change_1h"]  = round((closes[-1] / closes[-2]  - 1) * 100, 2) if len(closes) >= 2  and closes[-2]  > 0 else None
            pos["change_3h"]  = round((closes[-1] / closes[-4]  - 1) * 100, 2) if len(closes) >= 4  and closes[-4]  > 0 else None
            pos["funding"]    = _funding_summary(h.symbol)
            ml                = predict_symbol(h.symbol, ml_interval)
            pos["ml_prob_up"] = round(ml.get("ml_prob"), 4) if ml.get("ml_prob") is not None else None
            pos["ml_ap"]      = ml.get("ap")
        except Exception:
            pass
        positions.append(pos)
    return positions


def _build_pump_positions(
    holdings: list,
    prices: dict,
    last_buy_ts: dict[str, str],
) -> list[dict]:
    """Build pump (Tier-2) position dicts — 1h candles only, no ML or funding."""
    now_utc = datetime.now(timezone.utc)
    positions = []
    for h in holdings:
        price   = prices.get(h.symbol, 0.0)
        pnl_pct = ((price / h.avg_buy_price) - 1) * 100 if h.avg_buy_price > 0 and price > 0 else 0.0
        held_h: float | None = None
        if h.symbol in last_buy_ts:
            try:
                bought_dt = datetime.fromisoformat(last_buy_ts[h.symbol])
                held_h = round((now_utc - bought_dt).total_seconds() / 3600, 1)
            except Exception:
                pass
        pos = {
            "symbol":        h.symbol,
            "tier":          2 if h.symbol in set(_TIER2_SYMBOLS) else 1,
            "quantity":      round(h.quantity, 8),
            "avg_buy_price": round(h.avg_buy_price, 6),
            "current_price": round(price, 6),
            "value_usdc":    round(h.quantity * price, 2),
            "pnl_pct":       round(pnl_pct, 2),
            "held_hours":    held_h,
            "indicators":    None,
            "candles_1h":    None,
        }
        try:
            klines            = get_recent_klines(h.symbol, "1h", limit=100)
            pos["metrics"]    = compute_metrics(klines)
            pos["context_1h"] = compute_context(klines)
            pos["candles_1h"] = _condensed_candles(klines, n=8)
            ml                = predict_symbol(h.symbol, ML_INTERVAL)
            pos["ml_prob_up"] = ml.get("ml_prob")
            pos["ml_ap"]      = ml.get("ap")
        except Exception:
            pass
        positions.append(pos)
    return positions


# ── Shared cycle logic ────────────────────────────────────────────────────────

def collect_context(
    backend: PortfolioBackend, interval: str, ml_interval: str | None, pool: int
) -> dict:
    """Build full context dict for the classic (Tier-1) strategy."""
    ml_interval = ml_interval or ML_INTERVAL

    usdc      = backend.get_usdc()
    holdings  = backend.get_holdings()
    excluded  = get_excluded()
    held_syms = {h.symbol for h in holdings}

    tickers   = get_all_tickers_24h()
    earn_aprs = get_earn_aprs()

    _refresh_inactive_if_stale()
    trading = get_trading_symbols()

    market: dict = {}
    for t in tickers:
        sym = t["symbol"].removesuffix(QUOTE_CURRENCY)
        if sym in ("BTC", "ETH") and t["symbol"].endswith(QUOTE_CURRENCY):
            market[sym] = {
                "price":      round(float(t["lastPrice"]), 2),
                "change_24h": round(float(t["priceChangePercent"]), 2),
            }

    recent_txs               = backend.get_transactions(20)
    recent_trades, last_buy_ts = _extract_recent_trades(recent_txs)

    prices    = get_prices([h.symbol for h in holdings]) if holdings else {}
    positions = _build_positions(holdings, prices, last_buy_ts, interval, ml_interval)

    total_crypto = sum(p["value_usdc"] for p in positions)
    total_value  = usdc + total_crypto

    _raw = [
        {
            "symbol":     t["symbol"].removesuffix(QUOTE_CURRENCY),
            "price":      round(float(t["lastPrice"]), 8),
            "change_24h": round(float(t["priceChangePercent"]), 2),
            "vol_usdc":   float(t["quoteVolume"]),
        }
        for t in tickers
        if t["symbol"].endswith(QUOTE_CURRENCY)
        and t["symbol"].removesuffix(QUOTE_CURRENCY) not in STABLECOINS
        and t["symbol"].removesuffix(QUOTE_CURRENCY) not in excluded
        and (not trading or t["symbol"].removesuffix(QUOTE_CURRENCY) in trading)
        and t["symbol"].removesuffix(QUOTE_CURRENCY) not in held_syms
        and float(t["quoteVolume"]) >= 500_000
        and -15 <= float(t["priceChangePercent"]) <= 30
    ]
    vol_max = max((c["vol_usdc"] for c in _raw), default=1.0)
    for c in _raw:
        mom = max(-5.0, min(20.0, c["change_24h"])) / 20.0
        c["_score"] = (c["vol_usdc"] / vol_max) * 0.6 + mom * 0.4
    pool_raw = sorted(_raw, key=lambda x: x["_score"], reverse=True)

    candidates = []
    for c in pool_raw[:pool]:
        sym = c["symbol"]
        cand = {
            "symbol":        sym,
            "price":         c["price"],
            "change_24h":    c["change_24h"],
            "vol_usdc":      round(c["vol_usdc"]),
            "earn_apr":      earn_aprs.get(sym),
            "metrics":       None,
            "context_1h":    None,
            "candles_1h":    None,
            "change_1h":     None,
            "change_3h":     None,
            "vol_spike_15m": None,
            "funding":       None,
            "ml_prob_up":    None,
            "ml_ap":         None,
        }
        try:
            klines = get_recent_klines(sym, interval, limit=100)
            closes = [float(k[4]) for k in klines]
            cand["metrics"]    = compute_metrics(klines)
            cand["context_1h"] = compute_context(klines)
            cand["candles_1h"] = _condensed_candles(klines, n=8)
            cand["change_1h"]  = round((closes[-1] / closes[-2] - 1) * 100, 2) if len(closes) >= 2 and closes[-2] > 0 else None
            cand["change_3h"]  = round((closes[-1] / closes[-4] - 1) * 100, 2) if len(closes) >= 4 and closes[-4] > 0 else None
            try:
                klines_15m = get_recent_klines(sym, "15m", limit=10)
                vols_15m   = [float(k[5]) for k in klines_15m]
                if len(vols_15m) >= 4:
                    avg_15m = sum(vols_15m[-4:-1]) / 3
                    cand["vol_spike_15m"] = round(vols_15m[-1] / avg_15m, 2) if avg_15m > 0 else None
            except Exception:
                pass
            cand["funding"]    = _funding_summary(sym)
            ml                 = predict_symbol(sym, ml_interval)
            cand["ml_prob_up"] = round(ml.get("ml_prob"), 4) if ml.get("ml_prob") is not None else None
            cand["ml_ap"]      = ml.get("ap")
        except Exception:
            pass
        candidates.append(cand)

    _stored_init = backend.get_state("initial_balance")
    initial = float(_stored_init) if _stored_init and float(_stored_init) > 0 else round(total_value, 2)
    day_open   = _get_day_balance(backend, total_value)
    daily_pnl  = round((total_value / day_open - 1) * 100, 2) if day_open > 0 else 0.0

    if GROK_API_KEY:
        all_syms = (
            [c["symbol"] for c in candidates]
            + [p["symbol"] for p in positions]
        )
        sentiment = fetch_sentiment(list(dict.fromkeys(all_syms)))
        for c in candidates:
            c["sentiment_x"] = sentiment.get(c["symbol"])
        for p in positions:
            p["sentiment_x"] = sentiment.get(p["symbol"])

    return {
        "market_context":  market,
        "virtual_usdc":    round(usdc, 2),
        "total_value":     round(total_value, 2),
        "initial_balance": initial,
        "pnl_pct":         round((total_value / initial - 1) * 100, 2) if initial > 0 else 0.0,
        "positions":       positions,
        "candidates":      candidates,
        "recent_trades":   recent_trades,
        "constraints": {
            "stop_loss_pct":        -STOP_LOSS_PCT,
            "hard_take_profit_pct":  HARD_TAKE_PROFIT_PCT,
            "available_usdc":        round(usdc, 2),
            "daily_pnl_pct":         daily_pnl,
            "daily_target_pct":      DAILY_TARGET_PCT,
            "daily_target_reached":  daily_pnl >= DAILY_TARGET_PCT,
            "daily_stop_hit":        daily_pnl <= -DAILY_STOP_PCT,
        },
    }


def collect_pump_context(backend: PortfolioBackend) -> dict:
    """Build context for the Tier-2 pump detection strategy."""
    usdc      = backend.get_usdc()
    holdings  = backend.get_holdings()
    held_syms = {h.symbol for h in holdings}
    excluded  = get_excluded()

    _refresh_inactive_if_stale()
    trading = get_trading_symbols()

    try:
        all_tickers = get_all_tickers_24h()
        tickers_map = {
            t["symbol"].removesuffix(QUOTE_CURRENCY): t
            for t in all_tickers
            if t["symbol"].endswith(QUOTE_CURRENCY)
        }
    except Exception:
        tickers_map = {}

    market: dict = {}
    for sym in ("BTC", "ETH"):
        if sym in tickers_map:
            t = tickers_map[sym]
            market[sym] = {
                "price":      round(float(t["lastPrice"]), 2),
                "change_24h": round(float(t["priceChangePercent"]), 2),
            }

    recent_txs               = backend.get_transactions(20)
    recent_trades, last_buy_ts = _extract_recent_trades(recent_txs)

    all_syms  = list({h.symbol for h in holdings})
    prices    = get_prices(all_syms) if holdings else {}
    positions = _build_pump_positions(holdings, prices, last_buy_ts)

    # Drawdown protection: reduce available capital after consecutive losses
    raw_pnls    = backend.get_state("recent_pnls")
    recent_pnls = json.loads(raw_pnls) if raw_pnls else []
    risk_scale, consecutive_losses = _get_risk_scale(recent_pnls)

    skipped_syms: list[str] = []
    candidates = []
    for sym in _TIER2_SYMBOLS:
        if sym in excluded or sym in held_syms:
            continue
        if trading and sym not in trading:
            skipped_syms.append(f"{sym}(inactive)")
            continue
        try:
            klines = get_recent_klines(sym, "1h", limit=100)
            if len(klines) < 10:
                skipped_syms.append(f"{sym}(no_data)")
                continue
            closes  = [float(k[4]) for k in klines]
            volumes = [float(k[5]) for k in klines]

            def _chg(n: int, _c: list = closes) -> float | None:
                return round((_c[-1] / _c[-n] - 1) * 100, 2) if len(_c) >= n and _c[-n] > 0 else None

            # Intra-hour volume spike: last 15m candle vs avg of prior 3
            vol_spike_15m = None
            try:
                klines_15m = get_recent_klines(sym, "15m", limit=10)
                vols_15m   = [float(k[5]) for k in klines_15m]
                if len(vols_15m) >= 4:
                    recent_avg    = sum(vols_15m[-4:-1]) / 3
                    vol_spike_15m = round(vols_15m[-1] / recent_avg, 2) if recent_avg > 0 else None
            except Exception:
                pass

            m   = compute_metrics(klines)
            ctx = compute_context(klines)

            candidates.append({
                "symbol":        sym,
                "price":         round(closes[-1], 8),
                "change_1h":     _chg(2),
                "change_3h":     _chg(4),
                "change_6h":     _chg(7),
                "change_24h":    round(float(tickers_map[sym]["priceChangePercent"]), 2) if sym in tickers_map else None,
                "vol_spike_15m": vol_spike_15m,
                "context_1h":    ctx,
                "metrics":       m,
                "candles_1h":    _condensed_candles(klines, n=8),
            })
        except Exception as e:
            skipped_syms.append(f"{sym}({type(e).__name__})")
            continue

    if skipped_syms:
        console.print(f"[dim]Ignorés ({len(skipped_syms)}) : {', '.join(skipped_syms)}[/]")

    all_scanned    = list(candidates)
    has_signal     = [c for c in all_scanned if _has_pump_entry_signal(c)]
    filtered_out   = [
        {"symbol": c["symbol"], "reason": _pump_filter_reason(c)}
        for c in all_scanned
        if not _has_pump_entry_signal(c)
    ]
    has_signal_sorted = sorted(
        has_signal,
        key=lambda c: (c.get("metrics") or {}).get("volume_ratio") or 0,
        reverse=True,
    )
    candidates   = has_signal_sorted[:6]
    dropped_signals = [
        {"symbol": c["symbol"], "reason": f"signal_ok_vol={(c.get('metrics') or {}).get('volume_ratio', 0):.2f}_rank>6"}
        for c in has_signal_sorted[6:]
    ]
    filtered_out = filtered_out + dropped_signals

    # Add ML probability + model quality to pre-filtered candidates
    for c in candidates:
        try:
            ml = predict_symbol(c["symbol"], ML_INTERVAL)
            c["ml_prob_up"] = ml.get("ml_prob")   # None if model AP too low
            c["ml_ap"]      = ml.get("ap")         # lets brain weight confidence
        except Exception:
            c["ml_prob_up"] = None
            c["ml_ap"]      = None

    if GROK_API_KEY:
        all_syms = (
            [c["symbol"] for c in candidates]
            + [p["symbol"] for p in positions]
        )
        sentiment = fetch_sentiment(list(dict.fromkeys(all_syms)))
        for c in candidates:
            c["sentiment_x"] = sentiment.get(c["symbol"])
        for p in positions:
            p["sentiment_x"] = sentiment.get(p["symbol"])

    total_crypto  = sum(p["value_usdc"] for p in positions)
    available_cap = round(usdc * risk_scale, 2)
    total_value   = round(usdc + total_crypto, 2)
    _loop_ref    = backend.get_state("loop_ref_balance")
    _stored_init = backend.get_state("initial_balance")
    if _loop_ref and float(_loop_ref) > 0:
        initial = float(_loop_ref)
    elif _stored_init and float(_stored_init) > 0:
        initial = float(_stored_init)
    else:
        initial = total_value

    day_open   = _get_day_balance(backend, total_value)
    daily_pnl  = round((total_value / day_open - 1) * 100, 2) if day_open > 0 else 0.0
    return {
        "_all_scanned":    all_scanned,
        "strategy":        "pump_detection",
        "market_context":  market,
        "virtual_usdc":    round(usdc, 2),
        "total_value":     total_value,
        "initial_balance": initial,
        "pnl_pct":         round((total_value / initial - 1) * 100, 2) if initial > 0 else 0.0,
        "positions":       positions,
        "candidates":      candidates,
        "filtered_out":    filtered_out,
        "recent_trades":   recent_trades,
        "constraints": {
            "stop_loss_pct":        -STOP_LOSS_PCT_TIER2,
            "hard_take_profit_pct":  HARD_TAKE_PROFIT_PCT,
            "available_usdc":        available_cap,
            "risk_scale":            round(risk_scale, 2),
            "consecutive_losses":    consecutive_losses,
            "daily_pnl_pct":         daily_pnl,
            "daily_target_pct":      DAILY_TARGET_PCT,
            "daily_target_reached":  daily_pnl >= DAILY_TARGET_PCT,
            "daily_stop_hit":        daily_pnl <= -DAILY_STOP_PCT,
        },
    }


# ── Watchlist sub-cycle ───────────────────────────────────────────────────────

_WATCHLIST_TTL_MIN     = 30
_WATCHLIST_VOL_SPIKE   = 3.0   # vol_spike_15m threshold to add to watchlist
_WATCHLIST_CONFIRM_VOL = 1.8   # vol_spike_15m to confirm watchlist entry
_WATCHLIST_CONFIRM_PX  = 0.5   # or +0.5% price change is enough to confirm
_BREAKING_VOL_SPIKE    = 8.0   # vol_spike_15m threshold for immediate breaking alert
_BREAKING_CHANGE_1H    = 4.0   # change_1h% required alongside breaking spike


def collect_watchlist_context(backend: PortfolioBackend) -> dict:
    """Lightweight 5-min scan: detect vol spikes, manage watchlist, surface breaking changes."""
    import json
    from datetime import datetime, timezone

    usdc      = backend.get_usdc()
    holdings  = backend.get_holdings()
    held_syms = {h.symbol for h in holdings}
    excluded  = get_excluded()

    _refresh_inactive_if_stale()
    trading = get_trading_symbols()

    now_utc = datetime.now(timezone.utc)
    now_iso = now_utc.isoformat(timespec="seconds")

    # Load + purge expired watchlist entries
    raw_wl    = backend.get_state("watchlist") or "{}"
    watchlist: dict = json.loads(raw_wl)
    watchlist = {
        sym: e for sym, e in watchlist.items()
        if (now_utc - datetime.fromisoformat(e["ts"])).total_seconds() < _WATCHLIST_TTL_MIN * 60
    }

    breaking:  list[dict] = []
    confirmed: list[dict] = []
    new_wl:    list[str]  = []
    skipped:   list[str]  = []

    # Phase 1: quick 15m scan of all TIER2 symbols
    for sym in _TIER2_SYMBOLS:
        if sym in excluded or sym in held_syms:
            continue
        if trading and sym not in trading:
            skipped.append(f"{sym}(inactive)")
            continue
        try:
            klines_15m = get_recent_klines(sym, "15m", limit=12)
            if len(klines_15m) < 5:
                continue
            vols   = [float(k[5]) for k in klines_15m]
            closes = [float(k[4]) for k in klines_15m]

            recent_avg = sum(vols[-4:-1]) / 3 if len(vols) >= 4 else None
            vol_spike  = round(vols[-1] / recent_avg, 2) if recent_avg and recent_avg > 0 else 0.0
            change_1h  = round((closes[-1] / closes[-4] - 1) * 100, 2) if len(closes) >= 4 and closes[-4] > 0 else 0.0
            vol_avg_n  = recent_avg or 1.0
            candles_15m = [
                [round(float(k[1]), 8), round(float(k[4]), 8), round(float(k[5]) / vol_avg_n, 2)]
                for k in klines_15m[-6:]
            ]

            entry: dict = {
                "symbol":        sym,
                "price":         round(closes[-1], 8),
                "vol_spike_15m": vol_spike,
                "change_1h":     change_1h,
                "candles_15m":   candles_15m,
            }

            if sym in watchlist:
                wl       = watchlist[sym]
                wl_ts    = datetime.fromisoformat(wl["ts"])
                held_min = round((now_utc - wl_ts).total_seconds() / 60, 1)
                px_since = round((closes[-1] / wl["price"] - 1) * 100, 2) if wl["price"] > 0 else 0.0
                # Confirm if volume still elevated OR price has moved
                if vol_spike >= _WATCHLIST_CONFIRM_VOL or px_since >= _WATCHLIST_CONFIRM_PX:
                    entry.update({
                        "type":                  "confirmed",
                        "original_spike":        wl["spike"],
                        "held_watchlist_min":    held_min,
                        "change_since_watchlist": px_since,
                        "watchlist_source":      wl.get("source", "mechanical"),
                    })
                    if wl.get("reason"):
                        entry["watchlist_reason"] = wl["reason"]
                    confirmed.append(entry)
                    del watchlist[sym]
                # else: keep watching, do nothing this tick
            elif vol_spike >= _BREAKING_VOL_SPIKE and change_1h >= _BREAKING_CHANGE_1H:
                entry["type"] = "breaking"
                breaking.append(entry)
            elif vol_spike >= _WATCHLIST_VOL_SPIKE:
                watchlist[sym] = {"ts": now_iso, "price": round(closes[-1], 8), "spike": vol_spike}
                new_wl.append(sym)

        except Exception as e:
            skipped.append(f"{sym}({type(e).__name__})")

    # Phase 2: enrich actionable candidates with 1h metrics (only when needed)
    for c in breaking + confirmed:
        try:
            klines_1h       = get_recent_klines(c["symbol"], "1h", limit=100)
            c["metrics"]    = compute_metrics(klines_1h)
            c["context_1h"] = compute_context(klines_1h)
        except Exception:
            pass

    # Persist updated watchlist
    backend.set_state("watchlist", json.dumps(watchlist))

    # Current positions for exit-signal detection
    positions: list[dict] = []
    if holdings:
        prices = get_prices([h.symbol for h in holdings])
        last_buy_ts: dict[str, str] = {}
        for tx in backend.get_transactions(20):
            if tx["tx_type"] == "BUY" and tx["symbol"] not in last_buy_ts:
                last_buy_ts[tx["symbol"]] = tx["timestamp"]
        positions = _build_pump_positions(holdings, prices, last_buy_ts)

    raw_pnls    = backend.get_state("recent_pnls")
    recent_pnls = json.loads(raw_pnls) if raw_pnls else []
    risk_scale, consecutive_losses = _get_risk_scale(recent_pnls)

    if skipped:
        console.print(f"[dim]Watchlist ignorés ({len(skipped)}) : {', '.join(skipped[:8])}[/]")

    total_value = usdc + sum(p.get("value_usdc", 0) for p in positions)
    day_open    = _get_day_balance(backend, total_value)
    daily_pnl   = round((total_value / day_open - 1) * 100, 2) if day_open > 0 else 0.0
    return {
        "breaking":         breaking,
        "confirmed":        confirmed,
        "new_watchlist":    new_wl,
        "watchlist_active": sorted(watchlist.keys()),
        "positions":        positions,
        "virtual_usdc":     round(usdc, 2),
        "constraints": {
            "stop_loss_pct":        -STOP_LOSS_PCT_TIER2,
            "hard_take_profit_pct":  HARD_TAKE_PROFIT_PCT,
            "available_usdc":        round(usdc * risk_scale, 2),
            "risk_scale":            round(risk_scale, 2),
            "consecutive_losses":    consecutive_losses,
            "daily_pnl_pct":         daily_pnl,
            "daily_target_pct":      DAILY_TARGET_PCT,
            "daily_target_reached":  daily_pnl >= DAILY_TARGET_PCT,
            "daily_stop_hit":        daily_pnl <= -DAILY_STOP_PCT,
        },
    }


def _inject_mechanical_exits(actions: list[dict], positions: list[dict]) -> list[dict]:
    """Inject SELL for hard TP, early stop (-1.5% @90 min), and stagnant (<0% @2h)."""
    sell_syms = {a["symbol"] for a in actions if a.get("action") == "SELL"}
    for p in positions:
        sym  = p["symbol"]
        pnl  = p.get("pnl_pct") or 0
        held = p.get("held_hours")
        if sym in sell_syms:
            continue
        if pnl >= HARD_TAKE_PROFIT_PCT:
            actions.append({"action": "SELL", "symbol": sym,
                            "reason": f"hard_tp>={HARD_TAKE_PROFIT_PCT:.0f}%"})
            sell_syms.add(sym)
            console.print(f"[bold green]TP mécanique {sym} : {pnl:+.1f}%[/]")
        elif held is not None and held >= _EARLY_STOP_HOURS and pnl <= _EARLY_STOP_PCT:
            actions.append({"action": "SELL", "symbol": sym,
                            "reason": f"early_stop<={_EARLY_STOP_PCT}%@{held:.1f}h"})
            sell_syms.add(sym)
            console.print(f"[bold red]Stop précoce {sym} : {pnl:+.1f}% à {held:.1f}h[/]")
        elif held is not None and held >= _STAGNANT_HOURS and pnl < 0:
            actions.append({"action": "SELL", "symbol": sym,
                            "reason": f"stagnant<0%@{held:.1f}h"})
            sell_syms.add(sym)
            console.print(f"[bold yellow]Sortie stagnante {sym} : {pnl:+.1f}% à {held:.1f}h[/]")
    return actions


def _filter_cooldown_buys(actions: list[dict], backend: "PortfolioBackend") -> list[dict]:
    """Remove BUY actions for symbols sold within the last _COOLDOWN_HOURS."""
    now_utc = datetime.now(timezone.utc)
    result: list[dict] = []
    blocked: list[str] = []
    for a in actions:
        if a.get("action") != "BUY":
            result.append(a)
            continue
        sym = a["symbol"].upper()
        sold_raw = backend.get_state(f"cooldown_{sym}")
        if sold_raw:
            try:
                sold_dt = datetime.fromisoformat(sold_raw)
                if sold_dt.tzinfo is None:
                    sold_dt = sold_dt.replace(tzinfo=timezone.utc)
                if (now_utc - sold_dt).total_seconds() < _COOLDOWN_HOURS * 3600:
                    blocked.append(sym)
                    continue
            except Exception:
                pass
        result.append(a)
    if blocked:
        console.print(f"[dim]Cooldown 2h — {len(blocked)} achat(s) bloqué(s) : {', '.join(blocked)}[/]")
    return result


def _record_sell_cooldowns(executed: list[dict], backend: "PortfolioBackend") -> None:
    """Store sell timestamp for each executed SELL to enforce re-entry cooldown."""
    now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for e in executed:
        if e.get("action") == "SELL":
            backend.set_state(f"cooldown_{e['symbol'].upper()}", now_iso)


def run_watchlist_cycle(backend: PortfolioBackend, dry_run: bool = False) -> None:
    """Execute the 5-minute watchlist sub-cycle."""
    import json
    from ..ml.brain import get_watchlist_decisions

    now = _now_iso()

    with console.status("[cyan]Sous-cycle watchlist…[/]"):
        ctx = collect_watchlist_context(backend)

    new_wl    = ctx["new_watchlist"]
    active_wl = ctx["watchlist_active"]
    confirmed = ctx["confirmed"]
    breaking  = ctx["breaking"]

    if new_wl:
        console.print(f"[dim]Watchlist +{len(new_wl)} : {', '.join(new_wl)}[/]")
    if active_wl:
        console.print(f"[dim]En surveillance ({len(active_wl)}) : {', '.join(active_wl)}[/]")

    positions_need_exit = any(_has_pump_exit_signal(p) for p in ctx["positions"])
    candidates = breaking + confirmed

    if not candidates and not positions_need_exit:
        console.print("[dim]Sous-cycle — aucun signal.[/]")
        return

    if breaking:
        console.print(f"[bold red]Breaking ({len(breaking)}) :[/] {', '.join(c['symbol'] for c in breaking)}")
    if confirmed:
        console.print(f"[bold yellow]Confirmés ({len(confirmed)}) :[/] {', '.join(c['symbol'] for c in confirmed)}")

    if dry_run:
        console.print("[bold yellow]Mode dry-run — aucun ordre.[/]")

    api_ctx = {
        "strategy":     "watchlist",
        "virtual_usdc": ctx["virtual_usdc"],
        "positions":    ctx["positions"],
        "candidates":   candidates,
        "constraints":  ctx["constraints"],
    }

    brain_model = __import__("crypto_portfolio.config", fromlist=["BRAIN_MODEL"]).BRAIN_MODEL
    with console.status(f"[cyan]Watchlist — {brain_model}…[/]"):
        decisions = get_watchlist_decisions(api_ctx)

    summary = decisions.get("market_summary", "")
    actions = decisions.get("actions", [])
    actions, n_contra = _drop_contradictory_actions(actions)

    if n_contra:
        console.print(f"[yellow]{n_contra} action(s) contradictoire(s) ignorée(s).[/]")

    watch_actions = [a for a in actions if a.get("action") == "WATCH"]
    actions       = [a for a in actions if a.get("action") != "WATCH"]
    if watch_actions:
        _apply_watch_actions(backend, watch_actions)
        for w in watch_actions:
            console.print(f"[dim]WATCH {w['symbol']} (brain) : {(w.get('reason') or '')[:100]}[/]")

    if summary:
        console.print(f"[bold]Watchlist :[/] {summary}\n")

    wl_cand_map = {c["symbol"]: c for c in candidates}

    holds = [a for a in actions if a.get("action") == "HOLD"]
    for h in holds:
        console.print(f"[dim]HOLD {h['symbol']} : {(h.get('reason') or '')[:100]}[/]")
        log_brain_hold(h["symbol"], h.get("reason"), wl_cand_map.get(h["symbol"]), now, "watchlist")

    # Mechanical exits: hard TP, early stop (-1.5% @90min), stagnant (<0% @2h)
    actions = _inject_mechanical_exits(actions, ctx["positions"])

    # Enforce daily stop — block new BUYs
    if (ctx.get("constraints", {}) or {}).get("daily_stop_hit"):
        n_blocked = sum(1 for a in actions if a.get("action") == "BUY")
        actions = [a for a in actions if a.get("action") != "BUY"]
        if n_blocked:
            console.print(f"[bold red]Daily stop ({DAILY_STOP_PCT:.0f}%) — {n_blocked} achat(s) bloqué(s)[/]")

    # Block re-entry within 2h of sell
    actions = _filter_cooldown_buys(actions, backend)

    executed = backend.execute(actions, api_ctx, now, dry_run)
    _record_sell_cooldowns(executed, backend)

    if executed:
        _display_action_table(executed, "[bold red]Watchlist — Actions[/]")

    _s = backend.get_state("initial_balance")
    if not _s or float(_s) <= 0:
        final_hold   = backend.get_holdings()
        final_prices = get_prices([h.symbol for h in final_hold]) if final_hold else {}
        final_crypto = sum(h.quantity * final_prices.get(h.symbol, h.avg_buy_price) for h in final_hold)
        backend.set_state("initial_balance", str(round(backend.get_usdc() + final_crypto, 2)))

    if not dry_run and executed:
        final_hold   = backend.get_holdings()
        final_prices = get_prices([h.symbol for h in final_hold]) if final_hold else {}
        final_crypto = sum(h.quantity * final_prices.get(h.symbol, h.avg_buy_price) for h in final_hold)
        backend.add_cycle(now, backend.get_usdc(), backend.get_usdc() + final_crypto, len(executed), summary)

    executed_syms = {e["symbol"] for e in executed}
    n_skip = sum(1 for a in actions if a["action"] != "HOLD" and a["symbol"].upper() not in executed_syms)
    if n_skip:
        console.print(f"[dim]{n_skip} action(s) ignorée(s).[/]")


def is_classic_due(backend: PortfolioBackend) -> bool:
    last = backend.get_state("last_classic_cycle_at")
    if not last:
        return True
    age = datetime.now(timezone.utc) - datetime.fromisoformat(last)
    return age.total_seconds() >= _CLASSIC_CYCLE_HOURS * 3600


# ── Shared display ────────────────────────────────────────────────────────────

def _display_action_table(executed: list[dict], title: str) -> None:
    from rich import box
    from rich.table import Table

    table = Table(title=title, box=box.ROUNDED, title_justify="left", expand=True)
    table.add_column("Action", justify="center", no_wrap=True)
    table.add_column("Actif",  style="bold",     no_wrap=True)
    table.add_column("Qté",    justify="right",  no_wrap=True)
    table.add_column("Prix",   justify="right",  no_wrap=True)
    table.add_column("USDC",   justify="right",  no_wrap=True)
    table.add_column("Raison", ratio=1)

    for a in executed:
        color = "green" if a["action"] == "BUY" else "red"
        dry   = " [dim](dry)[/]" if a.get("dry_run") else ""
        if a["action"] == "BUY":
            usdc_col = f"{a['usdc_spent']:.2f}{dry}"
        else:
            pnl_str  = f" ({a['pnl_pct']:+.1f}%)" if "pnl_pct" in a else ""
            usdc_col = f"+{a['proceeds']:.2f}{pnl_str}{dry}"
        table.add_row(
            f"[{color}]{a['action']}[/]",
            a["symbol"], f"{a['qty']:.6g}", f"{a['price']:.4g}",
            usdc_col, a["reason"],
        )
    console.print(table)


# ── Shared cycle runners ──────────────────────────────────────────────────────

def _drop_contradictory_actions(actions: list[dict]) -> tuple[list[dict], int]:
    """Drop SELL actions for symbols that are also BUYed in the same batch.
    Returns (filtered_actions, n_dropped).
    Prevents buy-then-immediately-sell in a single cycle."""
    bought = {a["symbol"].upper() for a in actions if a["action"] == "BUY"}
    filtered, n_dropped = [], 0
    for a in actions:
        if a["action"] == "SELL" and a["symbol"].upper() in bought:
            n_dropped += 1
        else:
            filtered.append(a)
    return filtered, n_dropped


def _apply_watch_actions(backend: PortfolioBackend, watch_actions: list[dict]) -> None:
    """Add brain-requested WATCH symbols to the watchlist state."""
    if not watch_actions:
        return
    raw_wl    = backend.get_state("watchlist") or "{}"
    watchlist = json.loads(raw_wl)
    now_utc   = datetime.now(timezone.utc)
    now_iso   = now_utc.isoformat(timespec="seconds")

    syms_needed = [a["symbol"].upper() for a in watch_actions if a["symbol"].upper() not in watchlist]
    prices: dict = {}
    if syms_needed:
        try:
            prices = get_prices(syms_needed)
        except Exception:
            pass

    for wa in watch_actions:
        sym = wa["symbol"].upper()
        if sym in watchlist:
            continue
        watchlist[sym] = {
            "ts":     now_iso,
            "price":  round(prices.get(sym, 0.0), 8),
            "spike":  0.0,
            "source": "brain",
            "reason": (wa.get("reason") or "")[:100],
        }

    backend.set_state("watchlist", json.dumps(watchlist))


def run_cycle(
    backend: PortfolioBackend,
    interval: str,
    ml_interval: str | None,
    pool: int,
    verbose: bool = False,
    dry_run: bool = False,
) -> None:
    """Execute one full Tier-1 rebalancing cycle."""
    import json
    from ..ml.brain import get_decisions

    now = _now_iso()

    with console.status("[cyan]Collecte des données marché…[/]"):
        context = collect_context(backend, interval, ml_interval, pool)

    if verbose:
        console.print("[bold dim]── Contexte envoyé à l'API ──[/]")
        console.print(json.dumps(context, indent=2, ensure_ascii=False))
        console.print("[bold dim]─────────────────────────────[/]\n")

    usdc  = context["virtual_usdc"]
    total = context["total_value"]
    pnl   = context["pnl_pct"]
    pc    = "green" if pnl >= 0 else "red"
    console.print(
        f"[bold]Portefeuille :[/] {usdc:.2f} USDC + {total - usdc:.2f} crypto "
        f"= [bold]{total:.2f} USDC[/] ([{pc}]{pnl:+.1f}%[/] vs initial)"
    )
    cand_names = [c["symbol"] for c in context["candidates"]]
    cand_str   = f" : {', '.join(cand_names)}" if cand_names else ""
    console.print(f"[dim]{len(context['positions'])} position(s) | "
                  f"{len(cand_names)} candidat(s) → API{cand_str}[/]\n")

    brain_model = __import__("crypto_portfolio.config", fromlist=["BRAIN_MODEL"]).BRAIN_MODEL
    with console.status(f"[cyan]Appel {brain_model}…[/]"):
        decisions = get_decisions(context)

    summary = decisions.get("market_summary", "")
    actions = decisions.get("actions", [])

    actions, n_contra = _drop_contradictory_actions(actions)
    if n_contra:
        console.print(f"[yellow]{n_contra} action(s) contradictoire(s) ignorée(s) (SELL sur achat du même cycle).[/]")

    watch_actions = [a for a in actions if a.get("action") == "WATCH"]
    actions       = [a for a in actions if a.get("action") != "WATCH"]
    if watch_actions:
        _apply_watch_actions(backend, watch_actions)
        for w in watch_actions:
            console.print(f"[dim]WATCH {w['symbol']} (brain) : {(w.get('reason') or '')[:100]}[/]")

    if summary:
        console.print(f"[bold]Analyse marché :[/] {summary}\n")

    opp_map = {o["symbol"]: o for o in context.get("candidates", [])}
    pos_map = {p["symbol"]: p for p in context.get("positions", [])}

    holds = [a for a in actions if a.get("action") == "HOLD"]
    if holds:
        for h in holds:
            console.print(f"[dim]HOLD {h['symbol']} : {(h.get('reason') or '')[:100]}[/]")
            log_brain_hold(h["symbol"], h.get("reason"), opp_map.get(h["symbol"]), now, "classic")

    # Mechanical exits: hard TP, early stop (-1.5% @90min), stagnant (<0% @2h)
    actions = _inject_mechanical_exits(actions, context["positions"])

    # Enforce daily stop — block new BUYs if daily loss limit reached
    constraints = context.get("constraints", {})
    if constraints.get("daily_stop_hit"):
        n_blocked = sum(1 for a in actions if a.get("action") == "BUY")
        actions = [a for a in actions if a.get("action") != "BUY"]
        if n_blocked:
            console.print(f"[bold red]Daily stop ({DAILY_STOP_PCT:.0f}%) — {n_blocked} achat(s) bloqué(s)[/]")

    # Block re-entry within 2h of sell
    actions = _filter_cooldown_buys(actions, backend)

    if dry_run:
        console.print("[bold yellow]Mode dry-run — aucun ordre réel passé.[/]")

    executed = backend.execute(actions, context, now, dry_run)
    _record_sell_cooldowns(executed, backend)

    # Log each executed trade with its indicator snapshot
    for e in executed:
        sym = e["symbol"]
        if e["action"] == "BUY":
            log_classic_buy(e, opp_map.get(sym), now)
        elif e["action"] == "SELL":
            log_classic_sell(e, pos_map.get(sym), now)

    _s = backend.get_state("initial_balance")
    if not _s or float(_s) <= 0:
        backend.set_state("initial_balance", str(round(context["total_value"], 2)))

    final_hold   = backend.get_holdings()
    final_prices = get_prices([h.symbol for h in final_hold]) if final_hold else {}
    final_crypto = sum(h.quantity * final_prices.get(h.symbol, h.avg_buy_price) for h in final_hold)
    final_total  = backend.get_usdc() + final_crypto

    if not dry_run:
        backend.add_cycle(now, backend.get_usdc(), final_total, len(executed), summary)

    if executed:
        _display_action_table(executed, f"[bold green]Actions exécutées ({backend.label})[/]")
        _print_portfolio_snapshot(final_hold, final_prices, backend.get_usdc(), context["initial_balance"])
    else:
        console.print("[dim]Aucune action exécutée ce cycle.[/]")

    executed_syms = {e["symbol"] for e in executed}
    n_skip = sum(1 for a in actions
                 if a["action"] != "HOLD" and a["symbol"].upper() not in executed_syms)
    if n_skip:
        console.print(f"[dim]{n_skip} action(s) ignorée(s) (contraintes, données manquantes ou erreurs).[/]")


def run_pump_cycle(
    backend: PortfolioBackend,
    verbose: bool = False,
    dry_run: bool = False,
) -> None:
    """Execute one Tier-2 pump detection cycle."""
    import json
    from ..ml.brain import get_pump_decisions

    now = _now_iso()

    with console.status("[cyan]Scan Tier-2 — collecte des données…[/]"):
        context = collect_pump_context(backend)

    # Extract internal data before sending context to the API
    all_scanned  = context.pop("_all_scanned", [])
    log_pump_candidates(context["candidates"], context.get("filtered_out", []), now, all_scanned)

    if verbose:
        console.print("[bold dim]── Contexte pump envoyé à l'API ──[/]")
        console.print(json.dumps(context, indent=2, ensure_ascii=False))
        console.print("[bold dim]──────────────────────────────────[/]\n")

    usdc   = context["virtual_usdc"]
    total  = context["total_value"]
    pnl    = context["pnl_pct"]
    pc     = "green" if pnl >= 0 else "red"
    candidates    = context["candidates"]
    cand_names    = [c["symbol"] for c in candidates]
    filtered_out  = context.get("filtered_out", [])

    constraints   = context.get("constraints", {})
    risk_scale    = constraints.get("risk_scale", 1.0)
    consec_losses = constraints.get("consecutive_losses", 0)
    avail         = constraints.get("available_usdc", usdc)

    console.print(
        f"[bold]Portefeuille :[/] {usdc:.2f} USDC + {total - usdc:.2f} crypto "
        f"= [bold]{total:.2f} USDC[/] ([{pc}]{pnl:+.1f}%[/]) — disponible {avail:.2f}"
    )
    if risk_scale < 1.0:
        console.print(
            f"[bold yellow]⚠ Drawdown protection :[/] {consec_losses} pertes consécutives "
            f"→ capital réduit à {risk_scale:.0%}"
        )
    if cand_names:
        console.print(f"[dim]Tier-2 → API ({len(cand_names)}) : {', '.join(cand_names)}[/]")
    else:
        console.print("[dim]Aucun candidat Tier-2 — signal absent[/]")
    if filtered_out:
        console.print(f"[dim]Sans signal ({len(filtered_out)}) :[/]")
        for item in filtered_out:
            console.print(f"[dim]  {item['symbol']:<14} {item['reason']}[/]")
    console.print()

    positions_need_review = any(_has_pump_exit_signal(p) for p in context["positions"])
    if len(cand_names) == 0 and not positions_need_review:
        console.print("[dim]Aucun signal d'entrée ni de sortie détecté — appel API ignoré.[/]")
        return

    if dry_run:
        console.print("[bold yellow]Mode dry-run — aucun ordre réel passé.[/]")

    brain_model = __import__("crypto_portfolio.config", fromlist=["BRAIN_MODEL"]).BRAIN_MODEL
    with console.status(f"[cyan]Pump scan — {brain_model}…[/]"):
        decisions = get_pump_decisions(context)

    summary = decisions.get("market_summary", "")
    actions = decisions.get("actions", [])

    actions, n_contra = _drop_contradictory_actions(actions)
    if n_contra:
        console.print(f"[yellow]{n_contra} action(s) contradictoire(s) ignorée(s) (SELL sur achat du même cycle).[/]")

    watch_actions = [a for a in actions if a.get("action") == "WATCH"]
    actions       = [a for a in actions if a.get("action") != "WATCH"]
    if watch_actions:
        _apply_watch_actions(backend, watch_actions)
        for w in watch_actions:
            console.print(f"[dim]WATCH {w['symbol']} (brain) : {(w.get('reason') or '')[:100]}[/]")

    if summary:
        console.print(f"[bold]Analyse :[/] {summary}\n")

    cand_map = {c["symbol"]: c for c in context.get("candidates", [])}
    pos_map  = {p["symbol"]: p for p in context.get("positions", [])}

    holds = [a for a in actions if a.get("action") == "HOLD"]
    if holds:
        for h in holds:
            console.print(f"[dim]HOLD {h['symbol']} : {(h.get('reason') or '')[:100]}[/]")
            log_brain_hold(h["symbol"], h.get("reason"), cand_map.get(h["symbol"]), now, "pump")

    # Mechanical exits: hard TP, early stop (-1.5% @90min), stagnant (<0% @2h)
    actions = _inject_mechanical_exits(actions, context["positions"])

    # Enforce daily stop — block new BUYs if daily loss limit reached
    constraints = context.get("constraints", {})
    if constraints.get("daily_stop_hit"):
        n_blocked = sum(1 for a in actions if a.get("action") == "BUY")
        actions = [a for a in actions if a.get("action") != "BUY"]
        if n_blocked:
            console.print(f"[bold red]Daily stop ({DAILY_STOP_PCT:.0f}%) — {n_blocked} achat(s) bloqué(s)[/]")

    # Block re-entry within 2h of sell
    actions = _filter_cooldown_buys(actions, backend)

    executed = backend.execute(actions, context, now, dry_run)
    _record_sell_cooldowns(executed, backend)

    # Log each executed trade with its indicator snapshot
    for e in executed:
        sym = e["symbol"]
        if e["action"] == "BUY":
            log_pump_buy(e, cand_map.get(sym), now)
        elif e["action"] == "SELL":
            log_pump_sell(e, pos_map.get(sym), now)

    _s = backend.get_state("initial_balance")
    if not _s or float(_s) <= 0:
        backend.set_state("initial_balance", str(round(context["total_value"], 2)))

    final_hold   = backend.get_holdings()
    final_prices = get_prices([h.symbol for h in final_hold]) if final_hold else {}
    final_crypto = sum(h.quantity * final_prices.get(h.symbol, h.avg_buy_price) for h in final_hold)
    final_total  = backend.get_usdc() + final_crypto

    if not dry_run:
        backend.add_cycle(now, backend.get_usdc(), final_total, len(executed), summary)

    if executed:
        _display_action_table(executed, f"[bold magenta]Pump — Actions exécutées ({backend.label})[/]")
        _print_portfolio_snapshot(final_hold, final_prices, backend.get_usdc(), context["initial_balance"])
    else:
        console.print("[dim]Aucune action ce cycle.[/]")

    executed_syms = {e["symbol"] for e in executed}
    n_skip = sum(1 for a in actions
                 if a["action"] != "HOLD" and a["symbol"].upper() not in executed_syms)
    if n_skip:
        console.print(f"[dim]{n_skip} action(s) ignorée(s) (contraintes, données manquantes ou erreurs).[/]")


def run_combined_cycle(
    backend: PortfolioBackend,
    interval_klines: str = "1h",
    ml_interval: str | None = None,
    pool: int = 6,
    verbose: bool = False,
    force_classic: bool = False,
    dry_run: bool = False,
) -> None:
    """Run unified cycle — broad market, composite score selection, compute_metrics enrichment."""
    run_cycle(backend, interval_klines, ml_interval, pool=pool, verbose=verbose, dry_run=dry_run)
