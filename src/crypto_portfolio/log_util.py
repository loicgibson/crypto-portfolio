"""
Structured logging for pump scan cycles and trade executions.

Two JSONL files per day in log/:
  YYYY-MM-DD_candidates.jsonl  — one line per candidate per pump cycle
  YYYY-MM-DD_trades.jsonl      — one line per executed trade (BUY or SELL)

BUY records include the full metrics snapshot (compute_metrics output), context_1h,
ml_prob_up, ml_ap, and all change/spike fields — enabling retrospective analysis of
any indicator at the exact moment of entry.

SELL records include the full metrics snapshot at exit time, ml fields, pnl, and hold
duration — enabling exit-quality analysis and signal-vs-outcome backtesting.

Usage (backtest a rule):
  import json
  from pathlib import Path
  rows = [json.loads(l) for l in Path("log/2026-05-04_trades.jsonl").read_text().splitlines()]
  buys = [r for r in rows if r["type"] == "BUY"]
  # Access any metric: r["metrics"]["price_distance_ma25_pct"], r["metrics"]["pump_phase"], etc.
"""
import json
from pathlib import Path

_LOG_DIR = Path("log")


def _ensure() -> None:
    _LOG_DIR.mkdir(exist_ok=True)


def _write(path: Path, record: dict) -> None:
    _ensure()
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _date(ts: str) -> str:
    return ts[:10]


def _signal_name(c: dict) -> str:
    """Identify the primary pump signal that fired for a candidate."""
    m             = c.get("metrics") or {}
    vol_ratio     = m.get("volume_ratio")    or c.get("vol_ratio_1h") or 0
    vol_spike_15m = c.get("vol_spike_15m")   or 0
    change_1h     = c.get("change_1h")       or 0
    change_3h     = c.get("change_3h")       or 0
    rsi           = m.get("rsi_14")          or 50
    rsi_trend     = m.get("rsi_trend_val")   or 0

    p1 = vol_ratio > 2.0 and change_1h > 0
    p2 = change_1h > 2.0 and change_3h > 4.0 and vol_ratio > 1.5
    p3 = rsi_trend > 8 and rsi < 55
    p4 = vol_spike_15m > 3.0 and change_1h > 0

    if p2: return "p2"
    if p1: return "p1"
    if p4: return "p4"
    if p3: return "p3"
    return "none"


def _candidate_record(ts: str, c: dict, status: str, filter_reason: str | None = None) -> dict:
    return {
        "ts":            ts,
        "symbol":        c["symbol"],
        "status":        status,
        "filter_reason": filter_reason,
        "signal":        _signal_name(c) if status == "candidate" else None,
        "change_1h":     c.get("change_1h"),
        "change_3h":     c.get("change_3h"),
        "change_6h":     c.get("change_6h"),
        "change_24h":    c.get("change_24h"),
        "vol_spike_15m": c.get("vol_spike_15m"),
        "ml_prob_up":    c.get("ml_prob_up"),
        "ml_ap":         c.get("ml_ap"),
        "metrics":       c.get("metrics") or None,
        "context_1h":    c.get("context_1h") or None,
    }


def log_pump_candidates(
    candidates: list[dict],
    filtered_out: list[dict],
    ts: str,
    all_scanned: list[dict] | None = None,
) -> None:
    """
    Log every symbol scanned in a pump cycle.

    - candidates   : passed _has_pump_entry_signal
    - filtered_out : list of {"symbol": ..., "reason": ...}
    - all_scanned  : raw candidate dicts before filtering
    """
    path = _LOG_DIR / f"{_date(ts)}_candidates.jsonl"
    scanned_map = {c["symbol"]: c for c in (all_scanned or [])}

    for c in candidates:
        _write(path, _candidate_record(ts, c, "candidate"))

    for f in filtered_out:
        sym = f["symbol"]
        c   = scanned_map.get(sym) or {"symbol": sym}
        _write(path, _candidate_record(ts, c, "filtered", f.get("reason")))


def log_pump_buy(executed: dict, candidate: dict | None, ts: str) -> None:
    """Log an executed BUY with full feature snapshot from the pump cycle."""
    path = _LOG_DIR / f"{_date(ts)}_trades.jsonl"
    c    = candidate or {}
    _write(path, {
        "ts":          ts,
        "type":        "BUY",
        "cycle":       "pump",
        "symbol":      executed["symbol"],
        "price":       executed.get("price"),
        "quantity":    executed.get("qty"),
        "usdc":        executed.get("usdc_spent"),
        "signal":      _signal_name(c),
        "change_1h":   c.get("change_1h"),
        "change_3h":   c.get("change_3h"),
        "change_6h":   c.get("change_6h"),
        "change_24h":  c.get("change_24h"),
        "vol_spike_15m": c.get("vol_spike_15m"),
        "ml_prob_up":  c.get("ml_prob_up"),
        "ml_ap":       c.get("ml_ap"),
        "metrics":     c.get("metrics") or None,
        "context_1h":  c.get("context_1h") or None,
        "reason":      executed.get("reason"),
    })


def log_pump_sell(executed: dict, position: dict | None, ts: str) -> None:
    """Log an executed SELL with full feature snapshot at exit time."""
    path = _LOG_DIR / f"{_date(ts)}_trades.jsonl"
    pos  = position or {}
    buy_usdc = (
        (pos.get("avg_buy_price") or 0) * (executed.get("qty") or 0)
        if pos.get("avg_buy_price") else None
    )
    delta = (
        (executed.get("proceeds") or 0) - buy_usdc
        if buy_usdc is not None else None
    )
    _write(path, {
        "ts":            ts,
        "type":          "SELL",
        "cycle":         "pump",
        "symbol":        executed["symbol"],
        "price":         executed.get("price"),
        "quantity":      executed.get("qty"),
        "usdc":          executed.get("proceeds"),
        "pnl_pct":       executed.get("pnl_pct"),
        "delta_usdc":    round(delta, 4) if delta is not None else None,
        "avg_buy_price": pos.get("avg_buy_price"),
        "hold_hours":    pos.get("held_hours"),
        "ml_prob_up":    pos.get("ml_prob_up"),
        "ml_ap":         pos.get("ml_ap"),
        "metrics":       pos.get("metrics") or None,
        "reason":        executed.get("reason"),
    })


def log_classic_buy(executed: dict, opportunity: dict | None, ts: str) -> None:
    """Log an executed BUY from the classic (Tier-1) cycle."""
    path = _LOG_DIR / f"{_date(ts)}_trades.jsonl"
    opp  = opportunity or {}
    ind  = opp.get("indicators") or {}
    _write(path, {
        "ts":         ts,
        "type":       "BUY",
        "cycle":      "classic",
        "symbol":     executed["symbol"],
        "price":      executed.get("price"),
        "quantity":   executed.get("qty"),
        "usdc":       executed.get("usdc_spent"),
        "change_24h": opp.get("change_24h"),
        "vol_usdc":   opp.get("vol_usdc"),
        "rsi":        ind.get("rsi"),
        "rsi_trend":  ind.get("rsi_trend"),
        "macd_dir":   ind.get("macd_dir"),
        "bb_pct":     ind.get("bb_pct"),
        "above_ma20": ind.get("above_ma20"),
        "ml_prob_up": opp.get("ml_prob_up"),
        "earn_apr":   opp.get("earn_apr"),
        "reason":     executed.get("reason"),
    })


def log_brain_hold(symbol: str, brain_reason: str | None, candidate: dict | None, ts: str, cycle: str = "pump") -> None:
    """Log a brain HOLD decision with the candidate/opportunity feature snapshot."""
    path = _LOG_DIR / f"{_date(ts)}_candidates.jsonl"
    c = candidate or {}
    ind = c.get("indicators") or {}
    record: dict = {
        "ts":           ts,
        "symbol":       symbol,
        "status":       "brain_hold",
        "cycle":        cycle,
        "brain_reason": brain_reason,
    }
    if cycle == "pump" or cycle == "watchlist":
        record.update({
            "signal":        _signal_name(c),
            "change_1h":     c.get("change_1h"),
            "change_3h":     c.get("change_3h"),
            "change_6h":     c.get("change_6h"),
            "change_24h":    c.get("change_24h"),
            "vol_spike_15m": c.get("vol_spike_15m"),
            "ml_prob_up":    c.get("ml_prob_up"),
            "ml_ap":         c.get("ml_ap"),
            "metrics":       c.get("metrics") or None,
            "context_1h":    c.get("context_1h") or None,
        })
    else:  # classic
        record.update({
            "change_24h": c.get("change_24h"),
            "vol_usdc":   c.get("vol_usdc"),
            "ml_prob_up": c.get("ml_prob_up"),
            "earn_apr":   c.get("earn_apr"),
            "rsi":        ind.get("rsi"),
            "rsi_trend":  ind.get("rsi_trend"),
            "macd_dir":   ind.get("macd_dir"),
            "bb_pct":     ind.get("bb_pct"),
            "above_ma20": ind.get("above_ma20"),
        })
    _write(path, record)


def log_classic_sell(executed: dict, position: dict | None, ts: str) -> None:
    """Log an executed SELL from the classic (Tier-1) cycle."""
    path = _LOG_DIR / f"{_date(ts)}_trades.jsonl"
    pos  = position or {}
    buy_usdc = (
        (pos.get("avg_buy_price") or 0) * (executed.get("qty") or 0)
        if pos.get("avg_buy_price") else None
    )
    delta = (
        (executed.get("proceeds") or 0) - buy_usdc
        if buy_usdc is not None else None
    )
    _write(path, {
        "ts":            ts,
        "type":          "SELL",
        "cycle":         "classic",
        "symbol":        executed["symbol"],
        "price":         executed.get("price"),
        "quantity":      executed.get("qty"),
        "usdc":          executed.get("proceeds"),
        "pnl_pct":       executed.get("pnl_pct"),
        "delta_usdc":    round(delta, 4) if delta is not None else None,
        "avg_buy_price": pos.get("avg_buy_price"),
        "hold_hours":    pos.get("held_hours"),
        "ml_prob_up":    pos.get("ml_prob_up"),
        "reason":        executed.get("reason"),
    })
