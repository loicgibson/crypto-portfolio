"""
Tests for _has_pump_entry_signal and _pump_filter_reason.

Test cases are grounded in real trading data from the 2026-05-04/05 session.
Each test documents the asset, timestamp, real outcome, and which rule it exercises.
When adding a new filter rule, add a test case here before implementing it.
"""
import pytest
from crypto_portfolio.commands._market import _has_pump_entry_signal, _pump_filter_reason


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _c(
    symbol="TEST",
    change_1h=2.5, change_3h=5.0, change_6h=3.0, change_24h=10.0,
    vol_ratio_1h=2.5, vol_spike_15m=0.0,
    rsi=65.0, rsi_trend=10.0, macd_dir="up",
    bb_pct=0.85,
    above_ma20=True, stoch_k=75.0, atr_pct=2.0,
    candles_1h=None,
):
    """Build a minimal candidate dict. Defaults represent a healthy p2 entry."""
    macd_hist_dir = "strengthening" if macd_dir == "up" else "weakening"

    # Derive consecutive_green and volume_trend_5 from candles_1h if provided
    if candles_1h and len(candles_1h) >= 2:
        cg = 0
        for cand in reversed(candles_1h):
            if cand[1] > cand[0]:   # close > open
                cg += 1
            else:
                break
        vol_trend = (
            "rising" if candles_1h[-1][2] >= candles_1h[-2][2] else "stable"
        )
    else:
        cg = 2 if above_ma20 else 0
        vol_trend = "stable"

    return {
        "symbol":        symbol,
        "change_1h":     change_1h,
        "change_3h":     change_3h,
        "change_6h":     change_6h,
        "change_24h":    change_24h,
        "vol_spike_15m": vol_spike_15m,
        "candles_1h":    candles_1h or [],
        "metrics": {
            "rsi_14":                  rsi,
            "rsi_trend_val":           rsi_trend,
            "macd_hist_direction":     macd_hist_dir,
            "bb_position":             bb_pct,
            "stoch_k":                 stoch_k,
            "atr_pct":                 atr_pct,
            "volume_ratio":            vol_ratio_1h,
            "volume_trend_5":          vol_trend,
            "consecutive_green":       cg,
            "price_distance_ma25_pct": 1.0 if above_ma20 else -1.0,
        },
    }


# ---------------------------------------------------------------------------
# Sanity check: baseline healthy entry should pass
# ---------------------------------------------------------------------------

def test_baseline_passes():
    """Defaults represent a clean p2 entry — must pass."""
    assert _has_pump_entry_signal(_c())


# ---------------------------------------------------------------------------
# Hard filter: change_24h > 50 % — exhausted pump
# ---------------------------------------------------------------------------

def test_change_24h_exhaustion_blocks():
    """
    DOGS 03:45 — change_24h=64 %, still pumping but already ran 64 % in 24 h.
    Rule: change_24h > 50 % → block unconditionally.
    """
    c = _c(symbol="DOGS", change_1h=14.97, change_3h=20.0, change_24h=64.0,
           vol_ratio_1h=10.0, rsi=95.0, rsi_trend=23.0, stoch_k=95.0)
    assert not _has_pump_entry_signal(c)
    assert "50" in _pump_filter_reason(c)


def test_change_24h_at_threshold_passes():
    """change_24h == 50 is NOT > 50 — should still be evaluated."""
    c = _c(change_24h=50.0)
    assert _has_pump_entry_signal(c)


# ---------------------------------------------------------------------------
# Hard filter: RSI > 90 AND rsi_trend < 5 — extreme overbought, stalling
# ---------------------------------------------------------------------------

def test_rsi_extreme_stalling_blocks():
    """
    Generic exhaustion pattern: RSI way above 90, momentum dying.
    Rule: RSI > 90 AND rsi_trend < 5 → block.
    """
    c = _c(rsi=92.0, rsi_trend=3.0, stoch_k=80.0,
           change_1h=5.0, vol_ratio_1h=3.0)
    assert not _has_pump_entry_signal(c)
    assert "exhausted" in _pump_filter_reason(c)


def test_rsi_extreme_but_strong_trend_passes():
    """
    DOGS 02:56 — RSI=91.8, rsi_trend=23.7, vol×14. +17 % on that candle.
    RSI > 90 but momentum is still explosive → should reach Claude.
    """
    c = _c(symbol="DOGS", change_1h=17.89, change_3h=21.08, change_24h=35.6,
           vol_ratio_1h=14.61, rsi=91.8, rsi_trend=23.7, stoch_k=78.0,
           above_ma20=True)
    assert _has_pump_entry_signal(c)


def test_rsi_exactly_90_passes():
    """RSI == 90 is NOT > 90 — boundary check."""
    c = _c(rsi=90.0, rsi_trend=3.0, stoch_k=80.0)
    assert _has_pump_entry_signal(c)


# ---------------------------------------------------------------------------
# Hard filter: stoch_k > 95 AND rsi_trend < 10 — slow grind at multi-hour top
# ---------------------------------------------------------------------------

def test_act_blocked_stoch_grind():
    """
    ACT 05:55 — stoch_k=100, rsi_trend=6.5. Outcome: -3 % (-24 USDC).
    ACT ground up to its 14-h high over 2 hours with weak momentum.
    Early holders were ready to sell; entering here was chasing.
    Rule: stoch_k > 95 AND rsi_trend < 10 → block.
    """
    c = _c(symbol="ACT",
           change_1h=2.45, change_3h=4.37, change_24h=11.33,
           vol_ratio_1h=2.08, vol_spike_15m=0.46,
           rsi=69.3, rsi_trend=6.5, macd_dir="up",
           bb_pct=1.016, stoch_k=100.0, above_ma20=True)
    assert not _has_pump_entry_signal(c)
    reason = _pump_filter_reason(c)
    assert "stoch" in reason
    assert "rsi_trend" in reason


def test_lunc_blocked_stoch_grind():
    """
    LUNC 21:27 — stoch_k=97, rsi_trend=9.6. Outcome: -2.8 % (-51 USDC).
    Same slow-grind pattern as ACT: stoch at peak, weak momentum acceleration.
    """
    c = _c(symbol="LUNC",
           change_1h=1.18, change_3h=10.22, change_24h=5.0,
           vol_ratio_1h=3.34, rsi=73.1, rsi_trend=9.6,
           bb_pct=1.10, stoch_k=97.0, above_ma20=True)
    assert not _has_pump_entry_signal(c)


def test_not_passes_stoch_100_strong_trend():
    """
    NOT 02:39 — stoch_k=100, rsi_trend=21.1. Would have been +7 % if filter had allowed it.
    stoch is maxed but RSI is accelerating at 21 pts/h → genuine breakout, not a grind.
    Rule: stoch_k > 95 AND rsi_trend >= 10 → should pass.
    """
    c = _c(symbol="NOT",
           change_1h=2.41, change_3h=4.24, change_24h=15.6,
           vol_ratio_1h=2.33, rsi=81.8, rsi_trend=21.1,
           bb_pct=1.124, stoch_k=100.0, above_ma20=True)
    assert _has_pump_entry_signal(c)


def test_ilv_passes_stoch_100_decent_trend():
    """
    ILV 07:17 — stoch_k=100, rsi_trend=11.6 (actual executed buy).
    rsi_trend just above the 10-pt threshold → should pass the filter.
    """
    c = _c(symbol="ILV",
           change_1h=0.43, change_3h=1.51, change_24h=5.0,
           vol_ratio_1h=2.32, rsi=63.8, rsi_trend=11.6,
           stoch_k=100.0, macd_dir="up", above_ma20=True)
    assert _has_pump_entry_signal(c)


def test_stoch_exactly_95_passes():
    """stoch_k == 95 is NOT > 95 — boundary check."""
    c = _c(stoch_k=95.0, rsi_trend=5.0)
    assert _has_pump_entry_signal(c)


def test_rsi_trend_exactly_10_passes():
    """rsi_trend == 10 is NOT < 10 — boundary check."""
    c = _c(stoch_k=97.0, rsi_trend=10.0)
    assert _has_pump_entry_signal(c)


# ---------------------------------------------------------------------------
# Previously missed good entries (old RSI > 80 block, now removed)
# ---------------------------------------------------------------------------

def test_dogs_passes_high_rsi_strong_momentum():
    """
    DOGS 01:50 — RSI=82, rsi_trend=16.4, vol_ratio=1.95. Would have been +19–27 %.
    Old rule blocked at RSI > 80. New rule: RSI ≤ 90, stoch=94.7 < 95 → passes.
    """
    c = _c(symbol="DOGS",
           change_1h=3.26, change_3h=7.34, change_24h=16.3,
           vol_ratio_1h=1.95, rsi=82.0, rsi_trend=16.4,
           bb_pct=1.142, stoch_k=94.7, above_ma20=True)
    assert _has_pump_entry_signal(c)


def test_dogs_passes_rsi_88_explosive_volume():
    """
    DOGS 02:39 — RSI=88, rsi_trend=19.9, vol×10. Would have been +11–39 %.
    RSI is high but not > 90; stoch=78 < 95 → no hard block.
    """
    c = _c(symbol="DOGS",
           change_1h=7.63, change_3h=10.54, change_24h=24.8,
           vol_ratio_1h=10.29, rsi=88.0, rsi_trend=19.9,
           bb_pct=1.28, stoch_k=78.2, above_ma20=True)
    assert _has_pump_entry_signal(c)


# ---------------------------------------------------------------------------
# Primary signal logic
# ---------------------------------------------------------------------------

def test_p1_volume_spike_with_positive_candle():
    """p1: vol_ratio_1h > 2.0 AND change_1h > 0."""
    c = _c(change_1h=0.5, change_3h=1.0, vol_ratio_1h=2.5)
    assert _has_pump_entry_signal(c)


def test_p2_velocity_building():
    """p2: change_1h > 2 AND change_3h > 4 AND vol_ratio > 1.5."""
    c = _c(change_1h=2.5, change_3h=5.0, vol_ratio_1h=1.8,
           # vol below 2 so p1 doesn't fire, but p2 should
           )
    assert _has_pump_entry_signal(c)


def test_p3_rsi_breakout_from_low_base():
    """p3: rsi_trend > 8 AND rsi < 55 — RSI breaking out of neutral zone."""
    c = _c(change_1h=0.5, change_3h=1.0, vol_ratio_1h=0.8,
           rsi=52.0, rsi_trend=10.0)
    assert _has_pump_entry_signal(c)


def test_p4_vol_spike_15m():
    """p4: vol_spike_15m > 3.0 AND change_1h > 0."""
    c = _c(change_1h=0.3, change_3h=0.5, vol_ratio_1h=0.9,
           vol_spike_15m=4.0)
    assert _has_pump_entry_signal(c)


def test_no_primary_signal_blocks():
    """No primary signal → filtered regardless of other indicators."""
    c = _c(change_1h=0.5, change_3h=2.0, vol_ratio_1h=1.2,
           vol_spike_15m=0.5, rsi=60.0, rsi_trend=3.0)
    assert not _has_pump_entry_signal(c)
    assert "no_primary" in _pump_filter_reason(c)


# ---------------------------------------------------------------------------
# Confirmation logic (need at least one of above_ma20 / macd_up / consec_bull)
# ---------------------------------------------------------------------------

def test_no_confirmation_blocks():
    """Primary signal fires but none of the three confirmations present."""
    c = _c(change_1h=3.0, change_3h=5.0, vol_ratio_1h=2.5,
           above_ma20=False, macd_dir="down", candles_1h=[])
    assert not _has_pump_entry_signal(c)


def test_consec_bull_candles_as_confirmation():
    """Two consecutive bullish candles with rising vol_ratio count as confirmation."""
    candles = [
        [1.0, 1.05, 0.8],   # older: green
        [1.05, 1.10, 1.2],  # newer: green, higher vol_ratio
    ]
    c = _c(above_ma20=False, macd_dir="down", candles_1h=candles,
           change_1h=3.0, change_3h=5.0, vol_ratio_1h=2.5)
    assert _has_pump_entry_signal(c)
