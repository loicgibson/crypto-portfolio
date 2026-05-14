"""
Comprehensive indicator computation for the pump detection system.

compute_metrics(klines, klines_ext)  — full 15m/1h indicator set
compute_context(klines)              — macro regime summary (1h context)
"""
import bisect

from ..indicators import atr, bollinger, macd, rsi_series, sma, stochastic


# ── Helpers ───────────────────────────────────────────────────────────────────

def _percentile(values: list[float], v: float) -> float:
    """Percentile position of v within values (0–100)."""
    if not values:
        return 50.0
    s = sorted(values)
    return round(bisect.bisect_left(s, v) / len(s) * 100, 1)


def _rsi_divergence(closes: list[float], rsi_vals: list[float], window: int = 20) -> str:
    """
    Detect price/RSI divergence over the last `window` bars.
    Returns: bearish_strong | bearish_weak | none | bullish_weak | bullish_strong
    """
    n = min(window, len(closes), len(rsi_vals))
    if n < 6:
        return "none"
    pw = closes[-n:]
    rw = rsi_vals[-n:]

    def peaks(arr):
        return [i for i in range(1, len(arr) - 1) if arr[i] > arr[i-1] and arr[i] > arr[i+1]]

    def troughs(arr):
        return [i for i in range(1, len(arr) - 1) if arr[i] < arr[i-1] and arr[i] < arr[i+1]]

    pp = peaks(pw)
    if len(pp) >= 2:
        p1, p2 = pp[-2], pp[-1]
        if pw[p2] > pw[p1] and rw[p2] < rw[p1]:
            return "bearish_strong" if (rw[p1] - rw[p2]) > 5 else "bearish_weak"

    pt = troughs(pw)
    if len(pt) >= 2:
        t1, t2 = pt[-2], pt[-1]
        if pw[t2] < pw[t1] and rw[t2] > rw[t1]:
            return "bullish_strong" if (rw[t2] - rw[t1]) > 5 else "bullish_weak"

    return "none"


def _candle_pattern(opens: list[float], highs: list[float],
                    lows: list[float], closes: list[float]) -> str | None:
    """Identify the last candle's reversal pattern."""
    if len(closes) < 2:
        return None
    o, h, l, c = opens[-1], highs[-1], lows[-1], closes[-1]
    po, _ph, _pl, pc = opens[-2], highs[-2], lows[-2], closes[-2]

    rng = h - l
    if rng == 0:
        return None
    body = abs(c - o)
    uw = h - max(o, c)
    lw = min(o, c) - l
    body_ratio = body / rng

    if body_ratio < 0.1:
        return "doji"
    if uw > 2 * body and lw < body * 0.5 and c < o:
        return "shooting_star"
    if lw > 2 * body and uw < body * 0.5 and c > o:
        return "hammer"
    # Engulfing bullish: current green engulfs prior red body
    if c > o and pc < po and c >= po and o <= pc:
        return "engulfing_bull"
    # Engulfing bearish: current red engulfs prior green body
    if c < o and pc > po and o >= pc and c <= po:
        return "engulfing_bear"
    return None


def _pump_phase_metrics(closes: list[float], highs: list[float], lows: list[float],
                         volumes: list[float], ma25: float | None, atr_val: float) -> dict:
    """Breakout detection, pump phase, and extension from MA25."""
    base: dict = {
        "breakout_detected": False,
        "bars_since_breakout": None,
        "gain_since_breakout_pct": None,
        "pump_phase": "none",
        "extension_atr": None,
        "parabolic_score": 0,
    }
    if len(closes) < 20 or atr_val <= 0:
        return base

    close = closes[-1]
    vol_avg = sum(volumes[-10:]) / 10 if len(volumes) >= 10 else volumes[-1]
    vol_ratio = volumes[-1] / vol_avg if vol_avg > 0 else 1.0

    # Resistance = max high of bars [-25:-5] (exclude most recent 5)
    lookback_end = max(5, len(highs) - 20)
    resistance = max(highs[-25:-5]) if len(highs) >= 25 else max(highs[:-5] or highs)

    breakout = False
    bars_since = None
    entry_price = None
    for i in range(1, 6):
        if highs[-i] > resistance:
            # Confirm price was below resistance 3 bars prior
            prior_idx = -i - 3
            if len(closes) > abs(prior_idx) and closes[prior_idx] < resistance:
                breakout = True
                bars_since = i - 1
                entry_price = closes[-i]
                break

    base["breakout_detected"] = breakout
    base["bars_since_breakout"] = bars_since

    gain = None
    if breakout and entry_price and entry_price > 0:
        gain = round((close - entry_price) / entry_price * 100, 2)
    base["gain_since_breakout_pct"] = gain

    ext = None
    if ma25 and atr_val > 0:
        ext = round((close - ma25) / atr_val, 2)
    base["extension_atr"] = ext

    e = ext or 0
    if not breakout:
        phase = "none"
    elif e < 2:
        phase = "early"
    elif e < 5:
        phase = "mid"
    elif e < 8:
        phase = "late"
    else:
        phase = "exhaustion"
    base["pump_phase"] = phase

    score = 0
    if breakout:
        score += 20
        score += min(30, int(e * 5))
        score += min(20, int((vol_ratio - 1) * 5)) if vol_ratio > 1 else 0
        score += min(30, int(gain)) if gain and gain > 0 else 0
    base["parabolic_score"] = min(100, score)

    return base


# ── Main computation ──────────────────────────────────────────────────────────

def compute_metrics(klines: list, klines_ext: list | None = None) -> dict:
    """
    Compute comprehensive indicators from klines.

    klines:     recent bars (≥30) for current indicators
    klines_ext: wider window (≥100) for percentile calculations; falls back to klines
    """
    if len(klines) < 30:
        return {}

    ext = klines_ext or klines

    opens   = [float(k[1]) for k in klines]
    highs   = [float(k[2]) for k in klines]
    lows    = [float(k[3]) for k in klines]
    closes  = [float(k[4]) for k in klines]
    volumes = [float(k[5]) for k in klines]

    volumes_x = [float(k[5]) for k in ext]
    closes_x  = [float(k[4]) for k in ext]
    highs_x   = [float(k[2]) for k in ext]
    lows_x    = [float(k[3]) for k in ext]

    close = closes[-1]
    result: dict = {"close": round(close, 8)}

    # ── Price structure ───────────────────────────────────────────────────────
    n50 = min(50, len(highs))
    h50 = max(highs[-n50:])
    l50 = min(lows[-n50:])
    result["high_local_50"]          = round(h50, 8)
    result["low_local_50"]           = round(l50, 8)
    result["range_position"]         = round((close - l50) / (h50 - l50), 3) if h50 != l50 else 0.5
    recent_high                      = max(highs[-min(20, len(highs)):])
    result["drawdown_from_high_pct"] = round((close - recent_high) / recent_high * 100, 2) if recent_high > 0 else 0.0

    # ── Volume ────────────────────────────────────────────────────────────────
    vol_avg_20 = sum(volumes[-20:]) / min(20, len(volumes))
    vol_ratio  = volumes[-1] / vol_avg_20 if vol_avg_20 > 0 else 1.0
    result["volume_ratio"]  = round(vol_ratio, 2)
    result["volume_avg_20"] = round(vol_avg_20, 4)

    if len(volumes) >= 5:
        v5 = volumes[-5]
        slope = (volumes[-1] - v5) / v5 * 100 if v5 > 0 else 0
        result["volume_trend_5"] = "rising" if slope > 10 else "falling" if slope < -10 else "stable"
    else:
        result["volume_trend_5"] = "stable"

    consec_vol = 0
    for v in reversed(volumes[-10:]):
        if v > vol_avg_20:
            consec_vol += 1
        else:
            break
    result["volume_consecutive_above_avg"] = consec_vol
    result["volume_percentile_100"]        = _percentile(volumes_x, volumes[-1])

    buy_v   = sum(volumes[-5+i] if closes[-5+i] >= opens[-5+i] else 0 for i in range(min(5, len(volumes))))
    total_v = sum(volumes[-5:])
    result["buy_sell_ratio"] = round(buy_v / total_v, 2) if total_v > 0 else 0.5

    # ── MAs and trend ─────────────────────────────────────────────────────────
    ma7  = sma(closes, 7)
    ma25 = sma(closes, 25)
    ma99 = sma(closes, 99) if len(closes) >= 99 else None

    result["ma7"]  = round(ma7,  8) if ma7  else None
    result["ma25"] = round(ma25, 8) if ma25 else None
    result["ma99"] = round(ma99, 8) if ma99 else None

    if ma7 and ma25 and ma99:
        if   ma7 > ma25 > ma99: result["ma_alignment"] = "bullish_strong"
        elif ma7 > ma25:        result["ma_alignment"] = "bullish"
        elif ma7 < ma25 < ma99: result["ma_alignment"] = "bearish"
        else:                   result["ma_alignment"] = "mixed"
    elif ma7 and ma25:
        result["ma_alignment"] = "bullish" if ma7 > ma25 else "bearish"
    else:
        result["ma_alignment"] = "mixed"

    if ma25 and len(closes) >= 28:
        ma25_3ago = sma(closes[:-3], 25)
        result["ma25_slope_pct"] = round((ma25 - ma25_3ago) / ma25_3ago * 100, 4) if ma25_3ago else 0.0
    if ma7 and len(closes) >= 10:
        ma7_3ago = sma(closes[:-3], 7)
        result["ma7_slope_pct"] = round((ma7 - ma7_3ago) / ma7_3ago * 100, 4) if ma7_3ago else 0.0

    if ma7:  result["price_distance_ma7_pct"]  = round((close - ma7)  / ma7  * 100, 2)
    if ma25: result["price_distance_ma25_pct"] = round((close - ma25) / ma25 * 100, 2)

    # ── ATR ───────────────────────────────────────────────────────────────────
    atr_val = atr(highs, lows, closes)
    result["atr_14"]  = round(atr_val, 8)
    result["atr_pct"] = round(atr_val / close * 100, 2) if close > 0 else None

    if ma25 and atr_val > 0:
        result["price_distance_ma25_atr"] = round((close - ma25) / atr_val, 2)

    if len(closes_x) >= 15:
        atrs_x = [atr(highs_x[:i+1], lows_x[:i+1], closes_x[:i+1])
                  for i in range(14, len(closes_x))]
        result["atr_percentile_100"] = _percentile(atrs_x, atr_val)

    # ── Bollinger ─────────────────────────────────────────────────────────────
    bb_upper, bb_mid, bb_lower = bollinger(closes)
    if bb_upper and bb_mid and bb_lower and bb_upper != bb_lower:
        bb_w = (bb_upper - bb_lower) / bb_mid
        result["bb_upper"]    = round(bb_upper, 8)
        result["bb_middle"]   = round(bb_mid, 8)
        result["bb_lower"]    = round(bb_lower, 8)
        result["bb_position"] = round((close - bb_lower) / (bb_upper - bb_lower), 3)
        result["bb_width"]    = round(bb_w, 4)

        if len(closes_x) >= 20:
            bws = []
            for i in range(20, len(closes_x) + 1):
                _u, _m, _l = bollinger(closes_x[:i])
                if _u and _m and _l and _m > 0:
                    bws.append((_u - _l) / _m)
            if bws:
                result["bb_width_percentile_100"] = _percentile(bws, bb_w)
                result["bb_squeeze_active"]       = result["bb_width_percentile_100"] < 20

        # Consecutive candles above BB upper (using current BB as proxy)
        consec_above = 0
        for c_val in reversed(closes[-10:]):
            if c_val > bb_upper:
                consec_above += 1
            else:
                break
        result["candles_above_bb_upper"] = consec_above

    # ── Stochastic ────────────────────────────────────────────────────────────
    stoch_k, stoch_d = stochastic(highs, lows, closes)
    result["stoch_k"] = round(stoch_k, 1)
    result["stoch_d"] = round(stoch_d, 1)

    # ── RSI + momentum ────────────────────────────────────────────────────────
    rsi_vals = rsi_series(closes)
    rsi_now  = rsi_vals[-1]
    result["rsi_14"] = round(rsi_now, 1)

    rsi_5ago = rsi_vals[-6] if len(rsi_vals) >= 6 else rsi_now
    rsi_slope = rsi_now - rsi_5ago
    result["rsi_trend"]     = "rising" if rsi_slope > 3 else "falling" if rsi_slope < -3 else "flat"
    result["rsi_trend_val"] = round(rsi_slope, 1)
    result["rsi_divergence"] = _rsi_divergence(closes, rsi_vals)

    # MACD
    ml, sl, hist = macd(closes)
    result["macd_value"]  = round(ml[-1],   8)
    result["macd_signal"] = round(sl[-1],   8)
    result["macd_hist"]   = round(hist[-1], 8)
    if len(hist) >= 2:
        h0, h1 = hist[-1], hist[-2]
        if   h0 > h1 and h0 > 0:         result["macd_hist_direction"] = "strengthening"
        elif h0 < h1 and h0 < 0:         result["macd_hist_direction"] = "weakening"
        elif (h1 <= 0 < h0) or (h0 < 0 <= h1): result["macd_hist_direction"] = "flipping"
        else:                             result["macd_hist_direction"] = "weakening"

    # ROC and momentum acceleration
    if len(closes) >= 6:
        result["roc_5"] = round((closes[-1] / closes[-6] - 1) * 100, 2)
    if len(closes) >= 15:
        result["roc_14"] = round((closes[-1] / closes[-15] - 1) * 100, 2)
    if len(closes) >= 12:
        roc5_now  = (closes[-1] / closes[-6]  - 1) * 100
        roc5_prev = (closes[-2] / closes[-7]  - 1) * 100
        result["momentum_acceleration"] = round(roc5_now - roc5_prev, 2)

    # ── Candle structure ──────────────────────────────────────────────────────
    o, h, l, c = opens[-1], highs[-1], lows[-1], closes[-1]
    rng = h - l
    if rng > 0:
        body = abs(c - o)
        uw   = h - max(o, c)
        lw   = min(o, c) - l
        result["body_ratio"]       = round(body / rng,   2)
        result["upper_wick_ratio"] = round(uw / body,    2) if body > 0 else 0.0
        result["lower_wick_ratio"] = round(lw / body,    2) if body > 0 else 0.0

    if len(closes) >= 3:
        result["last_3_colors"] = [
            "green" if closes[-3+i] > opens[-3+i] else "red" for i in range(3)
        ]

    cg = cr = 0
    for i in range(-1, -min(11, len(closes) + 1), -1):
        if closes[i] > opens[i]:
            cg += 1
        else:
            break
    for i in range(-1, -min(11, len(closes) + 1), -1):
        if closes[i] < opens[i]:
            cr += 1
        else:
            break
    result["consecutive_green"] = cg
    result["consecutive_red"]   = cr
    result["pattern_detected"]  = _candle_pattern(opens, highs, lows, closes)

    # ── Pump phase ────────────────────────────────────────────────────────────
    result.update(_pump_phase_metrics(closes, highs, lows, volumes, ma25, atr_val))

    return result


# ── 1h macro context ──────────────────────────────────────────────────────────

def compute_context(klines: list) -> dict:
    """
    Compute macro regime context from klines (intended for 1h bars).
    Returns trend, market_phase, key MAs, RSI, ATR context.
    """
    if len(klines) < 30:
        return {}

    closes  = [float(k[4]) for k in klines]
    volumes = [float(k[5]) for k in klines]
    highs   = [float(k[2]) for k in klines]
    lows    = [float(k[3]) for k in klines]
    opens   = [float(k[1]) for k in klines]
    close   = closes[-1]

    ma7_1h  = sma(closes, 7)
    ma25_1h = sma(closes, 25)
    ma99_1h = sma(closes, 99) if len(closes) >= 99 else None

    rsi_1h = rsi_series(closes)[-1]
    atr_1h = atr(highs, lows, closes)

    trend = ("uptrend"   if ma25_1h and close > ma25_1h else
             "downtrend" if ma25_1h and close < ma25_1h else "ranging")

    vol_avg = sum(volumes[-24:]) / min(24, len(volumes))
    vol_r   = sum(volumes[-3:]) / 3 if len(volumes) >= 3 else volumes[-1]
    volume_trend_24h = ("rising"  if vol_r > vol_avg * 1.2 else
                        "falling" if vol_r < vol_avg * 0.8 else "stable")

    if ma7_1h and ma25_1h and ma99_1h:
        if   ma7_1h > ma25_1h > ma99_1h: ma_align = "bullish"
        elif ma7_1h < ma25_1h < ma99_1h: ma_align = "bearish"
        else:                             ma_align = "mixed"
    else:
        ma_align = "mixed"

    # BB width percentile for squeeze detection
    bb_width_pct_1h = None
    bb_upper, bb_mid, bb_lower = bollinger(closes)
    if bb_upper and bb_mid and bb_lower and bb_mid > 0:
        bb_w = (bb_upper - bb_lower) / bb_mid
        if len(closes) >= 20:
            bws = []
            for i in range(20, len(closes) + 1):
                _u, _m, _l = bollinger(closes[:i])
                if _u and _m and _l and _m > 0:
                    bws.append((_u - _l) / _m)
            if bws:
                bb_width_pct_1h = _percentile(bws, bb_w)

    # Market phase
    phase = "ranging"
    if ma25_1h:
        ma25_prev = sma(closes[:-3], 25) if len(closes) >= 28 else ma25_1h
        slope = (ma25_1h - ma25_prev) / ma25_prev * 100 if ma25_prev else 0
        if close > ma25_1h and slope > 0.05 and volume_trend_24h in ("rising", "stable"):
            phase = "markup"
        elif close < ma25_1h and slope < -0.05:
            phase = "markdown"
        elif bb_width_pct_1h is not None and bb_width_pct_1h < 25 and volume_trend_24h != "rising":
            phase = "accumulation"

    return {
        "trend_1h":               trend,
        "market_phase_1h":        phase,
        "ma25_1h":                round(ma25_1h, 8) if ma25_1h else None,
        "ma99_1h":                round(ma99_1h, 8) if ma99_1h else None,
        "price_above_ma99_1h":    bool(ma99_1h and close > ma99_1h),
        "ma_alignment_1h":        ma_align,
        "rsi_1h":                 round(rsi_1h, 1),
        "atr_pct_1h":             round(atr_1h / close * 100, 2) if close > 0 else None,
        "bb_width_percentile_1h": bb_width_pct_1h,
        "volume_trend_24h":       volume_trend_24h,
    }
