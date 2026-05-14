STABLECOINS = {"USDT", "BUSD", "TUSD", "USDP", "DAI", "FDUSD", "USDS", "EUR", "GBP", "WBTC", "WETH"}


def ema(values: list[float], period: int) -> list[float]:
    k = 2 / (period + 1)
    result = [values[0]]
    for v in values[1:]:
        result.append(result[-1] * (1 - k) + v * k)
    return result


def macd(closes: list[float]) -> tuple[list[float], list[float], list[float]]:
    """Returns (macd_line, signal_line, histogram)."""
    ema12 = ema(closes, 12)
    ema26 = ema(closes, 26)
    macd_line = [a - b for a, b in zip(ema12, ema26)]
    signal = ema(macd_line, 9)
    histogram = [m - s for m, s in zip(macd_line, signal)]
    return macd_line, signal, histogram


def bollinger(closes: list[float], period: int = 20, mult: float = 2.0) -> tuple[float | None, float | None, float | None]:
    """Returns (upper, middle, lower) for the last candle."""
    if len(closes) < period:
        return None, None, None
    window = closes[-period:]
    mid = sum(window) / period
    std = (sum((x - mid) ** 2 for x in window) / period) ** 0.5
    return mid + mult * std, mid, mid - mult * std


def stochastic(highs: list[float], lows: list[float], closes: list[float],
               k_period: int = 14, d_period: int = 3) -> tuple[float, float]:
    """Returns (%K, %D) for the last candle."""
    if len(closes) < k_period + d_period:
        return 50.0, 50.0
    k_vals = []
    for i in range(d_period):
        idx = len(closes) - d_period + i
        hh = max(highs[idx - k_period + 1:idx + 1])
        ll = min(lows[idx - k_period + 1:idx + 1])
        k_vals.append((closes[idx] - ll) / (hh - ll) * 100 if hh != ll else 50.0)
    return k_vals[-1], sum(k_vals) / d_period


def atr(highs: list[float], lows: list[float], closes: list[float], period: int = 14) -> float:
    trs = [max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
           for i in range(1, len(closes))]
    return sum(trs[-period:]) / min(period, len(trs)) if trs else 0.0


def ad_line(highs: list[float], lows: list[float], closes: list[float], volumes: list[float]) -> list[float]:
    result, ad = [], 0.0
    for h, l, c, v in zip(highs, lows, closes, volumes):
        mfm = ((c - l) - (h - c)) / (h - l) if h != l else 0.0
        ad += mfm * v
        result.append(ad)
    return result


def sma(closes: list[float], period: int) -> float | None:
    return sum(closes[-period:]) / period if len(closes) >= period else None


def rsi(closes: list[float], period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains = [max(d, 0) for d in deltas[-period:]]
    losses = [max(-d, 0) for d in deltas[-period:]]
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        return 100.0
    return 100 - (100 / (1 + avg_gain / avg_loss))


def rsi_series(closes: list[float], period: int = 14) -> list[float]:
    result = [50.0] * len(closes)
    if len(closes) < period + 1:
        return result
    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains = [max(d, 0) for d in deltas]
    losses = [max(-d, 0) for d in deltas]
    avg_g = sum(gains[:period]) / period
    avg_l = sum(losses[:period]) / period
    for i in range(period, len(deltas)):
        avg_g = (avg_g * (period - 1) + gains[i]) / period
        avg_l = (avg_l * (period - 1) + losses[i]) / period
        rs = avg_g / avg_l if avg_l else float("inf")
        result[i + 1] = 100 - (100 / (1 + rs))
    return result


def golden_cross_recent(closes: list[float], lookback: int = 5) -> bool:
    if len(closes) < 50 + lookback:
        return False
    for i in range(lookback, 0, -1):
        n = len(closes) - i
        ma20_now, ma50_now = sma(closes[:n], 20), sma(closes[:n], 50)
        ma20_prev, ma50_prev = sma(closes[:n - 1], 20), sma(closes[:n - 1], 50)
        if all(v is not None for v in [ma20_now, ma50_now, ma20_prev, ma50_prev]):
            if ma20_prev <= ma50_prev and ma20_now > ma50_now:
                return True
    return False


def death_cross_recent(closes: list[float], lookback: int = 5) -> bool:
    if len(closes) < 50 + lookback:
        return False
    for i in range(lookback, 0, -1):
        n = len(closes) - i
        ma20_now, ma50_now = sma(closes[:n], 20), sma(closes[:n], 50)
        ma20_prev, ma50_prev = sma(closes[:n - 1], 20), sma(closes[:n - 1], 50)
        if all(v is not None for v in [ma20_now, ma50_now, ma20_prev, ma50_prev]):
            if ma20_prev >= ma50_prev and ma20_now < ma50_now:
                return True
    return False


def price_cross_ma_recent(closes: list[float], period: int, lookback: int = 3) -> bool:
    if len(closes) < period + lookback:
        return False
    for i in range(lookback, 0, -1):
        n = len(closes) - i
        ma_now = sma(closes[:n], period)
        ma_prev = sma(closes[:n - 1], period)
        if ma_now and ma_prev and closes[n - 2] <= ma_prev and closes[n - 1] > ma_now:
            return True
    return False
