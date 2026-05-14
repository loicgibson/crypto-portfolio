"""
Multi-timeframe feature engineering for pump-detection ML models.

Target : max(high[t+1..t+TARGET_HORIZON]) / close[t] - 1  >=  TARGET_THRESHOLD
Frames : 15m (short-term momentum), 1h (primary), 4h (macro trend context)
"""
import numpy as np
import pandas as pd

from ..config import ML_HORIZON, ML_THRESHOLD

TARGET_THRESHOLD = ML_THRESHOLD   # default 0.05
TARGET_HORIZON   = ML_HORIZON     # default 4

# ── Feature column manifests ──────────────────────────────────────────────────

# Shared indicator kernel — computed at each timeframe, prefixed differently
_KERNEL = [
    "ret_1", "ret_3", "ret_6", "ret_12",
    "close_ma7", "close_ma25", "close_ma99",
    "above_ma25", "above_ma99", "ma25_vs_ma99",
    "ma7_slope", "ma25_slope",
    "rsi", "rsi_trend",
    "macd_hist", "macd_dir",
    "bb_pct", "bb_width", "bb_squeeze",
    "stoch_k", "stoch_d",
    "atr_pct",
    "vol_ratio", "vol_trend_up", "buy_sell_ratio",
    "body_ratio", "upper_wick", "lower_wick", "is_bullish",
    "consec_green", "consec_red",
    "range_pos",
    "roc_5", "roc_14",
]  # 34 indicators per timeframe

# 1h-only extras (pump-specific, time features, long-range indicators)
_EXTRA_1H = [
    "ret_24", "ret_48",
    "close_ma200", "above_ma200",
    "bb_width_pct", "rsi_div",
    "pattern_code",
    "vol_regime", "vol_consecutive",
    "ad_trend",
    "pump_phase_code", "extension_atr", "parabolic_score",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "vol_spike_15m",
]  # 18 indicators, no prefix

FEATURE_COLS = (
    [f"h1_{c}" for c in _KERNEL]         # 34 — 1h kernel
    + _EXTRA_1H                           # 18 — 1h extras
    + [f"h4_{c}" for c in _KERNEL]        # 34 — 4h context
    + [f"m15_{c}" for c in _KERNEL]       # 34 — 15m short-term
    + ["m15_vol_spike"]                   #  1 — 15m intra-hour spike
    + ["btc_ret_1", "btc_ret_24", "btc_ret_48", "btc_above_ma50", "sym_vs_btc_24h"]  # 5
    + ["funding_rate", "funding_ma_7d", "funding_cum_7d"]                             # 3
)
# Total: 34+18+34+34+1+5+3 = 129 features


# ── klines I/O ────────────────────────────────────────────────────────────────

def klines_to_df(rows) -> pd.DataFrame:
    """Convert raw kline rows (SQLite or Binance API) to a typed DataFrame."""
    if not rows:
        return pd.DataFrame()

    first = rows[0]
    if hasattr(first, "keys"):
        df = pd.DataFrame(
            [dict(r) for r in rows],
            columns=["open_time", "open", "high", "low", "close", "volume",
                     "close_time", "quote_volume", "num_trades"],
        )
    else:
        df = pd.DataFrame(rows).iloc[:, :9]
        df.columns = ["open_time", "open", "high", "low", "close", "volume",
                      "close_time", "quote_volume", "num_trades"]

    for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["open_time"] = pd.to_numeric(df["open_time"])
    df["close_time"] = pd.to_numeric(df["close_time"])
    return df.sort_values("open_time").reset_index(drop=True)


# ── Core indicator kernel (applies to any timeframe) ─────────────────────────

def _tf_features(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """
    Compute 34 indicators on df, return DataFrame with same index as df.
    df must have open/high/low/close/volume columns.
    """
    res = pd.DataFrame(index=df.index)
    c   = df["close"]
    h   = df["high"]
    lo  = df["low"]
    vol = df["volume"]
    op  = df["open"]

    # Returns
    for lag in (1, 3, 6, 12):
        res[f"{prefix}ret_{lag}"] = c.pct_change(lag)

    # Moving averages
    ma7  = c.rolling(7,  min_periods=7).mean()
    ma25 = c.rolling(25, min_periods=25).mean()
    ma99 = c.rolling(99, min_periods=99).mean()
    res[f"{prefix}close_ma7"]    = c / ma7.replace(0, np.nan) - 1
    res[f"{prefix}close_ma25"]   = c / ma25.replace(0, np.nan) - 1
    res[f"{prefix}close_ma99"]   = c / ma99.replace(0, np.nan) - 1
    res[f"{prefix}above_ma25"]   = (c > ma25).astype(float)
    res[f"{prefix}above_ma99"]   = (c > ma99).astype(float)
    res[f"{prefix}ma25_vs_ma99"] = (ma25 > ma99).astype(float)
    res[f"{prefix}ma7_slope"]    = ma7.pct_change(3)
    res[f"{prefix}ma25_slope"]   = ma25.pct_change(3)

    # RSI 14
    delta = c.diff()
    gain  = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
    rsi_s = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    res[f"{prefix}rsi"]       = rsi_s
    res[f"{prefix}rsi_trend"] = rsi_s - rsi_s.shift(5)

    # MACD (12/26/9)
    ema12   = c.ewm(span=12, adjust=False).mean()
    ema26   = c.ewm(span=26, adjust=False).mean()
    macd_l  = ema12 - ema26
    macd_h  = macd_l - macd_l.ewm(span=9, adjust=False).mean()
    res[f"{prefix}macd_hist"] = macd_h / c.replace(0, np.nan)
    res[f"{prefix}macd_dir"]  = np.sign(macd_h - macd_h.shift(1))

    # Bollinger (20, 2σ)
    sma20 = c.rolling(20, min_periods=20).mean()
    std20 = c.rolling(20, min_periods=20).std(ddof=0)
    bb_up = sma20 + 2 * std20
    bb_lo = sma20 - 2 * std20
    bb_rng = (bb_up - bb_lo).replace(0, np.nan)
    bb_w   = bb_rng / sma20.replace(0, np.nan)
    res[f"{prefix}bb_pct"]    = (c - bb_lo) / bb_rng
    res[f"{prefix}bb_width"]  = bb_w
    res[f"{prefix}bb_squeeze"] = (
        bb_w < bb_w.rolling(100, min_periods=20).quantile(0.2)
    ).astype(float)

    # Stochastic (14/3)
    lo14 = lo.rolling(14, min_periods=14).min()
    hi14 = h.rolling(14, min_periods=14).max()
    k_raw  = (c - lo14) / (hi14 - lo14).replace(0, np.nan) * 100
    stk    = k_raw.rolling(3, min_periods=3).mean()
    res[f"{prefix}stoch_k"] = stk
    res[f"{prefix}stoch_d"] = stk.rolling(3, min_periods=3).mean()

    # ATR 14 (EWM)
    prev_c = c.shift(1)
    tr  = pd.concat([h - lo, (h - prev_c).abs(), (lo - prev_c).abs()], axis=1).max(axis=1)
    atr = tr.ewm(com=13, adjust=False).mean()
    res[f"{prefix}atr_pct"] = atr / c.replace(0, np.nan)

    # Volume
    vol_ma20 = vol.rolling(20, min_periods=20).mean()
    res[f"{prefix}vol_ratio"]     = vol / vol_ma20.replace(0, np.nan)
    vol_s = vol.rolling(3,  min_periods=3).mean()
    vol_l = vol.rolling(10, min_periods=10).mean()
    res[f"{prefix}vol_trend_up"]  = (vol_s > vol_l).astype(float)
    is_green  = (c > op).astype(float)
    buy_vol   = (is_green * vol).rolling(5, min_periods=1).sum()
    tot_vol   = vol.rolling(5, min_periods=1).sum().replace(0, np.nan)
    res[f"{prefix}buy_sell_ratio"] = buy_vol / tot_vol

    # Candle shape
    rng   = (h - lo).replace(0, np.nan)
    body  = (c - op).abs()
    c_max = pd.concat([c, op], axis=1).max(axis=1)
    c_min = pd.concat([c, op], axis=1).min(axis=1)
    res[f"{prefix}body_ratio"]  = body / rng
    res[f"{prefix}upper_wick"]  = (h - c_max) / rng
    res[f"{prefix}lower_wick"]  = (c_min - lo) / rng
    res[f"{prefix}is_bullish"]  = is_green

    # Consecutive green / red (groupby cumcount approach)
    is_g = (c > op).astype(int)
    is_r = (c < op).astype(int)
    grp_g = (is_g != is_g.shift(fill_value=-1)).cumsum()
    grp_r = (is_r != is_r.shift(fill_value=-1)).cumsum()
    cg = (is_g.groupby(grp_g).cumcount() + 1).where(is_g == 1, 0)
    cr = (is_r.groupby(grp_r).cumcount() + 1).where(is_r == 1, 0)
    res[f"{prefix}consec_green"] = cg.clip(0, 10).astype(float)
    res[f"{prefix}consec_red"]   = cr.clip(0, 10).astype(float)

    # Range position (50-bar window)
    hi50 = h.rolling(50, min_periods=20).max()
    lo50 = lo.rolling(50, min_periods=20).min()
    res[f"{prefix}range_pos"] = (c - lo50) / (hi50 - lo50).replace(0, np.nan)

    # ROC
    res[f"{prefix}roc_5"]  = c.pct_change(5)
    res[f"{prefix}roc_14"] = c.pct_change(14)

    return res


# ── 1h-only extras ────────────────────────────────────────────────────────────

def _extra_1h(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 18 1h-specific indicators. Returns DataFrame with same index as df."""
    res = pd.DataFrame(index=df.index)
    c   = df["close"]
    h   = df["high"]
    lo  = df["low"]
    vol = df["volume"]
    op  = df["open"]

    # Long-range returns
    res["ret_24"] = c.pct_change(24)
    res["ret_48"] = c.pct_change(48)

    # MA200
    ma200 = c.rolling(200, min_periods=200).mean()
    res["close_ma200"] = c / ma200.replace(0, np.nan) - 1
    res["above_ma200"] = (c > ma200).astype(float)

    # BB width percentile
    sma20 = c.rolling(20, min_periods=20).mean()
    std20 = c.rolling(20, min_periods=20).std(ddof=0)
    bb_w  = 4 * std20 / sma20.replace(0, np.nan)
    res["bb_width_pct"] = bb_w.rolling(100, min_periods=20).rank(pct=True)

    # RSI divergence (price vs RSI over 20 bars)
    delta = c.diff()
    gain  = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
    rsi_s = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    pc    = c.pct_change(20)
    rc    = (rsi_s - rsi_s.shift(20)) / 100
    div   = pd.Series(0.0, index=df.index)
    div[(pc < -0.02) & (rc > 0.05)]  =  1.0   # bullish divergence
    div[(pc >  0.02) & (rc < -0.05)] = -1.0   # bearish divergence
    res["rsi_div"] = div

    # Candle pattern code
    rng   = (h - lo).replace(0, np.nan)
    body  = (c - op).abs()
    br    = body / rng
    c_max = pd.concat([c, op], axis=1).max(axis=1)
    c_min = pd.concat([c, op], axis=1).min(axis=1)
    uw    = (h - c_max) / rng
    lw    = (c_min - lo) / rng
    is_b  = c > op
    codes = pd.Series(0.0, index=df.index)
    codes[br < 0.15]                                                    =  1.0  # doji
    codes[~is_b & (lw > 0.6) & (br < 0.3)]                            =  2.0  # hammer
    prev_b = is_b.shift(1).fillna(True)
    codes[is_b & (br > 0.7) & (c > op.shift(1)) & ~prev_b]            =  3.0  # engulfing bull
    codes[is_b & (uw > 0.6) & (br < 0.3)]                             = -1.0  # shooting star
    prev_b2 = is_b.shift(1).fillna(False)
    codes[~is_b & (br > 0.7) & (c < op.shift(1)) & prev_b2]           = -2.0  # engulfing bear
    res["pattern_code"] = codes

    # Volatility regime (7-bar vs 90-day baseline)
    ret  = c.pct_change()
    vs   = ret.rolling(7, min_periods=7).std()
    vl   = ret.rolling(2160, min_periods=100).std()
    res["vol_regime"] = (vs / vl.replace(0, np.nan)).fillna(1.0)

    # Consecutive bars above average volume
    vol_ma20 = vol.rolling(20, min_periods=20).mean()
    above    = (vol > vol_ma20).astype(int)
    grp      = (above != above.shift(fill_value=-1)).cumsum()
    consec   = (above.groupby(grp).cumcount() + 1).where(above == 1, 0)
    res["vol_consecutive"] = consec.clip(0, 10).astype(float)

    # Accumulation / Distribution trend (6-bar)
    hl   = (h - lo).replace(0, np.nan)
    mfm  = ((c - lo) - (h - c)) / hl
    adl  = (mfm * vol).cumsum()
    v6   = vol.rolling(6, min_periods=6).mean().replace(0, np.nan)
    res["ad_trend"] = (adl.diff(6) / (c * v6)).fillna(0.0)

    # Pump phase (ATR extension above MA25)
    ma25 = c.rolling(25, min_periods=25).mean()
    prev_c = c.shift(1)
    tr   = pd.concat([h - lo, (h - prev_c).abs(), (lo - prev_c).abs()], axis=1).max(axis=1)
    atr  = tr.ewm(com=13, adjust=False).mean()
    ext  = (c - ma25) / atr.replace(0, np.nan)
    phase = pd.Series(0.0, index=df.index)
    phase[(ext >= 2) & (ext < 5)] = 1.0   # early/mid
    phase[(ext >= 5) & (ext < 8)] = 2.0   # late
    phase[ext >= 8]               = 3.0   # exhaustion
    res["pump_phase_code"] = phase
    res["extension_atr"]   = ext.clip(-10, 10)
    roc5 = c.pct_change(5).fillna(0) * 100
    res["parabolic_score"] = (
        roc5.clip(0, 30) / 30 * 60 + ext.clip(0, 10) / 10 * 40
    ).clip(0, 100)

    # Time features (cyclical)
    dt   = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    hour = dt.dt.hour + dt.dt.minute / 60
    res["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    res["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    res["dow_sin"]  = np.sin(2 * np.pi * dt.dt.dayofweek / 7)
    res["dow_cos"]  = np.cos(2 * np.pi * dt.dt.dayofweek / 7)

    # vol_spike_15m placeholder (overridden by add_15m_context)
    res["vol_spike_15m"] = 1.0

    return res


# ── Multi-TF merging helpers ──────────────────────────────────────────────────

def _merge_tf(df: pd.DataFrame, tf_df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Merge columns from tf_df into df using merge_asof on open_time (backward).
    Missing values are filled with 0.
    """
    tf_sub = tf_df[["open_time"] + cols].sort_values("open_time")
    merged = pd.merge_asof(
        df.sort_values("open_time"),
        tf_sub,
        on="open_time",
        direction="backward",
        suffixes=("", "_tf"),
    ).reset_index(drop=True)
    for col in cols:
        if col not in merged.columns:
            merged[col] = 0.0
        else:
            merged[col] = merged[col].fillna(0.0)
    return merged.sort_values("open_time").reset_index(drop=True)


# ── BTC market context ────────────────────────────────────────────────────────

def add_market_context(df: pd.DataFrame,
                       btc_df: pd.DataFrame | None,
                       is_btc: bool = False) -> pd.DataFrame:
    """Merge BTC context features into df (merge on open_time)."""
    BTC_COLS = ["btc_ret_1", "btc_ret_24", "btc_ret_48", "btc_above_ma50", "sym_vs_btc_24h"]

    if is_btc or btc_df is None or btc_df.empty:
        df = df.copy()
        for col in ("btc_ret_1", "btc_ret_24", "btc_ret_48", "sym_vs_btc_24h"):
            df[col] = 0.0
        df["btc_above_ma50"] = 1
        return df

    btc = btc_df[["open_time", "close"]].sort_values("open_time").copy()
    btc_c    = btc["close"]
    btc_ma50 = btc_c.rolling(50, min_periods=50).mean()
    ctx = pd.DataFrame({
        "open_time":      btc["open_time"].values,
        "btc_ret_1":      btc_c.pct_change(1).values,
        "btc_ret_24":     btc_c.pct_change(24).values,
        "btc_ret_48":     btc_c.pct_change(48).values,
        "btc_above_ma50": (btc_c > btc_ma50).astype(int).where(btc_ma50.notna(), 1).values,
    })

    df = df.merge(ctx, on="open_time", how="left").copy()
    df["sym_vs_btc_24h"] = df.get("ret_24", pd.Series(0.0, index=df.index)).fillna(0) \
                         - df["btc_ret_24"].fillna(0)
    for col in BTC_COLS:
        df[col] = df[col].fillna(0.0)
    df["btc_above_ma50"] = df["btc_above_ma50"].fillna(1).astype(int)
    return df.reset_index(drop=True)


# ── Funding rates ─────────────────────────────────────────────────────────────

def add_funding_features(df: pd.DataFrame,
                         funding_df: "pd.DataFrame | None") -> pd.DataFrame:
    """Merge perpetual funding rate features into df."""
    FCOLS = ["funding_rate", "funding_ma_7d", "funding_cum_7d"]

    if funding_df is None or funding_df.empty:
        for col in FCOLS:
            df[col] = 0.0
        return df

    fr = funding_df.sort_values("funding_time")[["funding_time", "rate"]].copy()
    fr["rate"] = fr["rate"].astype(float)
    df = df.sort_values("open_time").reset_index(drop=True)
    df = pd.merge_asof(
        df,
        fr.rename(columns={"rate": "funding_rate", "funding_time": "_ft"}),
        left_on="open_time", right_on="_ft", direction="backward",
    ).drop(columns=["_ft"], errors="ignore").copy()
    df["funding_ma_7d"]  = df["funding_rate"].rolling(168, min_periods=1).mean()
    df["funding_cum_7d"] = df["funding_rate"].rolling(168, min_periods=1).sum()
    for col in FCOLS:
        df[col] = df[col].fillna(0.0)
    return df


# ── Target ────────────────────────────────────────────────────────────────────

def add_target(df: pd.DataFrame,
               horizon: int = TARGET_HORIZON,
               threshold: float = TARGET_THRESHOLD,
               sl_frac: float | None = None) -> pd.DataFrame:
    """
    y = 1 if max(high[t+1..t+horizon]) / close[t] - 1 >= threshold, else 0.

    If sl_frac is given (e.g. 0.05 for a 5% stop loss), also requires that
    min(low[t+1..t+horizon]) / close[t] - 1 > -sl_frac — i.e. the stop loss
    is not triggered within the same window. This avoids labelling as winners
    trades that would have been stopped out before the TP is reached.

    Drops the last `horizon` rows (no future data).
    """
    if len(df) <= horizon:
        return df.iloc[0:0]

    highs = [df["high"].shift(-i) for i in range(1, horizon + 1)]
    max_h = pd.concat(highs, axis=1).max(axis=1)
    valid = max_h.notna()
    df    = df[valid].copy()
    max_h = max_h[valid]

    tp_hit = (max_h / df["close"] - 1) >= threshold

    if sl_frac is not None:
        lows  = [df["low"].shift(-i) for i in range(1, horizon + 1)]
        min_l = pd.concat(lows, axis=1).min(axis=1)[valid]
        sl_hit = (min_l / df["close"] - 1) <= -sl_frac
        df["target"] = (tp_hit & ~sl_hit).astype(int)
    else:
        df["target"] = tp_hit.astype(int)

    return df.reset_index(drop=True)


# ── Main entry point ──────────────────────────────────────────────────────────

def compute_features(
    df_1h: pd.DataFrame,
    df_4h: pd.DataFrame | None = None,
    df_15m: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Compute all multi-TF features on df_1h.

    Returns a DataFrame with all FEATURE_COLS columns (except BTC context and
    funding, which are added separately via add_market_context / add_funding_features).
    Rows with NaN in the 1h base indicators are dropped.
    """
    df = df_1h.copy()

    # 1h kernel
    feats_1h = _tf_features(df, "h1_")
    extra_1h = _extra_1h(df)
    df = pd.concat([df, feats_1h, extra_1h], axis=1)

    # 4h context — merge_asof on open_time
    h4_cols = [f"h4_{c}" for c in _KERNEL]
    if df_4h is not None and not df_4h.empty:
        feats_4h = _tf_features(df_4h, "h4_")
        feats_4h.insert(0, "open_time", df_4h["open_time"].values)
        df = _merge_tf(df, feats_4h, h4_cols)
    else:
        df = pd.concat(
            [df, pd.DataFrame(0.0, index=df.index, columns=h4_cols)], axis=1
        )

    # 15m context — merge_asof on open_time
    m15_cols = [f"m15_{c}" for c in _KERNEL] + ["m15_vol_spike"]
    if df_15m is not None and not df_15m.empty:
        feats_15m = _tf_features(df_15m, "m15_")
        # vol_spike: last 15m bar vol / rolling(3) mean of prior bars
        vol_15m = df_15m["volume"]
        feats_15m["m15_vol_spike"] = (
            vol_15m / vol_15m.shift(1).rolling(3, min_periods=1).mean().replace(0, np.nan)
        ).fillna(1.0).values
        feats_15m.insert(0, "open_time", df_15m["open_time"].values)
        df = _merge_tf(df, feats_15m, m15_cols)
        # Override vol_spike_15m (1h extra) with the 15m-derived value
        df["vol_spike_15m"] = df["m15_vol_spike"]
    else:
        df = pd.concat(
            [df, pd.DataFrame(0.0, index=df.index, columns=m15_cols)], axis=1
        )

    # Drop rows with insufficient 1h history (NaN in base kernel cols)
    _external = set(
        [f"h4_{c}" for c in _KERNEL]
        + [f"m15_{c}" for c in _KERNEL]
        + ["m15_vol_spike"]
        + ["btc_ret_1", "btc_ret_24", "btc_ret_48", "btc_above_ma50", "sym_vs_btc_24h"]
        + ["funding_rate", "funding_ma_7d", "funding_cum_7d"]
        + ["vol_regime", "ret_24", "ret_48", "close_ma200", "above_ma200",
           "vol_spike_15m", "ad_trend"]
    )
    base_1h = [c for c in ([f"h1_{k}" for k in _KERNEL] + _EXTRA_1H) if c not in _external]
    df = df.dropna(subset=base_1h).reset_index(drop=True)

    # Fill remaining NaN (mostly from 4h/15m cols with no data at series start)
    fill_cols = [c for c in df.columns if c in set(
        [f"h4_{k}" for k in _KERNEL] + [f"m15_{k}" for k in _KERNEL] + ["m15_vol_spike"]
    )]
    df[fill_cols] = df[fill_cols].fillna(0.0)

    return df
