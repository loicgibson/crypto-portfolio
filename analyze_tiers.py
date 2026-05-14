import sqlite3
import math
import sys
from collections import defaultdict

# Force UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

DB_PATH = r'c:\Users\loicg\crypto-portfolio\portfolio.db'
MIN_CANDLES = 500

def connect():
    return sqlite3.connect(DB_PATH)

def get_eligible_symbols(conn):
    cur = conn.cursor()
    cur.execute("""
        SELECT symbol, COUNT(*) as cnt
        FROM klines
        WHERE interval = '1h'
        GROUP BY symbol
        HAVING cnt >= ?
        ORDER BY cnt DESC
    """, (MIN_CANDLES,))
    return cur.fetchall()

def get_symbol_data(conn, symbol):
    cur = conn.cursor()
    cur.execute("""
        SELECT open_time, open, high, low, close, quote_volume
        FROM klines
        WHERE interval = '1h' AND symbol = ?
        ORDER BY open_time ASC
    """, (symbol,))
    return cur.fetchall()

def mean(lst):
    if not lst: return 0.0
    return sum(lst) / len(lst)

def std(lst):
    if len(lst) < 2: return 0.0
    m = mean(lst)
    variance = sum((x - m) ** 2 for x in lst) / (len(lst) - 1)
    return math.sqrt(variance)

def percentile(lst, p):
    if not lst: return 0.0
    sorted_lst = sorted(lst)
    idx = (len(sorted_lst) - 1) * p / 100.0
    lo = int(idx)
    hi = lo + 1
    if hi >= len(sorted_lst):
        return sorted_lst[lo]
    frac = idx - lo
    return sorted_lst[lo] + frac * (sorted_lst[hi] - sorted_lst[lo])

def analyze_symbol(rows):
    # rows: (open_time_ms, open, high, low, close, quote_volume)
    # Group into days
    daily_data = defaultdict(lambda: {'highs': [], 'lows': [], 'closes': [], 'volumes': []})

    for open_time, o, h, l, c, qv in rows:
        day_key = open_time // (86400 * 1000)  # day bucket
        daily_data[day_key]['highs'].append(float(h))
        daily_data[day_key]['lows'].append(float(l))
        daily_data[day_key]['closes'].append(float(c))
        daily_data[day_key]['volumes'].append(float(qv))

    days = sorted(daily_data.keys())

    # Daily close (last close of day), daily high/low, daily volume
    daily_closes = []
    daily_highs = []
    daily_lows = []
    daily_volumes = []

    for d in days:
        dd = daily_data[d]
        daily_closes.append(dd['closes'][-1])  # last close of the day
        daily_highs.append(max(dd['highs']))
        daily_lows.append(min(dd['lows']))
        daily_volumes.append(sum(dd['volumes']))

    # avg_daily_vol_usdc
    avg_daily_vol_usdc = mean(daily_volumes)

    # daily_return_std: std of daily close-to-close returns in %
    daily_returns = []
    for i in range(1, len(daily_closes)):
        if daily_closes[i-1] > 0:
            ret = (daily_closes[i] / daily_closes[i-1] - 1) * 100
            daily_returns.append(ret)
    daily_return_std = std(daily_returns)

    # ATR as % of close: avg of 14-day rolling ATR / close
    # We compute true range per day vs prior day close
    atr_pcts = []
    for i in range(1, len(days)):
        tr = max(
            daily_highs[i] - daily_lows[i],
            abs(daily_highs[i] - daily_closes[i-1]),
            abs(daily_lows[i] - daily_closes[i-1])
        )
        if daily_closes[i] > 0:
            atr_pcts.append(tr / daily_closes[i] * 100)
    # Use 14-day rolling average of ATR%
    rolling_atr_pcts = []
    for i in range(13, len(atr_pcts)):
        rolling_atr_pcts.append(mean(atr_pcts[i-13:i+1]))
    atr_pct_avg = mean(rolling_atr_pcts) if rolling_atr_pcts else mean(atr_pcts)

    # max_1d_gain_pct, max_1d_loss_pct
    max_1d_gain_pct = max(daily_returns) if daily_returns else 0.0
    max_1d_loss_pct = min(daily_returns) if daily_returns else 0.0

    # pump_events: using 1h candles for rolling 24h window
    # Use close prices from 1h candles
    closes_1h = [float(r[4]) for r in rows]

    pump_20 = 0
    pump_40 = 0
    # rolling 24-candle window (24h)
    for i in range(24, len(closes_1h)):
        if closes_1h[i-24] > 0:
            gain = (closes_1h[i] / closes_1h[i-24] - 1) * 100
            if gain > 20:
                pump_20 += 1
            if gain > 40:
                pump_40 += 1

    # date range
    first_ts = rows[0][0] / 1000
    last_ts = rows[-1][0] / 1000
    import datetime
    first_date = datetime.datetime.utcfromtimestamp(first_ts).strftime('%Y-%m-%d')
    last_date = datetime.datetime.utcfromtimestamp(last_ts).strftime('%Y-%m-%d')
    date_range = f"{first_date} to {last_date}"

    candle_count = len(rows)

    return {
        'avg_daily_vol_usdc': avg_daily_vol_usdc,
        'daily_return_std': daily_return_std,
        'atr_pct_avg': atr_pct_avg,
        'max_1d_gain_pct': max_1d_gain_pct,
        'max_1d_loss_pct': max_1d_loss_pct,
        'pump_events_20pct': pump_20,
        'pump_events_40pct': pump_40,
        'candle_count': candle_count,
        'date_range': date_range,
        'daily_returns': daily_returns,  # keep for threshold analysis
        'closes_1h': closes_1h,          # keep for RSI/volume analysis
        'daily_volumes': daily_volumes,
    }

def compute_rsi(closes, period=14):
    """Compute RSI for a list of closes, returns list of RSI values (same length, NaN for early ones)."""
    rsi_values = [None] * len(closes)
    if len(closes) <= period:
        return rsi_values

    gains = []
    losses = []
    for i in range(1, period + 1):
        change = closes[i] - closes[i-1]
        gains.append(max(change, 0))
        losses.append(max(-change, 0))

    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period

    for i in range(period, len(closes)):
        if i > period:
            change = closes[i] - closes[i-1]
            avg_gain = (avg_gain * (period - 1) + max(change, 0)) / period
            avg_loss = (avg_loss * (period - 1) + max(-change, 0)) / period

        if avg_loss == 0:
            rsi_values[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi_values[i] = 100 - (100 / (1 + rs))

    return rsi_values

def compute_bollinger_bands(closes, period=20, num_std=2.0):
    """Returns list of (middle, upper, lower) or None for early values."""
    bands = [None] * len(closes)
    for i in range(period - 1, len(closes)):
        window = closes[i - period + 1: i + 1]
        m = mean(window)
        s = std(window)
        bands[i] = (m, m + num_std * s, m - num_std * s)
    return bands

def analyze_pump_patterns(rows, pump_threshold_pct=20.0):
    """
    For each 24h pump event, look back:
    - avg RSI in 6h before pump start
    - avg volume ratio in 3h before pump (current / 7d avg)
    - % of pumps following a BB squeeze (bandwidth < 4% for 3+ hours)
    """
    closes = [float(r[4]) for r in rows]
    volumes = [float(r[5]) for r in rows]

    # Compute RSI on 1h closes
    rsi_values = compute_rsi(closes, period=14)

    # Compute Bollinger Band bandwidth (upper-lower)/middle as %
    bb = compute_bollinger_bands(closes, period=20)
    bb_bw = []
    for b in bb:
        if b is None:
            bb_bw.append(None)
        else:
            m, u, l = b
            bw = (u - l) / m * 100 if m > 0 else None
            bb_bw.append(bw)

    # 7-day rolling volume average (168 hourly candles)
    vol_7d_avg = []
    for i in range(len(volumes)):
        if i < 168:
            vol_7d_avg.append(mean(volumes[:i+1]))
        else:
            vol_7d_avg.append(mean(volumes[i-167:i+1]))

    # Find pump events (same rolling 24h logic)
    pump_indices = []
    for i in range(24, len(closes)):
        if closes[i-24] > 0:
            gain = (closes[i] / closes[i-24] - 1) * 100
            if gain > pump_threshold_pct:
                pump_indices.append(i)  # i = end of 24h pump window; start = i-24

    # Deduplicate: keep only first index in each run of consecutive pumps
    # (to avoid counting the same pump event multiple times)
    deduped_pumps = []
    last_pump = -25
    for idx in pump_indices:
        if idx - last_pump >= 24:
            deduped_pumps.append(idx)
            last_pump = idx

    if not deduped_pumps:
        return None

    pump_start_indices = [i - 24 for i in deduped_pumps]

    rsi_before_list = []
    vol_ratio_list = []
    bb_squeeze_count = 0

    for start_idx in pump_start_indices:
        # RSI: avg of 6 candles before pump start (start_idx-6 to start_idx-1)
        rsi_window = []
        for j in range(max(0, start_idx - 6), start_idx):
            if rsi_values[j] is not None:
                rsi_window.append(rsi_values[j])
        if rsi_window:
            rsi_before_list.append(mean(rsi_window))

        # Volume ratio: avg of 3h before pump start
        vol_ratio_window = []
        for j in range(max(0, start_idx - 3), start_idx):
            if vol_7d_avg[j] > 0:
                ratio = volumes[j] / vol_7d_avg[j]
                vol_ratio_window.append(ratio)
        if vol_ratio_window:
            vol_ratio_list.append(mean(vol_ratio_window))

        # BB squeeze: check if bb bandwidth < 4% for at least 3 of the 6 hours before pump
        squeeze_hours = 0
        for j in range(max(0, start_idx - 6), start_idx):
            if bb_bw[j] is not None and bb_bw[j] < 4.0:
                squeeze_hours += 1
        if squeeze_hours >= 3:
            bb_squeeze_count += 1

    n_pumps = len(deduped_pumps)
    return {
        'n_pump_events': n_pumps,
        'avg_rsi_before': mean(rsi_before_list) if rsi_before_list else None,
        'avg_vol_ratio_before': mean(vol_ratio_list) if vol_ratio_list else None,
        'pct_after_bb_squeeze': bb_squeeze_count / n_pumps * 100 if n_pumps > 0 else 0,
    }

def format_vol(v):
    if v >= 1e9:
        return f"{v/1e9:.2f}B"
    elif v >= 1e6:
        return f"{v/1e6:.2f}M"
    elif v >= 1e3:
        return f"{v/1e3:.2f}K"
    else:
        return f"{v:.2f}"

def main():
    conn = connect()

    print("=" * 80)
    print("CRYPTO TIER CLASSIFICATION ANALYSIS")
    print("=" * 80)

    eligible = get_eligible_symbols(conn)
    print(f"\nTotal symbols with >= {MIN_CANDLES} 1h candles: {len(eligible)}")

    results = {}
    print("\nLoading and computing metrics for each symbol...")

    for i, (symbol, cnt) in enumerate(eligible):
        rows = get_symbol_data(conn, symbol)
        if len(rows) < MIN_CANDLES:
            continue
        metrics = analyze_symbol(rows)
        results[symbol] = metrics
        if (i + 1) % 20 == 0:
            print(f"  Processed {i+1}/{len(eligible)} symbols...")

    print(f"  Done. Computed metrics for {len(results)} symbols.\n")

    # Sort by daily_return_std
    sorted_symbols = sorted(results.keys(), key=lambda s: results[s]['daily_return_std'])

    # Print summary table
    print("=" * 130)
    print(f"{'SYMBOL':<12} {'Candles':>7} {'AvgDailyVol':>13} {'RetStd%':>8} {'ATR%':>7} {'MaxGain%':>9} {'MaxLoss%':>9} {'Pump20':>7} {'Pump40':>7}  {'Date Range'}")
    print("-" * 130)

    for sym in sorted_symbols:
        m = results[sym]
        print(
            f"{sym:<12} {m['candle_count']:>7} {format_vol(m['avg_daily_vol_usdc']):>13} "
            f"{m['daily_return_std']:>8.2f} {m['atr_pct_avg']:>7.2f} "
            f"{m['max_1d_gain_pct']:>9.2f} {m['max_1d_loss_pct']:>9.2f} "
            f"{m['pump_events_20pct']:>7} {m['pump_events_40pct']:>7}  {m['date_range']}"
        )

    print("=" * 130)

    # -------------------------------------------------------------------------
    # Distribution analysis for threshold selection
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("DISTRIBUTION ANALYSIS - daily_return_std percentiles")
    print("=" * 80)

    stds = sorted([results[s]['daily_return_std'] for s in results])
    vols = sorted([results[s]['avg_daily_vol_usdc'] for s in results])

    print(f"\ndaily_return_std distribution (n={len(stds)}):")
    for p in [10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 95]:
        print(f"  P{p:2d}: {percentile(stds, p):.2f}%")

    print(f"\navg_daily_vol_usdc distribution:")
    for p in [10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 95]:
        print(f"  P{p:2d}: {format_vol(percentile(vols, p))}")

    # Natural clusters: look for gaps in std distribution
    print("\nNatural breakpoints in daily_return_std (gaps >= 0.3%):")
    gaps = []
    for i in range(1, len(stds)):
        gap = stds[i] - stds[i-1]
        if gap >= 0.3:
            gaps.append((stds[i-1], stds[i], gap))
    for lo, hi, gap in sorted(gaps, key=lambda x: -x[2])[:10]:
        # Count how many symbols are below lo
        below = sum(1 for s in stds if s <= lo)
        print(f"  Gap: {lo:.2f}% -> {hi:.2f}% (+{gap:.2f}%) | {below} symbols below this gap")

    # -------------------------------------------------------------------------
    # Threshold suggestions
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("TIER CLASSIFICATION SUGGESTIONS")
    print("=" * 80)

    # Try a few threshold combinations
    threshold_scenarios = [
        ("Conservative", 2.5, 5e6),
        ("Moderate",     3.5, 2e6),
        ("Aggressive",   5.0, 1e6),
    ]

    for name, std_thresh, vol_thresh in threshold_scenarios:
        tier1 = [s for s in results if results[s]['daily_return_std'] < std_thresh and results[s]['avg_daily_vol_usdc'] >= vol_thresh]
        tier2 = [s for s in results if s not in tier1]
        print(f"\n[{name}] std < {std_thresh}% AND avg_daily_vol > {format_vol(vol_thresh)}:")
        print(f"  Tier 1 (stable):   {len(tier1):3d} symbols: {', '.join(sorted(tier1))}")
        print(f"  Tier 2 (volatile): {len(tier2):3d} symbols")

    # Recommended (Moderate)
    std_thresh = 3.5
    vol_thresh = 2e6
    tier1 = sorted([s for s in results if results[s]['daily_return_std'] < std_thresh and results[s]['avg_daily_vol_usdc'] >= vol_thresh], key=lambda s: results[s]['daily_return_std'])
    tier2 = sorted([s for s in results if s not in tier1], key=lambda s: results[s]['daily_return_std'])

    print(f"\n{'='*80}")
    print(f"RECOMMENDED THRESHOLDS: std < 3.5% AND avg_daily_vol > 2M USDC")
    print(f"{'='*80}")
    print(f"\nTIER 1 (STABLE) - {len(tier1)} symbols:")
    print(f"{'SYMBOL':<12} {'RetStd%':>8} {'AvgDailyVol':>13} {'ATR%':>7}")
    print("-" * 45)
    for sym in tier1:
        m = results[sym]
        print(f"{sym:<12} {m['daily_return_std']:>8.2f} {format_vol(m['avg_daily_vol_usdc']):>13} {m['atr_pct_avg']:>7.2f}")

    print(f"\nTIER 2 (VOLATILE) - {len(tier2)} symbols (top 30 by std):")
    print(f"{'SYMBOL':<12} {'RetStd%':>8} {'AvgDailyVol':>13} {'ATR%':>7} {'Pump20':>7}")
    print("-" * 52)
    for sym in tier2[:30]:
        m = results[sym]
        print(f"{sym:<12} {m['daily_return_std']:>8.2f} {format_vol(m['avg_daily_vol_usdc']):>13} {m['atr_pct_avg']:>7.2f} {m['pump_events_20pct']:>7}")

    # -------------------------------------------------------------------------
    # Pump pattern analysis
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PUMP PATTERN ANALYSIS (symbols with > 5 pump events at +20%)")
    print("=" * 80)

    pump_candidates = [s for s in results if results[s]['pump_events_20pct'] > 5]
    pump_candidates = sorted(pump_candidates, key=lambda s: -results[s]['pump_events_20pct'])

    print(f"\nFound {len(pump_candidates)} symbols with > 5 pump events (+20% in 24h)\n")
    print(f"{'SYMBOL':<12} {'Pumps20':>8} {'Pumps40':>8} {'AvgRSI-6h':>10} {'VolRatio-3h':>12} {'BB-Squeeze%':>12}")
    print("-" * 70)

    pump_pattern_results = {}
    for sym in pump_candidates:
        rows = get_symbol_data(conn, sym)
        pattern = analyze_pump_patterns(rows, pump_threshold_pct=20.0)
        pump_pattern_results[sym] = pattern

        rsi_str = f"{pattern['avg_rsi_before']:.1f}" if pattern['avg_rsi_before'] is not None else "N/A"
        vr_str = f"{pattern['avg_vol_ratio_before']:.2f}x" if pattern['avg_vol_ratio_before'] is not None else "N/A"
        bb_str = f"{pattern['pct_after_bb_squeeze']:.1f}%"
        print(f"{sym:<12} {pattern['n_pump_events']:>8} {results[sym]['pump_events_40pct']:>8} {rsi_str:>10} {vr_str:>12} {bb_str:>12}")

    # Aggregate pump stats
    if pump_pattern_results:
        all_rsi = [v['avg_rsi_before'] for v in pump_pattern_results.values() if v['avg_rsi_before'] is not None]
        all_vr = [v['avg_vol_ratio_before'] for v in pump_pattern_results.values() if v['avg_vol_ratio_before'] is not None]
        all_bb = [v['pct_after_bb_squeeze'] for v in pump_pattern_results.values()]

        print("\nAGGREGATE PUMP PATTERN STATS (across all pump-prone symbols):")
        print(f"  Avg RSI 6h before pump start:     {mean(all_rsi):.1f}" + (f"  (range: {min(all_rsi):.1f} - {max(all_rsi):.1f})" if all_rsi else ""))
        print(f"  Avg vol ratio 3h before pump:     {mean(all_vr):.2f}x" + (f"  (range: {min(all_vr):.2f}x - {max(all_vr):.2f}x)" if all_vr else ""))
        print(f"  % of pumps after BB squeeze:      {mean(all_bb):.1f}%")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

    conn.close()

if __name__ == '__main__':
    main()
