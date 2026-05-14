from ..binance import get_all_tickers_24h, get_recent_klines
from ..config import QUOTE_CURRENCY
from ..display import console
from ..indicators import (STABLECOINS, atr, bollinger, death_cross_recent, golden_cross_recent,
                          macd, price_cross_ma_recent, rsi, rsi_series, sma, stochastic,
                          ad_line)
from ..storage import get_excluded, init_db


def cmd_scan(args) -> None:
    from rich import box
    from rich.table import Table

    top_n = args.top
    interval = args.interval
    min_volume = args.min_volume

    with console.status("[cyan]Récupération des tickers 24h…[/]"):
        all_tickers = get_all_tickers_24h()

    init_db()
    excluded = get_excluded()

    candidates = [
        {"symbol": t["symbol"].removesuffix(QUOTE_CURRENCY),
         "change_pct": float(t["priceChangePercent"]),
         "quote_vol": float(t["quoteVolume"])}
        for t in all_tickers
        if t["symbol"].endswith(QUOTE_CURRENCY)
        and t["symbol"].removesuffix(QUOTE_CURRENCY) not in STABLECOINS
        and t["symbol"].removesuffix(QUOTE_CURRENCY) not in excluded
        and float(t["quoteVolume"]) >= min_volume
        and float(t["priceChangePercent"]) <= 20
    ]
    candidates.sort(key=lambda x: x["quote_vol"], reverse=True)
    candidates = candidates[:top_n * 4]

    results = []
    with console.status("[cyan]Recherche des débuts de montée…[/]") as status:
        for c in candidates:
            status.update(f"[cyan]Analyse {c['symbol']}…[/]")
            try:
                klines = get_recent_klines(c["symbol"], interval, limit=60)
                closes  = [float(k[4]) for k in klines]
                volumes = [float(k[5]) for k in klines]
                highs   = [float(k[2]) for k in klines]
                lows    = [float(k[3]) for k in klines]
            except Exception:
                continue

            if len(closes) < 50:
                continue

            last = closes[-1]
            rsi_vals  = rsi_series(closes)
            rsi_now   = rsi_vals[-1]
            rsi_prev5 = rsi_vals[-6] if len(rsi_vals) >= 6 else rsi_now
            ma20, ma50 = sma(closes, 20), sma(closes, 50)
            _, _, histogram = macd(closes)
            bb_upper, _, _ = bollinger(closes)
            stoch_k, stoch_d = stochastic(highs, lows, closes)
            stoch_k_prev, stoch_d_prev = stochastic(highs[:-1], lows[:-1], closes[:-1])
            ad = ad_line(highs, lows, closes, volumes)
            vol_avg = sum(volumes[-10:]) / 10 if len(volumes) >= 10 else 0

            golden_cross     = golden_cross_recent(closes, lookback=5)
            price_cross_ma20 = price_cross_ma_recent(closes, 20, lookback=3)
            macd_cross_up    = len(histogram) >= 2 and histogram[-2] < 0 <= histogram[-1]
            stoch_cross_up   = stoch_k_prev < stoch_d_prev and stoch_k > stoch_d and stoch_k < 40
            bb_breakout      = bb_upper is not None and last > bb_upper
            vol_breakout     = vol_avg > 0 and volumes[-1] > vol_avg * 1.5 and bb_breakout
            ad_accum         = len(ad) >= 6 and ad[-1] > ad[-6] and closes[-1] <= closes[-6]
            rsi_rising       = rsi_now > rsi_prev5 + 3
            vol_buildup      = vol_avg > 0 and sum(volumes[-3:]) / 3 > vol_avg * 1.2

            score, signals = 0, []

            if (ma50 and last > ma50) and (40 <= rsi_now <= 62) and histogram and histogram[-1] > 0:
                score += 3; signals.append("Confluence MA/RSI/MACD")
            if golden_cross:       score += 3; signals.append("Golden cross")
            if macd_cross_up:      score += 2; signals.append("MACD ↑")
            if price_cross_ma20:   score += 2; signals.append("Breakout MA20")
            if stoch_cross_up:     score += 2; signals.append(f"Stoch ↑{stoch_k:.0f}")
            if vol_breakout:       score += 2; signals.append(f"BB breakout vol×{volumes[-1]/vol_avg:.1f}")
            if rsi_rising and 38 <= rsi_now <= 58: score += 1; signals.append(f"RSI ↑{rsi_now:.0f}")
            if ad_accum:           score += 1; signals.append("A/D accum")
            if vol_buildup and not bb_breakout: score += 1; signals.append("Vol ↑")
            if 0 <= c["change_pct"] <= 8: score += 1; signals.append(f"+{c['change_pct']:.1f}%")

            if score >= 4:
                results.append({**c, "rsi": rsi_now, "above_ma50": bool(ma50 and last > ma50),
                                 "score": score, "signals": signals})

    results.sort(key=lambda x: x["score"], reverse=True)

    table = Table(title=f"[bold cyan]Débuts de montée[/] — {interval} — Top {top_n}",
                  box=box.ROUNDED, title_justify="left")
    table.add_column("Actif", style="bold", min_width=8)
    table.add_column("Score", justify="center")
    table.add_column(f"Vol {QUOTE_CURRENCY}", justify="right")
    table.add_column("RSI", justify="right")
    table.add_column("24h", justify="right")
    table.add_column("Signaux")

    for r in results[:top_n]:
        score_color  = "green" if r["score"] >= 6 else "yellow" if r["score"] >= 4 else "dim"
        change_color = "green" if r["change_pct"] > 0 else "red"
        table.add_row(r["symbol"], f"[{score_color}]{r['score']}[/]",
                      f"{r['quote_vol']:,.0f}", f"{r['rsi']:.0f}",
                      f"[{change_color}]{r['change_pct']:+.1f}%[/]", "  ".join(r["signals"]))

    console.print(table)
    console.print("[dim]Score ≥6 = setup précoce solide. Golden cross + Breakout MA20 = signal le plus fort.[/]")


def cmd_dip(args) -> None:
    from rich import box
    from rich.table import Table

    top_n = args.top
    interval = args.interval
    min_volume = args.min_volume

    with console.status("[cyan]Récupération des tickers 24h…[/]"):
        all_tickers = get_all_tickers_24h()

    init_db()
    excluded = get_excluded()

    candidates = [
        {"symbol": t["symbol"].removesuffix(QUOTE_CURRENCY),
         "change_pct": float(t["priceChangePercent"]),
         "quote_vol": float(t["quoteVolume"])}
        for t in all_tickers
        if t["symbol"].endswith(QUOTE_CURRENCY)
        and t["symbol"].removesuffix(QUOTE_CURRENCY) not in STABLECOINS
        and t["symbol"].removesuffix(QUOTE_CURRENCY) not in excluded
        and float(t["quoteVolume"]) >= min_volume
        and float(t["priceChangePercent"]) < 0
    ]
    candidates.sort(key=lambda x: x["quote_vol"], reverse=True)
    candidates = candidates[:top_n * 3]

    results = []
    with console.status("[cyan]Analyse des dips…[/]") as status:
        for c in candidates:
            status.update(f"[cyan]Analyse {c['symbol']}…[/]")
            try:
                klines  = get_recent_klines(c["symbol"], interval, limit=60)
                closes  = [float(k[4]) for k in klines]
                volumes = [float(k[5]) for k in klines]
            except Exception:
                continue

            last = closes[-1]
            rsi_val = rsi(closes)
            ma20 = sma(closes, 20)
            ma50 = sma(closes, 50)

            recent_high = max(closes[-30:]) if len(closes) >= 30 else max(closes)
            dip_pct  = (recent_high - last) / recent_high * 100
            vol_avg  = sum(volumes[-10:]) / 10 if len(volumes) >= 10 else 0
            vol_ratio = volumes[-1] / vol_avg if vol_avg > 0 else 1.0
            uptrend  = ma20 is not None and ma50 is not None and ma20 > ma50
            above_ma50 = ma50 is not None and last > ma50

            score, signals = 0, []
            if uptrend:              score += 2; signals.append("Tendance ↑")
            if above_ma50:           score += 1; signals.append("↑MA50")
            if 5 <= dip_pct <= 25:   score += 2; signals.append(f"Dip -{dip_pct:.1f}%")
            elif 25 < dip_pct <= 40: score += 1; signals.append(f"Dip -{dip_pct:.1f}% (fort)")
            if 25 <= rsi_val <= 42:  score += 2; signals.append(f"RSI {rsi_val:.0f}")
            elif 42 < rsi_val <= 50: score += 1; signals.append(f"RSI {rsi_val:.0f}")
            if vol_ratio < 0.8:      score += 1; signals.append("Vol faible")

            results.append({**c, "rsi": rsi_val, "dip_pct": dip_pct, "uptrend": uptrend,
                             "above_ma50": above_ma50, "vol_ratio": vol_ratio,
                             "score": score, "signals": signals})

    results.sort(key=lambda x: x["score"], reverse=True)

    table = Table(title=f"[bold cyan]Buy the Dip[/] — {interval} — Top {top_n} candidats",
                  box=box.ROUNDED, title_justify="left")
    table.add_column("Actif", style="bold", min_width=8)
    table.add_column("Score", justify="center")
    table.add_column(f"Vol {QUOTE_CURRENCY}", justify="right")
    table.add_column("RSI", justify="right")
    table.add_column("Dip", justify="right")
    table.add_column("Tendance", justify="center")
    table.add_column("↑MA50", justify="center")
    table.add_column("Signaux")

    for r in results[:top_n]:
        score_color = "green" if r["score"] >= 6 else "yellow" if r["score"] >= 4 else "dim"
        table.add_row(r["symbol"], f"[{score_color}]{r['score']}[/]",
                      f"{r['quote_vol']:,.0f}", f"{r['rsi']:.0f}", f"-{r['dip_pct']:.1f}%",
                      "[green]↑[/]" if r["uptrend"] else "[red]↓[/]",
                      "[green]✓[/]" if r["above_ma50"] else "[red]✗[/]",
                      "  ".join(r["signals"]))

    console.print(table)
    console.print("[dim]Score ≥6 = dip dans une tendance haussière avec RSI en survente — à confirmer avant d'acheter.[/]")


def register(sub):
    p = sub.add_parser("scan", help="Scanner les paires Binance en début de progression")
    p.add_argument("--top", type=int, default=15)
    p.add_argument("--interval", default="1h")
    p.add_argument("--min-volume", type=float, default=50_000, dest="min_volume",
                   metavar="USDC")

    p = sub.add_parser("dip", help="Scanner les paires Binance en dip dans une tendance haussière")
    p.add_argument("--top", type=int, default=15)
    p.add_argument("--interval", default="1h")
    p.add_argument("--min-volume", type=float, default=50_000, dest="min_volume",
                   metavar="USDC")

    return {"scan": cmd_scan, "dip": cmd_dip}
