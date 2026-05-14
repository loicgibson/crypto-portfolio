"""
ML commands: ml-fetch, ml-train, ml-scan, ml-analyze.

ml-fetch   — download 1h (5yr) + 4h (5yr) + 15m (2yr) klines for all liquid USDC pairs
ml-train   — train one model per symbol using 1h+4h+15m data (LGBM/RF/LR, best AP)
ml-scan    — scanner with tech + ML probability columns
ml-analyze — portfolio analysis with tech + ML probability columns
"""
import sys
import time
from datetime import datetime, timezone

from ..binance import (get_all_tickers_24h, get_earn_aprs, get_funding_rates,
                       get_klines, get_recent_klines, get_usdc_pairs_by_status)
from ..config import ML_INTERVAL, QUOTE_CURRENCY
from ..display import console
from ..indicators import (STABLECOINS, bollinger, death_cross_recent,
                           golden_cross_recent, macd, price_cross_ma_recent,
                           rsi_series, sma, stochastic, ad_line, atr, rsi)
from ..portfolio import get_portfolio
from ..storage import (app_set_state, get_excluded, get_last_funding_time,
                       get_last_kline_time, init_db, set_inactive_symbols,
                       set_trading_symbols, upsert_funding_rates, upsert_klines)


# ── Shared helpers ────────────────────────────────────────────────────────────

def _tech_score_scan(klines: list) -> tuple[int, list[str]]:
    """Return (score, signals) for scan entry logic (same as commands/scanner.py)."""
    closes  = [float(k[4]) for k in klines]
    volumes = [float(k[5]) for k in klines]
    highs   = [float(k[2]) for k in klines]
    lows    = [float(k[3]) for k in klines]

    if len(closes) < 50:
        return 0, []

    last = closes[-1]
    rsi_vals  = rsi_series(closes)
    rsi_now   = rsi_vals[-1]
    rsi_prev5 = rsi_vals[-6] if len(rsi_vals) >= 6 else rsi_now
    ma20, ma50    = sma(closes, 20), sma(closes, 50)
    _, _, histogram = macd(closes)
    bb_upper, _, _ = bollinger(closes)
    stoch_k, stoch_d = stochastic(highs, lows, closes)
    stoch_k_p, stoch_d_p = stochastic(highs[:-1], lows[:-1], closes[:-1])
    ad = ad_line(highs, lows, closes, volumes)
    vol_avg = sum(volumes[-10:]) / 10 if len(volumes) >= 10 else 0

    golden_cross     = golden_cross_recent(closes)
    price_cross_ma20 = price_cross_ma_recent(closes, 20)
    macd_cross_up    = len(histogram) >= 2 and histogram[-2] < 0 <= histogram[-1]
    stoch_cross_up   = stoch_k_p < stoch_d_p and stoch_k > stoch_d and stoch_k < 40
    bb_breakout      = bb_upper is not None and last > bb_upper
    vol_breakout     = vol_avg > 0 and volumes[-1] > vol_avg * 1.5 and bb_breakout
    ad_accum         = len(ad) >= 6 and ad[-1] > ad[-6] and closes[-1] <= closes[-6]
    rsi_rising       = rsi_now > rsi_prev5 + 3
    vol_buildup      = vol_avg > 0 and sum(volumes[-3:]) / 3 > vol_avg * 1.2

    score, signals = 0, []
    if (ma50 and last > ma50) and (40 <= rsi_now <= 62) and histogram and histogram[-1] > 0:
        score += 3; signals.append("Conf MA/RSI/MACD")
    if golden_cross:     score += 3; signals.append("Golden cross")
    if macd_cross_up:    score += 2; signals.append("MACD ↑")
    if price_cross_ma20: score += 2; signals.append("Breakout MA20")
    if stoch_cross_up:   score += 2; signals.append(f"Stoch ↑{stoch_k:.0f}")
    if vol_breakout and vol_avg: score += 2; signals.append(f"BB break ×{volumes[-1]/vol_avg:.1f}")
    if rsi_rising and 38 <= rsi_now <= 58: score += 1; signals.append(f"RSI ↑{rsi_now:.0f}")
    if ad_accum:         score += 1; signals.append("A/D accum")
    if vol_buildup and not bb_breakout: score += 1; signals.append("Vol ↑")
    return score, signals


def _tech_score_analyze(klines: list, avg_buy_price: float) -> tuple[int, list[str], float]:
    """Return (score, signals, rsi_val) for analyze exit logic."""
    closes  = [float(k[4]) for k in klines]
    volumes = [float(k[5]) for k in klines]
    highs   = [float(k[2]) for k in klines]
    lows    = [float(k[3]) for k in klines]

    if len(closes) < 26:
        return 0, [], 50.0

    last = closes[-1]
    rsi_vals = rsi_series(closes)
    rsi_val  = rsi_vals[-1]
    ma20, ma50 = sma(closes, 20), sma(closes, 50)
    _, _, histogram = macd(closes)
    bb_upper, _, _ = bollinger(closes)
    stoch_k, stoch_d = stochastic(highs, lows, closes)
    stoch_k_p, stoch_d_p = stochastic(highs[:-1], lows[:-1], closes[:-1])
    atr_val = atr(highs, lows, closes)
    ad = ad_line(highs, lows, closes, volumes)
    vol_avg   = sum(volumes[-10:]) / 10 if len(volumes) >= 10 else 0
    vol_last3 = sum(volumes[-3:]) / 3 if len(volumes) >= 3 else 0

    death_cross      = death_cross_recent(closes)
    macd_cross_down  = len(histogram) >= 2 and histogram[-2] > 0 >= histogram[-1]
    stoch_cross_down = stoch_k_p > stoch_d_p and stoch_k < stoch_d and stoch_k > 70
    above_ma20  = ma20 is not None and last > ma20
    above_ma50  = ma50 is not None and last > ma50
    at_bb_upper = bb_upper is not None and last > bb_upper
    ad_distrib  = len(ad) >= 6 and ad[-1] < ad[-6] and closes[-1] >= closes[-6]
    vol_drying  = vol_last3 < vol_avg * 0.7 and vol_avg > 0 and last > (closes[-4] if len(closes) >= 4 else last)
    atr_stop    = avg_buy_price > 0 and last < avg_buy_price - 2 * atr_val

    score, signals = 0, []
    if rsi_val > 80:       score += 3; signals.append(f"[red]RSI {rsi_val:.0f}[/]")
    elif rsi_val > 70:     score += 2; signals.append(f"[yellow]RSI {rsi_val:.0f}[/]")
    if death_cross:        score += 3; signals.append("[red]Death cross[/]")
    if macd_cross_down:    score += 2; signals.append("[red]MACD ↓[/]")
    if stoch_cross_down:   score += 2; signals.append(f"[red]Stoch ↓{stoch_k:.0f}[/]")
    if ad_distrib:         score += 2; signals.append("[red]A/D distrib[/]")
    if at_bb_upper:        score += 1; signals.append("[yellow]BB sup[/]")
    if atr_stop:           score += 2; signals.append("[red]Stop ATR[/]")
    if not above_ma50:     score += 2; signals.append("[red]<MA50[/]")
    elif not above_ma20:   score += 1; signals.append("[yellow]<MA20[/]")
    if vol_drying:         score += 1; signals.append("[yellow]Vol tarit[/]")
    return score, signals, rsi_val


_AP_FLOOR = 0.15   # AP d'un classifieur aléatoire (≈ taux de positifs)
_AP_CEIL  = 0.40   # AP à partir duquel on fait pleinement confiance au modèle


def _ap_confidence(ap: float | None) -> float:
    """Confidence weight in [0, 1] based on model quality (Average Precision)."""
    if ap is None:
        return 0.0
    return max(0.0, min(1.0, (ap - _AP_FLOOR) / (_AP_CEIL - _AP_FLOOR)))


def _combined(tech_score: float, ml_prob: float | None, ap: float | None = None,
              tech_weight: float = 0.4) -> float | None:
    """
    Combined score where ML weight is earned by AP quality.
    When AP is low, ML weight is redistributed to tech — the signal itself is
    never altered (no blending toward 0.5).
    """
    if ml_prob is None:
        return None
    tech_norm = min(tech_score / 10.0, 1.0)
    ap_conf   = _ap_confidence(ap)
    ml_w      = (1.0 - tech_weight) * ap_conf   # 0 when AP=floor, 0.6 when AP=ceil
    tech_w    = 1.0 - ml_w                       # tech picks up the slack
    return round(tech_w * tech_norm + ml_w * ml_prob, 4)


# ── ml-fetch ─────────────────────────────────────────────────────────────────

def _fetch_interval(symbols: list, interval: str, years: float,
                    total: int) -> tuple[int, int, int]:
    """Download klines for one interval. Returns (ok, skipped, errors)."""
    start_default_ms = int((time.time() - years * 365 * 86400) * 1000)
    now_ms = int(time.time() * 1000)
    ok, skipped, errors = 0, 0, 0

    for i, sym in enumerate(symbols, 1):
        last     = get_last_kline_time(sym, interval)
        start_ms = last + 1 if last else start_default_ms

        if start_ms >= now_ms - 60_000:
            skipped += 1
            continue

        inserted_total = 0
        try:
            cur = start_ms
            while cur < now_ms:
                rows = get_klines(sym, interval, cur)
                if not rows:
                    break
                inserted_total += upsert_klines(sym, interval, rows)
                cur = int(rows[-1][6]) + 1
                if len(rows) < 1000:
                    break
            console.print(
                f"[green][{interval}][{i}/{total}] {sym} — +{inserted_total}[/]"
            )
            ok += 1
        except Exception as e:
            console.print(f"[red][{interval}][{i}/{total}] {sym} — {e}[/]")
            errors += 1

    return ok, skipped, errors


def cmd_ml_fetch(args) -> None:
    init_db()
    since_years  = args.years
    years_15m    = min(args.years_15m, args.years)

    with console.status("[cyan]Récupération des paires USDC (exchangeInfo)…[/]"):
        all_pairs, inactive = get_usdc_pairs_by_status()

    now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
    set_trading_symbols(set(all_pairs), now_iso)
    set_inactive_symbols(inactive, now_iso)
    app_set_state("exchange_info_updated_at", now_iso)
    console.print(f"[dim]Statut de trading : {len(all_pairs)} actifs, "
                  f"{len(inactive)} inactifs enregistrés.[/]")

    excluded = get_excluded()
    symbols  = sorted([
        sym for sym in all_pairs
        if sym not in STABLECOINS and sym not in excluded
    ])
    total = len(symbols)

    console.print(
        f"[bold]{total} paires USDC[/] — "
        f"1h+4h ({since_years} ans), 15m ({years_15m} ans)\n"
    )

    for interval, yrs in [("1h", since_years), ("4h", since_years), ("15m", years_15m)]:
        console.print(f"[bold cyan]━ Intervalle {interval} ({yrs} ans) ━[/]")
        ok, skipped, errors = _fetch_interval(symbols, interval, yrs, total)
        console.print(
            f"[bold]{interval}[/] — {ok} maj, {skipped} déjà à jour, {errors} erreurs.\n"
        )

    # ── Funding rates (perpetual futures, USDT-margined) ─────────────────────
    console.print("[bold]Mise à jour des funding rates futures…[/]\n")
    fr_ok, fr_skip, fr_none = 0, 0, 0
    for i, sym in enumerate(symbols, 1):
        last_ft  = get_last_funding_time(sym)
        start_ms = last_ft + 1 if last_ft else None

        all_rates: list[dict] = []
        cursor = start_ms
        while True:
            batch = get_funding_rates(sym, cursor)
            if not batch:
                break
            all_rates.extend(batch)
            if len(batch) < 1000:
                break
            cursor = int(batch[-1]["fundingTime"]) + 1
            time.sleep(0.05)

        if not all_rates:
            fr_none += 1
            continue

        inserted = upsert_funding_rates(sym, all_rates)
        if inserted > 0:
            console.print(f"[green][{i}/{total}] {sym} funding — +{inserted}[/]")
            fr_ok += 1
        else:
            fr_skip += 1
        time.sleep(0.05)

    console.print(f"\n[bold]Funding rates[/] — {fr_ok} maj, {fr_skip} déjà à jour, "
                  f"{fr_none} sans futures.")


# ── ml-train ─────────────────────────────────────────────────────────────────

def cmd_ml_train(args) -> None:
    from rich import box
    from rich.table import Table

    from ..ml.trainer import train_all, train_symbol

    init_db()
    interval = args.interval

    if args.symbol:
        symbols = [s.upper() for s in args.symbol]
        console.print(f"[bold]Entraînement pour : {', '.join(symbols)} [{interval}][/]\n")
    else:
        import sqlite3
        from ..config import DB_PATH
        with sqlite3.connect(DB_PATH) as conn:
            rows = conn.execute(
                "SELECT DISTINCT symbol, COUNT(*) as n FROM klines "
                "WHERE interval=? GROUP BY symbol HAVING n >= 500",
                (interval,),
            ).fetchall()
        symbols = [r[0] for r in rows]
        console.print(f"[bold]{len(symbols)} symboles éligibles [{interval}][/]\n")

    results = []
    with console.status("[cyan]Entraînement…[/]") as status:
        for i, sym in enumerate(symbols, 1):
            status.update(f"[cyan][{i}/{len(symbols)}] {sym}…[/]")
            r = train_symbol(sym, interval, progress_cb=lambda msg: status.update(msg))
            results.append(r)

    ok     = [r for r in results if "error" not in r]
    failed = [r for r in results if "error" in r]

    if ok:
        table = Table(title="[bold green]Résultats d'entraînement[/]",
                      box=box.ROUNDED, title_justify="left")
        table.add_column("Symbole", style="bold")
        table.add_column("Modèle",  justify="center")
        table.add_column("TF",      justify="center")
        table.add_column("CV AP",   justify="right")
        table.add_column("Test AP", justify="right")
        table.add_column("AUC",     justify="right")
        table.add_column("Prec@50", justify="right")
        table.add_column("Prec@65", justify="right")
        table.add_column("N train", justify="right")
        table.add_column("N test",  justify="right")

        for r in sorted(ok, key=lambda x: x["metrics"]["ap"], reverse=True):
            m   = r["metrics"]
            cv  = max(r["cv_scores"].values())
            c   = "green" if m["ap"] >= 0.4 else "yellow" if m["ap"] >= 0.3 else "red"
            tf  = "1h"
            if r.get("has_4h"):  tf += "+4h"
            if r.get("has_15m"): tf += "+15m"
            table.add_row(
                r["symbol"], r["best_model"], tf,
                f"{cv:.3f}", f"[{c}]{m['ap']:.3f}[/]",
                f"{m['auc']:.3f}", f"{m['precision_50']:.3f}",
                f"{m['precision_65']:.3f}",
                str(r["n_train"]), str(r["n_test"]),
            )
        console.print(table)

    if failed:
        console.print(f"\n[red]{len(failed)} échec(s) :[/]")
        for r in failed:
            console.print(f"  [dim]{r['symbol']} : {r['error']}[/]")

    console.print(f"\n[bold]{len(ok)} modèle(s) sauvegardé(s)[/] — "
                  "target : max_gain_4h ≥ 5 %, métrique : Average Precision")
    console.print("[dim]AP ≥ 0.4 = bon signal. TF = granularités utilisées (1h toujours, "
                  "4h/15m si disponibles). Prec@65 = précision quand prob ≥ 0.65.[/]")


# ── ml-scan ──────────────────────────────────────────────────────────────────

def cmd_ml_scan(args) -> None:
    from rich import box
    from rich.table import Table

    from ..ml.predictor import predict_symbol

    init_db()
    excluded   = get_excluded()
    interval   = args.interval
    top_n      = args.top
    min_volume = args.min_volume
    ml_interval = args.ml_interval or ML_INTERVAL

    with console.status("[cyan]Récupération des tickers 24h…[/]"):
        tickers = get_all_tickers_24h()

    with console.status("[cyan]Récupération APR Earn…[/]"):
        earn_aprs = get_earn_aprs()   # {} if no API keys / endpoint fails

    candidates = [
        {"symbol": t["symbol"].removesuffix(QUOTE_CURRENCY),
         "change_pct": float(t["priceChangePercent"]),
         "quote_vol": float(t["quoteVolume"])}
        for t in tickers
        if t["symbol"].endswith(QUOTE_CURRENCY)
        and t["symbol"].removesuffix(QUOTE_CURRENCY) not in STABLECOINS
        and t["symbol"].removesuffix(QUOTE_CURRENCY) not in excluded
        and float(t["quoteVolume"]) >= min_volume
        and float(t["priceChangePercent"]) <= 20
    ]
    candidates.sort(key=lambda x: x["quote_vol"], reverse=True)

    results = []
    with console.status("[cyan]Scan + ML…[/]") as status:
        for c in candidates[:args.pool]:
            sym = c["symbol"]
            status.update(f"[cyan]Analyse {sym}…[/]")
            try:
                klines = get_recent_klines(sym, interval, limit=60)
            except Exception:
                continue

            tech_score, signals = _tech_score_scan(klines)

            # ML prediction
            ml = predict_symbol(sym, ml_interval)
            ml_prob = ml.get("ml_prob")

            ml_ap    = ml.get("ap")
            combined = _combined(tech_score, ml_prob, ml_ap, args.tech_weight)

            # Filter: must meet at least one threshold
            if tech_score < args.min_tech and (ml_prob is None or ml_prob < args.min_ml):
                continue

            results.append({
                **c,
                "tech_score": tech_score,
                "signals":    signals,
                "ml_prob":    ml_prob,
                "ml_model":   ml.get("model_name"),
                "ml_ap":      ml_ap,
                "combined":   combined,
                "earn_apr":   earn_aprs.get(sym),
            })

    # Sort: combined first, fallback to tech_score for symbols without model
    results.sort(
        key=lambda x: (x["combined"] if x["combined"] is not None else x["tech_score"] / 10),
        reverse=True,
    )

    table = Table(
        title=f"[bold cyan]ML-Scan[/] — {interval} (tech) + {ml_interval} (ML) — Top {top_n}",
        box=box.ROUNDED, title_justify="left",
    )
    table.add_column("Actif",    style="bold", min_width=7)
    table.add_column("Tech",     justify="center")
    table.add_column("ML prob",  justify="center")
    table.add_column("Combined", justify="center")
    table.add_column("Earn APR", justify="right")
    table.add_column("24h",      justify="right")
    table.add_column("Vol USDC", justify="right")
    table.add_column("Signaux tech")

    for r in results[:top_n]:
        tc   = "green" if r["tech_score"] >= 6 else "yellow" if r["tech_score"] >= 4 else "white"
        chg  = "green" if r["change_pct"] > 0 else "red"

        if r["ml_prob"] is not None:
            ap     = r["ml_ap"]
            conf   = _ap_confidence(ap)
            # colour by raw prob; dim when model is not yet trustworthy
            mc     = "green" if r["ml_prob"] >= 0.6 else "yellow" if r["ml_prob"] >= 0.45 else "red"
            ap_str = f" AP={ap:.2f} w={conf:.0%}" if ap is not None else ""
            ml_cell = f"[{mc}]{r['ml_prob']:.2f}[/][dim]{ap_str}[/]"
        else:
            ml_cell = "[dim]N/A[/]"

        if r["combined"] is not None:
            cc = "green" if r["combined"] >= 0.6 else "yellow" if r["combined"] >= 0.45 else "white"
            comb_cell = f"[{cc}]{r['combined']:.2f}[/]"
        else:
            cc = tc
            comb_cell = f"[{cc}]{min(r['tech_score']/10, 1.0):.2f}[/][dim]*[/]"

        earn = r.get("earn_apr")
        if earn is not None and earn > 0:
            ec = "green" if earn >= 10 else "yellow" if earn >= 5 else "white"
            earn_cell = f"[{ec}]{earn:.1f}%[/]"
        else:
            earn_cell = "[dim]—[/]"

        table.add_row(
            r["symbol"],
            f"[{tc}]{r['tech_score']}[/]",
            ml_cell,
            comb_cell,
            earn_cell,
            f"[{chg}]{r['change_pct']:+.1f}%[/]",
            f"{r['quote_vol']:,.0f}",
            "  ".join(r["signals"]),
        )

    console.print(table)
    console.print(
        "[dim]Tech = score indicateurs techniques (0-12+). "
        "ML prob = signal brut du modèle (w = poids ML dans combined, basé sur AP). "
        f"Combined = (1-w)×tech + w×ML. w=0% si AP≤{_AP_FLOOR}, w=60% si AP≥{_AP_CEIL}. "
        "Earn APR = taux annuel Simple Earn flexible (vert ≥10%, jaune ≥5%). "
        "[*] = pas de modèle ML.[/]"
    )


# ── ml-analyze ───────────────────────────────────────────────────────────────

def cmd_ml_analyze(args) -> None:
    from rich import box
    from rich.table import Table

    from ..ml.predictor import predict_symbol

    init_db()
    holdings = get_portfolio()
    if not holdings:
        console.print("[yellow]Aucune position dans le portefeuille.[/]")
        return

    interval    = args.interval
    ml_interval = args.ml_interval or ML_INTERVAL

    results = []
    with console.status("[cyan]Analyse ML du portefeuille…[/]") as status:
        for h in holdings:
            sym = h.holding.symbol
            status.update(f"[cyan]Analyse {sym}…[/]")
            try:
                klines = get_recent_klines(sym, interval, limit=60)
            except Exception:
                continue

            tech_score, signals, rsi_val = _tech_score_analyze(
                klines, h.holding.avg_buy_price
            )

            ml = predict_symbol(sym, ml_interval)
            ml_prob = ml.get("ml_prob")
            ml_ap   = ml.get("ap")

            # Invert ML prob for exit signal: low P(up) → high exit signal
            ml_exit = (1.0 - ml_prob) if ml_prob is not None else None
            combined_exit = _combined(tech_score, ml_exit, ml_ap, args.tech_weight)

            if tech_score >= 5 or (combined_exit is not None and combined_exit >= 0.6):
                reco = "[red]SORTIR[/]"
            elif tech_score >= 3 or (combined_exit is not None and combined_exit >= 0.45):
                reco = "[yellow]SURVEILLER[/]"
            else:
                reco = "[green]TENIR[/]"

            results.append({
                "symbol":        sym,
                "pnl_pct":       h.pnl_pct,
                "current_value": h.current_value,
                "rsi":           rsi_val,
                "tech_score":    tech_score,
                "signals":       signals,
                "ml_prob":       ml_prob,
                "ml_exit":       ml_exit,
                "combined":      combined_exit,
                "ml_model":      ml.get("model_name"),
                "ml_ap":         ml_ap,
                "reco":          reco,
            })

    results.sort(
        key=lambda x: (x["combined"] if x["combined"] is not None else x["tech_score"] / 10),
        reverse=True,
    )

    table = Table(
        title=f"[bold cyan]ML-Analyze[/] — {interval} (tech) + {ml_interval} (ML)",
        box=box.ROUNDED, title_justify="left",
    )
    table.add_column("Actif",     style="bold")
    table.add_column("Valeur",    justify="right")
    table.add_column("P&L",       justify="right")
    table.add_column("RSI",       justify="right")
    table.add_column("Tech exit", justify="center")
    table.add_column("ML prob ↑", justify="center")
    table.add_column("Combined",  justify="center")
    table.add_column("Signaux tech")
    table.add_column("Reco",      justify="center")

    for r in results:
        pnl_c  = "green" if r["pnl_pct"] >= 0 else "red"
        tc     = "red" if r["tech_score"] >= 5 else "yellow" if r["tech_score"] >= 3 else "green"

        if r["ml_prob"] is not None:
            ap     = r["ml_ap"]
            conf   = _ap_confidence(ap)
            mc     = "green" if r["ml_prob"] >= 0.55 else "yellow" if r["ml_prob"] >= 0.4 else "red"
            ap_str = f" AP={ap:.2f} w={conf:.0%}" if ap is not None else ""
            ml_cell = f"[{mc}]{r['ml_prob']:.2f}[/][dim]{ap_str}[/]"
        else:
            ml_cell = "[dim]N/A[/]"

        if r["combined"] is not None:
            cc = "red" if r["combined"] >= 0.6 else "yellow" if r["combined"] >= 0.45 else "green"
            comb_cell = f"[{cc}]{r['combined']:.2f}[/]"
        else:
            cc = tc
            comb_cell = f"[{cc}]{min(r['tech_score']/10, 1.0):.2f}[/][dim]*[/]"

        table.add_row(
            r["symbol"],
            f"{r['current_value']:,.2f}",
            f"[{pnl_c}]{r['pnl_pct']:+.1f}%[/]",
            f"{r['rsi']:.0f}",
            f"[{tc}]{r['tech_score']}[/]",
            ml_cell,
            comb_cell,
            "  ".join(r["signals"]),
            r["reco"],
        )

    console.print(table)
    console.print(
        "[dim]Tech exit = score de sortie (↑ = vendre). "
        "ML prob ↑ = brut→effectif ajusté AP (↑ = garder). "
        f"Combined = signal de sortie fusionné (AP∈[{_AP_FLOOR},{_AP_CEIL}]). "
        "[*] = pas de modèle ML.[/]"
    )


# ── register ──────────────────────────────────────────────────────────────────

def register(sub):
    p = sub.add_parser("ml-fetch",
                       help="Télécharger 1h+4h (5 ans) + 15m (2 ans) pour toutes les paires USDC")
    p.add_argument("--min-volume", type=float, default=1_000_000, dest="min_volume",
                   metavar="USDC", help="Volume 24h minimum (défaut: 1 000 000)")
    p.add_argument("--years",      type=float, default=5.0,
                   help="Profondeur historique 1h et 4h en années (défaut: 5)")
    p.add_argument("--years-15m",  type=float, default=2.0, dest="years_15m",
                   help="Profondeur historique 15m en années (défaut: 2)")

    p = sub.add_parser("ml-train",
                       help="Entraîner un modèle ML par crypto (LGBM/RF/LR, best Average Precision)")
    p.add_argument("--symbol", nargs="+", metavar="SYM",
                   help="Symboles à entraîner (défaut: tous les symboles disponibles)")
    p.add_argument("--interval", default="1h")

    p = sub.add_parser("ml-scan",
                       help="Scanner les opportunités d'entrée avec scores technique + ML")
    p.add_argument("--top",         type=int,   default=20)
    p.add_argument("--interval",    default="1h",  help="Intervalle klines pour le score tech")
    p.add_argument("--ml-interval", default=None,  dest="ml_interval",
                   help="Intervalle des modèles ML (défaut: ML_INTERVAL depuis config)")
    p.add_argument("--min-volume",  type=float, default=500_000, dest="min_volume")
    p.add_argument("--pool",        type=int,   default=100,
                   help="Nombre de candidats à analyser avant filtrage (défaut: 100)")
    p.add_argument("--min-tech",    type=int,   default=3,  dest="min_tech",
                   help="Score tech minimum pour apparaître (défaut: 3)")
    p.add_argument("--min-ml",      type=float, default=0.45, dest="min_ml",
                   help="Probabilité ML minimum pour apparaître (défaut: 0.45)")
    p.add_argument("--tech-weight", type=float, default=0.4,  dest="tech_weight",
                   help="Poids du score technique dans le combined (défaut: 0.4)")

    p = sub.add_parser("ml-analyze",
                       help="Analyser le portefeuille avec scores technique + ML")
    p.add_argument("--interval",    default="1h")
    p.add_argument("--ml-interval", default=None, dest="ml_interval")
    p.add_argument("--tech-weight", type=float, default=0.4, dest="tech_weight")

    return {
        "ml-fetch":   cmd_ml_fetch,
        "ml-train":   cmd_ml_train,
        "ml-scan":    cmd_ml_scan,
        "ml-analyze": cmd_ml_analyze,
    }
