import sys

from ..binance import get_klines, get_recent_klines
from ..config import QUOTE_CURRENCY
from ..display import console
from ..indicators import (atr, bollinger, death_cross_recent, macd, rsi_series,
                          sma, stochastic, ad_line)
from ..portfolio import get_portfolio
from ..storage import (add_excluded, get_excluded, get_inactive_symbols,
                       get_last_kline_time, get_trading_symbols, init_db,
                       remove_excluded, upsert_klines)


def cmd_analyze(args) -> None:
    from rich import box
    from rich.table import Table

    init_db()
    holdings = get_portfolio()
    if not holdings:
        console.print("[yellow]Aucune position dans le portefeuille.[/]")
        return

    interval = args.interval
    results = []

    with console.status("[cyan]Analyse des positions…[/]") as status:
        for h in holdings:
            symbol = h.holding.symbol
            status.update(f"[cyan]Analyse {symbol}…[/]")
            try:
                klines  = get_recent_klines(symbol, interval, limit=60)
                closes  = [float(k[4]) for k in klines]
                volumes = [float(k[5]) for k in klines]
                highs   = [float(k[2]) for k in klines]
                lows    = [float(k[3]) for k in klines]
            except Exception:
                continue

            if len(closes) < 26:
                continue

            last    = closes[-1]
            pnl_pct = h.pnl_pct

            rsi_vals   = rsi_series(closes)
            rsi_val    = rsi_vals[-1]
            ma20, ma50 = sma(closes, 20), sma(closes, 50)
            _, _, histogram = macd(closes)
            bb_upper, _, _ = bollinger(closes)
            stoch_k, stoch_d = stochastic(highs, lows, closes)
            stoch_k_prev, stoch_d_prev = stochastic(highs[:-1], lows[:-1], closes[:-1])
            atr_val = atr(highs, lows, closes)
            ad = ad_line(highs, lows, closes, volumes)
            vol_avg   = sum(volumes[-10:]) / 10 if len(volumes) >= 10 else 0
            vol_last3 = sum(volumes[-3:]) / 3 if len(volumes) >= 3 else 0

            death_cross      = death_cross_recent(closes, lookback=5)
            macd_cross_down  = len(histogram) >= 2 and histogram[-2] > 0 >= histogram[-1]
            stoch_cross_down = stoch_k_prev > stoch_d_prev and stoch_k < stoch_d and stoch_k > 70
            above_ma20  = ma20 is not None and last > ma20
            above_ma50  = ma50 is not None and last > ma50
            at_bb_upper = bb_upper is not None and last > bb_upper
            ad_distrib  = len(ad) >= 6 and ad[-1] < ad[-6] and closes[-1] >= closes[-6]
            vol_drying  = vol_last3 < vol_avg * 0.7 and last > (closes[-4] if len(closes) >= 4 else last)
            atr_stop    = h.holding.avg_buy_price > 0 and last < h.holding.avg_buy_price - 2 * atr_val

            score, signals = 0, []
            if rsi_val > 80:   score += 3; signals.append(f"[red]RSI {rsi_val:.0f} surachat fort[/]")
            elif rsi_val > 70: score += 2; signals.append(f"[yellow]RSI {rsi_val:.0f} surachat[/]")
            if death_cross:      score += 3; signals.append("[red]Death cross[/]")
            if macd_cross_down:  score += 2; signals.append("[red]MACD ↓[/]")
            if stoch_cross_down: score += 2; signals.append(f"[red]Stoch ↓{stoch_k:.0f}[/]")
            if ad_distrib:       score += 2; signals.append("[red]A/D distribution[/]")
            if at_bb_upper:      score += 1; signals.append("[yellow]BB supérieure[/]")
            if atr_stop:         score += 2; signals.append("[red]Stop ATR[/]")
            if not above_ma50:   score += 2; signals.append("[red]Sous MA50[/]")
            elif not above_ma20: score += 1; signals.append("[yellow]Sous MA20[/]")
            if vol_drying:       score += 1; signals.append("[yellow]Vol. tarit[/]")
            if pnl_pct >= 100:   score += 2; signals.append(f"[green]P&L +{pnl_pct:.0f}%[/]")
            elif pnl_pct >= 50:  score += 1; signals.append(f"[green]P&L +{pnl_pct:.0f}%[/]")

            if score >= 5:   reco = "[red]SORTIR[/]"
            elif score >= 3: reco = "[yellow]SURVEILLER[/]"
            else:            reco = "[green]TENIR[/]"

            results.append({"symbol": symbol, "pnl_pct": pnl_pct, "current_value": h.current_value,
                             "rsi": rsi_val, "above_ma20": above_ma20, "above_ma50": above_ma50,
                             "score": score, "signals": signals, "reco": reco})

    results.sort(key=lambda x: x["score"], reverse=True)

    table = Table(title=f"[bold cyan]Analyse du portefeuille[/] — {interval}",
                  box=box.ROUNDED, title_justify="left")
    table.add_column("Actif", style="bold", min_width=8)
    table.add_column(f"Valeur {QUOTE_CURRENCY}", justify="right")
    table.add_column("P&L %", justify="right")
    table.add_column("RSI", justify="right")
    table.add_column("↑MA20", justify="center")
    table.add_column("↑MA50", justify="center")
    table.add_column("Score", justify="center")
    table.add_column("Signaux")
    table.add_column("Reco", justify="center")

    for r in results:
        pnl_color   = "green" if r["pnl_pct"] >= 0 else "red"
        score_color = "red" if r["score"] >= 5 else "yellow" if r["score"] >= 3 else "green"
        table.add_row(r["symbol"], f"{r['current_value']:,.2f}",
                      f"[{pnl_color}]{r['pnl_pct']:+.1f}%[/]", f"{r['rsi']:.0f}",
                      "[green]✓[/]" if r["above_ma20"] else "[red]✗[/]",
                      "[green]✓[/]" if r["above_ma50"] else "[red]✗[/]",
                      f"[{score_color}]{r['score']}[/]", "  ".join(r["signals"]), r["reco"])

    console.print(table)
    console.print("[dim]Score ≥5 → SORTIR | ≥3 → SURVEILLER | <3 → TENIR[/]")


def cmd_exclude(args) -> None:
    init_db()
    if args.action == "add":
        add_excluded(args.symbols)
        console.print(f"[green]Exclus : {', '.join(s.upper() for s in args.symbols)}[/]")
    elif args.action == "remove":
        remove_excluded(args.symbols)
        console.print(f"[green]Retirés de la liste : {', '.join(s.upper() for s in args.symbols)}[/]")
    elif args.action == "list":
        excluded = get_excluded()
        if excluded:
            console.print("Exclus manuellement : " + ", ".join(sorted(excluded)))
        else:
            console.print("[dim]Aucun symbole exclu manuellement.[/]")
        inactive = get_inactive_symbols()
        if inactive:
            console.print(f"[dim]Inactifs Binance ({len(inactive)}) : "
                          + ", ".join(sorted(inactive)) + "[/]")
        else:
            console.print("[dim]Aucun symbole inactif en cache.[/]")
        trading = get_trading_symbols()
        console.print(f"[dim]Whitelist : {len(trading)} symboles actifs en cache"
                      + (" (vide — lance ml-fetch ou sim-run pour synchro)" if not trading else "") + "[/]")
        if args.symbols:
            for sym in args.symbols:
                s = sym.upper()
                status = "[green]TRADING[/]" if s in trading else "[red]absent de la whitelist[/]"
                console.print(f"  {s} → {status}")


def cmd_fetch(args) -> None:
    import time
    from datetime import datetime, timezone

    init_db()
    symbol   = args.symbol.upper()
    interval = args.interval

    if args.since:
        start_ms = int(datetime.strptime(args.since, "%Y-%m-%d")
                       .replace(tzinfo=timezone.utc).timestamp() * 1000)
    else:
        last = get_last_kline_time(symbol, interval)
        if last:
            start_ms = last + 1
        else:
            start_ms = int((time.time() - 5 * 365 * 86400) * 1000)

    now_ms = int(time.time() * 1000)
    total_inserted = 0

    with console.status(f"[cyan]Téléchargement des klines {symbol} [{interval}]…[/]") as status:
        while start_ms < now_ms:
            try:
                rows = get_klines(symbol, interval, start_ms)
            except Exception as e:
                console.print(f"[red]Erreur Binance : {e}[/]")
                break
            if not rows:
                break
            inserted = upsert_klines(symbol, interval, rows)
            total_inserted += inserted
            last_close = int(rows[-1][6])
            status.update(
                f"[cyan]{symbol} [{interval}] — {total_inserted} bougies ajoutées "
                f"(jusqu'au {datetime.fromtimestamp(last_close / 1000).strftime('%Y-%m-%d %H:%M')})[/]"
            )
            if len(rows) < 1000:
                break
            start_ms = last_close + 1

    console.print(f"[green]{total_inserted} nouvelles bougies stockées pour {symbol} [{interval}].[/]")
    console.print(
        f"[dim]Lecture : pd.read_sql_query(\"SELECT * FROM klines WHERE symbol='{symbol}' "
        f"AND interval='{interval}' ORDER BY open_time\", conn)[/]"
    )


def register(sub):
    p = sub.add_parser("analyze", help="Analyser les positions du portefeuille pour identifier les sorties")
    p.add_argument("--interval", default="1h")

    p = sub.add_parser("exclude", help="Gérer la liste des symboles exclus des scans")
    p.add_argument("action", choices=["add", "remove", "list"])
    p.add_argument("symbols", nargs="*")

    p = sub.add_parser("fetch", help="Télécharger l'historique des klines et le stocker localement")
    p.add_argument("symbol")
    p.add_argument("--interval", default="15m")
    p.add_argument("--since", metavar="YYYY-MM-DD")

    p = sub.add_parser("sentiment", help="Afficher le sentiment X (Grok) pour des symboles")
    p.add_argument("symbols", nargs="*", metavar="SYMBOL",
                   help="Symboles à analyser (défaut : positions du portefeuille)")

    return {
        "analyze":   cmd_analyze,
        "exclude":   cmd_exclude,
        "fetch":     cmd_fetch,
        "sentiment": cmd_sentiment,
    }


def cmd_sentiment(args) -> None:
    from rich import box
    from rich.table import Table

    from ..config import GROK_API_KEY
    from ..sentiment import fetch_sentiment
    from ..storage import app_set_state

    init_db()

    if not GROK_API_KEY:
        console.print("[red]Clé Grok non configurée. Lance : crypto-portfolio setup-grok[/]")
        return

    symbols = [s.upper() for s in args.symbols] if args.symbols else None
    if not symbols:
        from ..storage import get_holdings
        holdings = get_holdings()
        symbols  = [h.symbol for h in holdings]
    if not symbols:
        console.print("[yellow]Aucun symbole. Spécifie des symboles ou ajoute des positions.[/]")
        return

    # Force refresh en vidant le cache
    app_set_state("grok_sentiment_ts", "0")

    from ..sentiment import _call_grok
    with console.status(f"[cyan]Interrogation de Grok sur {', '.join(symbols)}…[/]"):
        try:
            result = _call_grok(symbols)
        except Exception as e:
            console.print(f"[red]Erreur Grok : {e}[/]")
            return

    if not result:
        console.print("[red]Aucune réponse parseable de Grok (réponse vide ou JSON invalide).[/]")
        return

    table = Table(box=box.ROUNDED, title="[bold cyan]Sentiment X (Grok)[/]", title_justify="left")
    table.add_column("Symbole", style="bold")
    table.add_column("Score",   justify="center")
    table.add_column("Spike",   justify="center")
    table.add_column("Résumé")

    score_style = {"bullish": "green", "bearish": "red", "neutral": "dim"}
    for sym in symbols:
        s = result.get(sym)
        if not s:
            table.add_row(sym, "[dim]—[/]", "[dim]—[/]", "[dim]pas de données[/]")
            continue
        score = s.get("score", "neutral")
        spike = s.get("spike", False)
        color = score_style.get(score, "dim")
        table.add_row(
            sym,
            f"[{color}]{score}[/]",
            "[bold yellow]oui[/]" if spike else "[dim]non[/]",
            s.get("summary", ""),
        )

    console.print(table)
