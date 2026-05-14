import sys
import time

from ..binance import (get_all_tickers_24h, get_earn_balances, get_prices as binance_get_prices,
                       get_recent_klines, get_spot_balance, has_api_keys,
                       place_market_order, redeem_earn)
from ..config import (MAX_POSITIONS, POSITION_SIZE_PCT, QUOTE_CURRENCY,
                      STOP_LOSS_PCT, TAKE_PROFIT_1_PCT, TAKE_PROFIT_2_PCT, USDC_RESERVE_PCT)
from ..display import console
from ..indicators import (STABLECOINS, death_cross_recent, golden_cross_recent,
                           macd, price_cross_ma_recent, rsi_series, sma)
from ..portfolio import buy, get_portfolio, sell
from ..storage import get_excluded, init_db


def _ensure_spot(asset: str, needed: float) -> None:
    spot = get_spot_balance(asset)
    if spot >= needed:
        return
    deficit = needed - spot
    console.print(f"[yellow]Solde spot : {spot:.6g} {asset}. Rachat de {deficit:.6g} depuis Simple Earn…[/]")
    redeemed = redeem_earn(asset, deficit)
    if redeemed < deficit - 1e-8:
        console.print(f"[red]Rachat insuffisant : {redeemed:.6g} disponibles, {deficit:.6g} nécessaires.[/]")
        sys.exit(1)
    console.print(f"[green]Rachat effectué : {redeemed:.6g} {asset} → Spot.[/]")
    time.sleep(3)


def cmd_order(args) -> None:
    init_db()
    if not has_api_keys():
        console.print("[red]Aucune clé API configurée.[/]")
        sys.exit(1)

    side = args.side.upper()
    symbol = args.symbol.upper()
    quantity = args.quantity
    quote_qty = args.quote_qty

    if side == "SELL":
        _ensure_spot(symbol, quantity)
    else:
        needed = quote_qty if quote_qty is not None else quantity * binance_get_prices([symbol]).get(symbol, 0.0)
        _ensure_spot(QUOTE_CURRENCY, needed)

    desc = f"dépenser {quote_qty} {QUOTE_CURRENCY}" if quote_qty is not None else f"{quantity} {symbol}"
    console.print(f"\n[bold yellow]⚠ Ordre marché réel sur Binance[/]")
    console.print(f"  {side}  {desc}  ({symbol}{QUOTE_CURRENCY})")

    if input("\nConfirmer ? [o/N] ").strip().lower() != "o":
        console.print("[dim]Annulé.[/]")
        return

    try:
        avg_price, filled = place_market_order(symbol, side, quantity=quantity, quote_qty=quote_qty)
    except Exception as e:
        console.print(f"[red]Erreur Binance : {e}[/]")
        sys.exit(1)

    if side == "BUY":
        buy(symbol, filled, avg_price)
    else:
        try:
            sell(symbol, filled, avg_price)
        except ValueError as e:
            console.print(f"[yellow]Ordre exécuté mais erreur portefeuille local : {e}[/]")
            sys.exit(1)

    console.print(
        f"[green]✓ {side} {filled:.6g} {symbol} @ {avg_price:,.4f} = {filled * avg_price:,.2f} {QUOTE_CURRENCY}[/]"
    )


def cmd_rebalance(args) -> None:
    from rich import box
    from rich.table import Table

    init_db()
    if not has_api_keys():
        console.print("[red]Clés API requises.[/]")
        sys.exit(1)

    interval = args.interval

    with console.status("[cyan]Collecte des données…[/]"):
        holdings = get_portfolio()
        spot_usdc = get_spot_balance(QUOTE_CURRENCY)
        earn_usdc = get_earn_balances().get(QUOTE_CURRENCY, 0.0)

    total_usdc = spot_usdc + earn_usdc
    portfolio_value = sum(h.current_value for h in holdings)
    total_value = portfolio_value + total_usdc
    position_size_usdc = total_value * args.position_size / 100

    console.print(
        f"[bold]Valeur totale : {total_value:,.2f} {QUOTE_CURRENCY}[/]"
        f"  (portfolio : {portfolio_value:,.2f} | {QUOTE_CURRENCY} : {total_usdc:,.2f})\n"
    )

    # ── Sorties ───────────────────────────────────────────────────────────────
    exits = []
    with console.status("[cyan]Analyse des sorties…[/]") as status:
        for h in holdings:
            pnl = h.pnl_pct
            sym = h.holding.symbol
            status.update(f"[cyan]Sortie {sym}…[/]")

            if pnl <= -args.stop_loss:
                exits.append({"symbol": sym, "qty": h.holding.quantity, "value": h.current_value,
                               "pct": 100, "pnl_pct": pnl, "reason": "Stop-loss", "priority": 0})
                continue
            if pnl >= args.tp2:
                exits.append({"symbol": sym, "qty": h.holding.quantity, "value": h.current_value,
                               "pct": 100, "pnl_pct": pnl, "reason": "Take profit 2", "priority": 1})
                continue
            if pnl >= args.tp1:
                exits.append({"symbol": sym, "qty": h.holding.quantity * 0.5, "value": h.current_value * 0.5,
                               "pct": 50, "pnl_pct": pnl, "reason": "Take profit 1 (50%)", "priority": 2})
                continue

            try:
                klines = get_recent_klines(sym, interval, limit=60)
                closes  = [float(k[4]) for k in klines]
                volumes = [float(k[5]) for k in klines]
                if len(closes) < 20:
                    continue
                last = closes[-1]
                rsi = rsi_series(closes)[-1]
                ma20, ma50 = sma(closes, 20), sma(closes, 50)
                vol_avg   = sum(volumes[-10:]) / 10 if len(volumes) >= 10 else 0
                vol_drying = sum(volumes[-3:]) / 3 < vol_avg * 0.7 if vol_avg else False
                _, _, histogram = macd(closes)
                score, sigs = 0, []
                if rsi > 80:                        score += 3; sigs.append(f"RSI {rsi:.0f}")
                elif rsi > 70:                      score += 2; sigs.append(f"RSI {rsi:.0f}")
                if death_cross_recent(closes):      score += 3; sigs.append("Death cross")
                if ma50 and last < ma50:            score += 2; sigs.append("Sous MA50")
                elif ma20 and last < ma20:          score += 1; sigs.append("Sous MA20")
                if vol_drying:                      score += 1; sigs.append("Vol tarit")
                if histogram and histogram[-1] < 0 and len(histogram) >= 2 and histogram[-2] > 0:
                    score += 2; sigs.append("MACD ↓")
                if score >= 5:
                    exits.append({"symbol": sym, "qty": h.holding.quantity, "value": h.current_value,
                                   "pct": 100, "pnl_pct": pnl, "reason": f"Technique : {', '.join(sigs)}", "priority": 3})
            except Exception:
                pass

    usdc_freed = sum(e["value"] for e in exits)
    exited_full = {e["symbol"] for e in exits if e["pct"] == 100}
    remaining = len(holdings) - len(exited_full)

    # ── Entrées ───────────────────────────────────────────────────────────────
    min_reserve = total_value * args.reserve / 100
    deployable  = max(0.0, total_usdc + usdc_freed - min_reserve)
    max_new     = min(args.max_positions - remaining,
                      int(deployable / position_size_usdc) if position_size_usdc else 0)

    entries  = []
    excluded = get_excluded()
    existing = {h.holding.symbol for h in holdings}

    if max_new > 0:
        with console.status("[cyan]Scan des opportunités…[/]") as status:
            pool = [
                {"symbol": t["symbol"].removesuffix(QUOTE_CURRENCY),
                 "change_pct": float(t["priceChangePercent"]),
                 "quote_vol": float(t["quoteVolume"])}
                for t in get_all_tickers_24h()
                if t["symbol"].endswith(QUOTE_CURRENCY)
                and t["symbol"].removesuffix(QUOTE_CURRENCY) not in STABLECOINS
                and t["symbol"].removesuffix(QUOTE_CURRENCY) not in excluded
                and t["symbol"].removesuffix(QUOTE_CURRENCY) not in existing
                and float(t["quoteVolume"]) >= 50_000
                and abs(float(t["priceChangePercent"])) <= 20
            ]
            pool.sort(key=lambda x: x["quote_vol"], reverse=True)

            opportunities = []
            for c in pool[:60]:
                status.update(f"[cyan]Scan {c['symbol']}…[/]")
                try:
                    klines = get_recent_klines(c["symbol"], interval, limit=60)
                    closes  = [float(k[4]) for k in klines]
                    volumes = [float(k[5]) for k in klines]
                except Exception:
                    continue
                if len(closes) < 50:
                    continue

                last = closes[-1]
                rsi_vals = rsi_series(closes)
                rsi_now, rsi_prev5 = rsi_vals[-1], (rsi_vals[-6] if len(rsi_vals) >= 6 else rsi_vals[-1])
                ma20, ma50 = sma(closes, 20), sma(closes, 50)
                _, _, histogram = macd(closes)
                vol_avg   = sum(volumes[-10:]) / 10 if len(volumes) >= 10 else 0
                vol_build = sum(volumes[-3:]) / 3 > vol_avg * 1.2 if vol_avg else False
                uptrend   = ma20 and ma50 and ma20 > ma50
                dip_pct   = (max(closes[-30:]) - last) / max(closes[-30:]) * 100 if len(closes) >= 30 else 0

                score, sigs = 0, []
                if golden_cross_recent(closes):                     score += 3; sigs.append("Golden cross")
                if price_cross_ma_recent(closes, 20):               score += 2; sigs.append("Breakout MA20")
                if rsi_now > rsi_prev5 + 3 and 38 <= rsi_now <= 58:score += 2; sigs.append(f"RSI ↑{rsi_now:.0f}")
                elif 45 <= rsi_now <= 58:                           score += 1; sigs.append(f"RSI {rsi_now:.0f}")
                if uptrend and 25 <= rsi_now <= 42:                 score += 2; sigs.append(f"Dip RSI {rsi_now:.0f}")
                if histogram and histogram[-1] > 0 and len(histogram) >= 2 and histogram[-2] < 0:
                    score += 2; sigs.append("MACD ↑")
                if vol_build:                                        score += 1; sigs.append("Vol ↑")
                if ma50 and last > ma50:                             score += 1; sigs.append("↑MA50")
                if uptrend and 5 <= dip_pct <= 25:                  score += 1; sigs.append(f"Dip -{dip_pct:.1f}%")

                if score >= 5:
                    opportunities.append({**c, "score": score, "signals": sigs})

            opportunities.sort(key=lambda x: x["score"], reverse=True)
            entries = [{"symbol": o["symbol"], "amount": position_size_usdc,
                        "score": o["score"], "signals": o["signals"],
                        "change_pct": o["change_pct"]} for o in opportunities[:max_new]]

    # ── Affichage ─────────────────────────────────────────────────────────────
    if not exits and not entries:
        console.print("[green]Portefeuille équilibré — aucune action suggérée.[/]")
        return

    if exits:
        te = Table(title="[bold red]Sorties suggérées[/]", box=box.ROUNDED, title_justify="left")
        te.add_column("Actif", style="bold"); te.add_column("Qté", justify="right")
        te.add_column(QUOTE_CURRENCY, justify="right"); te.add_column("P&L", justify="right")
        te.add_column("Raison")
        for e in sorted(exits, key=lambda x: x["priority"]):
            c = "green" if e["pnl_pct"] >= 0 else "red"
            te.add_row(e["symbol"], f"{e['qty']:.6g}", f"{e['value']:,.2f}",
                       f"[{c}]{e['pnl_pct']:+.1f}%[/]", e["reason"])
        console.print(te)
        console.print(f"  USDC estimé libéré : [bold green]{usdc_freed:,.2f}[/]\n")

    if entries:
        tn = Table(title="[bold green]Entrées suggérées[/]", box=box.ROUNDED, title_justify="left")
        tn.add_column("Actif", style="bold"); tn.add_column(QUOTE_CURRENCY, justify="right")
        tn.add_column("24h", justify="right"); tn.add_column("Score", justify="center")
        tn.add_column("Signaux")
        for e in entries:
            c = "green" if e["change_pct"] > 0 else "red"
            tn.add_row(e["symbol"], f"{e['amount']:,.2f}", f"[{c}]{e['change_pct']:+.1f}%[/]",
                       f"[green]{e['score']}[/]", "  ".join(e["signals"]))
        console.print(tn)
        console.print(f"  USDC déployé : [bold]{sum(e['amount'] for e in entries):,.2f}[/]"
                      f"  |  Réserve : [bold]{min_reserve:,.2f}[/]\n")

    if input(f"Exécuter {len(exits) + len(entries)} trade(s) ? [o/N] ").strip().lower() != "o":
        console.print("[dim]Annulé.[/]")
        return

    for e in exits:
        sym, qty = e["symbol"], e["qty"]
        try:
            _ensure_spot(sym, qty)
            avg_price, filled = place_market_order(sym, "SELL", quantity=qty)
            sell(sym, filled, avg_price)
            console.print(f"[green]✓ SELL {filled:.6g} {sym} @ {avg_price:,.4f} = {filled * avg_price:,.2f} {QUOTE_CURRENCY}[/]")
        except Exception as err:
            console.print(f"[red]✗ {sym} : {err}[/]")

    if entries:
        time.sleep(3)

    for e in entries:
        sym, amount = e["symbol"], e["amount"]
        try:
            _ensure_spot(QUOTE_CURRENCY, amount)
            avg_price, filled = place_market_order(sym, "BUY", quote_qty=amount)
            buy(sym, filled, avg_price)
            console.print(f"[green]✓ BUY {filled:.6g} {sym} @ {avg_price:,.4f} = {amount:,.2f} {QUOTE_CURRENCY}[/]")
        except Exception as err:
            console.print(f"[red]✗ {sym} : {err}[/]")


def register(sub):
    p = sub.add_parser("order", help="Passer un ordre marché réel sur Binance")
    p.add_argument("side", choices=["buy", "sell"])
    p.add_argument("symbol")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("quantity", type=float, nargs="?", help="Quantité en actif de base")
    g.add_argument("--for", dest="quote_qty", type=float, metavar="AMOUNT",
                   help=f"Montant en {QUOTE_CURRENCY} (ex: --for 91.6)")

    p = sub.add_parser("rebalance", help="Suggère et exécute les rééquilibrages")
    p.add_argument("--interval", default="1h")
    p.add_argument("--stop-loss",      type=float, default=STOP_LOSS_PCT,      dest="stop_loss",      metavar="PCT")
    p.add_argument("--tp1",            type=float, default=TAKE_PROFIT_1_PCT,  metavar="PCT")
    p.add_argument("--tp2",            type=float, default=TAKE_PROFIT_2_PCT,  metavar="PCT")
    p.add_argument("--position-size",  type=float, default=POSITION_SIZE_PCT,  dest="position_size",  metavar="PCT")
    p.add_argument("--max-positions",  type=int,   default=MAX_POSITIONS,      dest="max_positions",  metavar="N")
    p.add_argument("--reserve",        type=float, default=USDC_RESERVE_PCT,   metavar="PCT")

    return {"order": cmd_order, "rebalance": cmd_rebalance}
