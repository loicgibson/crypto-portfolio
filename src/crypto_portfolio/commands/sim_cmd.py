"""
sim-reset   : initialize / reset the virtual portfolio (1000 USDC by default)
sim-status  : display current virtual portfolio state
sim-run     : one rebalancing cycle (scan → Claude brain → virtual execution)
sim-loop    : repeat cycles every N minutes
sim-history : show recent virtual transactions and cycle performance
"""
import time

from ..binance import get_prices
from ..display import console
from ..storage import (init_db, sim_add_cycle, sim_add_transaction, sim_get_cycles,
                       sim_get_holdings, sim_get_state, sim_get_transactions,
                       sim_get_usdc, sim_reset, sim_set_state, sim_set_usdc,
                       sim_upsert_holding)
from ._market import (PortfolioBackend, _now_iso,
                      _print_portfolio_snapshot, run_combined_cycle,
                      run_watchlist_cycle)


# ── Virtual trade execution ───────────────────────────────────────────────────

def _record_pnl(pnl_pct: float) -> None:
    import json
    raw  = sim_get_state("recent_pnls")
    pnls = json.loads(raw) if raw else []
    pnls.insert(0, round(pnl_pct, 2))
    sim_set_state("recent_pnls", json.dumps(pnls[:10]))

def _sim_buy(sym: str, usdc_amount: float, price: float,
             reason: str, now: str) -> dict | None:
    if price <= 0 or usdc_amount < 5:
        return None
    usdc        = sim_get_usdc()
    usdc_amount = min(usdc_amount, usdc)
    if usdc_amount < 5:
        return None

    qty        = usdc_amount / price
    usdc_after = usdc - usdc_amount
    holdings   = {h.symbol: h for h in sim_get_holdings()}
    if sym in holdings:
        h         = holdings[sym]
        total_qty = h.quantity + qty
        new_avg   = (h.quantity * h.avg_buy_price + qty * price) / total_qty
        sim_upsert_holding(sym, total_qty, new_avg)
    else:
        sim_upsert_holding(sym, qty, price)
    sim_set_usdc(usdc_after)
    sim_add_transaction(now, sym, "BUY", qty, price, usdc, usdc_after, reason)
    return {"action": "BUY", "symbol": sym, "qty": qty, "price": price,
            "usdc_spent": usdc_amount, "reason": reason}


def _sim_sell(sym: str, price: float, reason: str, now: str) -> dict | None:
    if price <= 0:
        return None
    holdings = {h.symbol: h for h in sim_get_holdings()}
    if sym not in holdings:
        return None
    h          = holdings[sym]
    proceeds   = h.quantity * price
    usdc       = sim_get_usdc()
    usdc_after = usdc + proceeds
    sim_upsert_holding(sym, 0, 0)
    sim_set_usdc(usdc_after)
    sim_add_transaction(now, sym, "SELL", h.quantity, price, usdc, usdc_after, reason)
    pnl_pct = ((price / h.avg_buy_price) - 1) * 100 if h.avg_buy_price > 0 else 0.0
    _record_pnl(pnl_pct)
    return {"action": "SELL", "symbol": sym, "qty": h.quantity, "price": price,
            "proceeds": proceeds, "pnl_pct": pnl_pct, "reason": reason}



def _sim_execute(actions: list, context: dict, now: str, _dry_run: bool) -> list[dict]:
    """Execute decisions against the virtual portfolio. dry_run is ignored (sim is already virtual)."""
    symbols  = list({a["symbol"].upper() for a in actions if a["action"] in ("BUY", "SELL")})
    prices   = get_prices(symbols) if symbols else {}
    executed = []
    for action in actions:
        sym    = action["symbol"].upper()
        act    = action["action"]
        reason = action.get("reason", "")
        price  = prices.get(sym, 0.0)
        result = None
        if act == "SELL":
            result = _sim_sell(sym, price, reason, now)
        elif act == "BUY":
            usdc_amount = float(action.get("usdc_amount", 0.0))
            result      = _sim_buy(sym, usdc_amount, price, reason, now)
        if result:
            executed.append(result)
    return executed


_SIM_BACKEND = PortfolioBackend(
    label="Sim",
    get_usdc=sim_get_usdc,
    get_holdings=sim_get_holdings,
    get_transactions=sim_get_transactions,
    get_state=sim_get_state,
    set_state=sim_set_state,
    add_cycle=sim_add_cycle,
    execute=_sim_execute,
)


# ── Commands ──────────────────────────────────────────────────────────────────

def cmd_sim_reset(args) -> None:
    init_db()
    sim_reset()
    now     = _now_iso()
    balance = args.balance
    sim_set_state("initial_balance", str(balance))
    sim_set_state("started_at", now)
    sim_set_usdc(balance)
    console.print(
        f"[bold green]Simulation initialisée — {balance:.2f} USDC[/]\n"
        f"[dim]Lance [bold]sim-run[/] pour un premier cycle ou "
        "[bold]sim-loop[/] pour démarrer la boucle automatique.[/]"
    )


def cmd_sim_status(args) -> None:
    from rich import box
    from rich.table import Table

    init_db()
    if not sim_get_state("initial_balance"):
        console.print("[yellow]Simulation non initialisée. Lance : sim-reset[/]")
        return

    usdc     = sim_get_usdc()
    holdings = sim_get_holdings()
    initial  = float(sim_get_state("initial_balance", "1000"))
    started  = sim_get_state("started_at", "?")
    prices   = get_prices([h.symbol for h in holdings]) if holdings else {}

    table = Table(title="[bold cyan]Simulation — Portefeuille virtuel[/]",
                  box=box.ROUNDED, title_justify="left")
    table.add_column("Actif",       style="bold")
    table.add_column("Qté",         justify="right")
    table.add_column("Prix moy.",   justify="right")
    table.add_column("Prix actuel", justify="right")
    table.add_column("Valeur USDC", justify="right")
    table.add_column("P&L %",       justify="right")

    total_crypto = 0.0
    for h in holdings:
        price   = prices.get(h.symbol, 0.0)
        cur_val = h.quantity * price
        pnl_pct = ((price / h.avg_buy_price) - 1) * 100 if h.avg_buy_price > 0 and price > 0 else 0.0
        total_crypto += cur_val
        c     = "green" if pnl_pct >= 0 else "red"
        label = h.symbol
        table.add_row(
            label, f"{h.quantity:.6g}", f"{h.avg_buy_price:.4g}",
            f"{price:.4g}", f"{cur_val:.2f}", f"[{c}]{pnl_pct:+.1f}%[/]",
        )

    total = usdc + total_crypto
    perf  = (total / initial - 1) * 100 if initial > 0 else 0.0
    pc    = "green" if perf >= 0 else "red"

    if holdings:
        table.add_section()
    table.add_row("[bold]USDC[/]",  "", "", "", f"[bold]{usdc:.2f}[/]", "")
    table.add_row("[bold]TOTAL[/]", "", "", "", f"[bold]{total:.2f}[/]",
                  f"[bold {pc}]{perf:+.1f}%[/]")

    console.print(table)
    console.print(f"[dim]Capital initial : {initial:.2f} USDC | Démarré : {started}[/]")

    cycles = sim_get_cycles(limit=1)
    if cycles:
        c = cycles[0]
        console.print(f"[dim]Dernier cycle : {c['timestamp']} — {c['actions_taken']} action(s)[/]")


def cmd_sim_run(args) -> None:
    init_db()
    if not sim_get_state("initial_balance"):
        console.print("[red]Simulation non initialisée. Lance : sim-reset[/]")
        return
    run_combined_cycle(_SIM_BACKEND, verbose=args.verbose)


def cmd_sim_loop(args) -> None:
    init_db()
    if not sim_get_state("initial_balance"):
        console.print("[red]Simulation non initialisée. Lance : sim-reset[/]")
        return
    _SUB_INTERVAL_MIN  = 5
    main_interval_secs = args.interval * 60
    sub_interval_secs  = _SUB_INTERVAL_MIN * 60

    console.print(
        f"[bold cyan]Simulation loop[/] — sous-cycle [bold]{_SUB_INTERVAL_MIN} min[/] (watchlist)"
        f" + cycle principal [bold]{args.interval} min[/]. "
        f"Ctrl+C pour arrêter.\n"
    )

    main_cycle   = 0
    sub_cycle    = 0
    last_main_ts = 0.0  # force le cycle principal dès la première itération

    while True:
        sub_cycle += 1

        if time.time() - last_main_ts >= main_interval_secs:
            last_main_ts = time.time()
            main_cycle  += 1
            console.print(f"[bold]━━ Cycle principal {main_cycle} / sous-cycle {sub_cycle} — {_now_iso()} ━━[/]")
            try:
                run_combined_cycle(_SIM_BACKEND, verbose=args.verbose)
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                console.print(f"[red]Erreur cycle principal {main_cycle} : {exc}[/]")
        else:
            console.print(f"[bold]━━ Sous-cycle {sub_cycle} — {_now_iso()} ━━[/]")
            try:
                run_watchlist_cycle(_SIM_BACKEND)
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                console.print(f"[red]Erreur sous-cycle {sub_cycle} : {exc}[/]")

        console.print(f"\n[dim]Prochain sous-cycle dans {_SUB_INTERVAL_MIN} min…[/]\n")
        try:
            time.sleep(sub_interval_secs)
        except KeyboardInterrupt:
            console.print("\n[yellow]Loop arrêtée.[/]")
            break


def cmd_sim_history(args) -> None:
    from rich import box
    from rich.table import Table

    init_db()

    cycles = sim_get_cycles(limit=args.cycles)
    if cycles:
        initial = float(sim_get_state("initial_balance", "1000"))
        ctable  = Table(title=f"[bold cyan]Cycles (derniers {len(cycles)})[/]",
                        box=box.SIMPLE, title_justify="left")
        ctable.add_column("Date",           style="dim")
        ctable.add_column("Total USDC",     justify="right")
        ctable.add_column("P&L vs initial", justify="right")
        ctable.add_column("Actions",        justify="center")
        ctable.add_column("Résumé marché")

        for cy in reversed(cycles):
            perf = (cy["total_value"] / initial - 1) * 100 if initial > 0 else 0.0
            pc   = "green" if perf >= 0 else "red"
            ctable.add_row(
                cy["timestamp"][:19], f"{cy['total_value']:.2f}",
                f"[{pc}]{perf:+.1f}%[/]", str(cy["actions_taken"]),
                (cy["market_summary"] or "")[:80],
            )
        console.print(ctable)

    txs = sim_get_transactions(limit=args.limit)
    if not txs:
        console.print("[dim]Aucune transaction de simulation.[/]")
        return

    table = Table(title=f"[bold cyan]Transactions virtuelles (dernières {len(txs)})[/]",
                  box=box.ROUNDED, title_justify="left")
    table.add_column("Date",   style="dim")
    table.add_column("Action", justify="center")
    table.add_column("Actif",  style="bold")
    table.add_column("Qté",    justify="right")
    table.add_column("Prix",   justify="right")
    table.add_column("ΔUSDC",  justify="right")
    table.add_column("Raison")

    for tx in txs:
        color      = "green" if tx["tx_type"] == "BUY" else "red"
        usdc_delta = tx["usdc_after"] - tx["usdc_before"]
        table.add_row(
            tx["timestamp"][:19],
            f"[{color}]{tx['tx_type']}[/]",
            tx["symbol"], f"{tx['quantity']:.6g}", f"{tx['price']:.4g}",
            f"[{color}]{usdc_delta:+.2f}[/]",
            (tx["reason"] or "")[:70],
        )
    console.print(table)


def cmd_sim_recap(args) -> None:
    from collections import defaultdict
    from datetime import datetime, timedelta, timezone

    from rich import box
    from rich.table import Table

    init_db()

    tz_p2    = timezone(timedelta(hours=2))
    date_str = args.date or datetime.now(tz_p2).strftime("%d-%m-%Y")

    try:
        day_naive = datetime.strptime(date_str, "%d-%m-%Y")
    except ValueError:
        console.print(f"[red]Format invalide : {date_str}. Utilise dd-MM-YYYY (ex: 04-05-2026)[/]")
        return

    tz_p2     = timezone(timedelta(hours=2))
    day_start = day_naive.replace(tzinfo=tz_p2)
    day_end   = day_start + timedelta(days=1)

    def _parse(ts_str: str) -> datetime:
        dt = datetime.fromisoformat(ts_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(tz_p2)

    def _on_day(dt: datetime) -> bool:
        return day_start <= dt < day_end

    # ── FIFO pairing ──────────────────────────────────────────────────────────
    all_txs = list(reversed(sim_get_transactions(limit=10000)))
    txs = all_txs
    for tx in txs:
        tx["_ts"] = _parse(tx["timestamp"])

    open_buys: dict[str, list] = defaultdict(list)
    pairs: list[dict] = []

    for tx in txs:
        sym = tx["symbol"]
        if tx["tx_type"] == "BUY":
            open_buys[sym].append(tx)
        elif tx["tx_type"] == "SELL":
            if open_buys[sym]:
                # One SELL always closes the full position — could span multiple BUYs.
                # Split proportionally: each buy receives buy_qty × sell_price.
                for buy_tx in open_buys[sym]:
                    pairs.append({
                        "symbol":    sym,
                        "buy":       buy_tx,
                        "sell":      tx,
                        "sell_qty":  buy_tx["quantity"],  # proportional slice
                    })
                open_buys[sym] = []
            else:
                pairs.append({
                    "symbol":   sym,
                    "buy":      None,
                    "sell":     tx,
                    "sell_qty": tx["quantity"],
                })

    for sym, buys in open_buys.items():
        for b in buys:
            pairs.append({"symbol": sym, "buy": b, "sell": None, "sell_qty": None})

    filtered = [
        p for p in pairs
        if (p["buy"]  and _on_day(p["buy"]["_ts"]))
        or (p["sell"] and _on_day(p["sell"]["_ts"]))
    ]

    if not filtered:
        console.print(f"[dim]Aucune transaction le {date_str}.[/]")
        return

    # ── Current prices for open positions ─────────────────────────────────────
    open_syms  = list({p["symbol"] for p in filtered if p["sell"] is None})
    cur_prices = get_prices(open_syms) if open_syms else {}

    # ── Build rows ────────────────────────────────────────────────────────────
    rows = []
    for p in filtered:
        buy_tx  = p["buy"]
        sell_tx = p["sell"]
        sym     = p["symbol"]

        buy_price  = buy_tx["price"]    if buy_tx  else None
        sell_price = sell_tx["price"]   if sell_tx else None
        cur_price  = cur_prices.get(sym)

        buy_qty   = buy_tx["quantity"] if buy_tx else None
        # sell_qty is the proportional slice (= buy_qty when the sell covered multiple buys)
        eff_sell_qty = p.get("sell_qty")
        buy_usdc  = buy_qty      * buy_price  if buy_qty      and buy_price  else None
        sell_usdc = eff_sell_qty * sell_price if eff_sell_qty and sell_price else None
        cur_usdc  = buy_qty      * cur_price  if buy_qty      and cur_price and not sell_tx else None

        ref_price  = sell_price if sell_tx else cur_price
        ref_usdc   = sell_usdc  if sell_tx else cur_usdc
        pnl_pct    = ((ref_price / buy_price) - 1) * 100 if buy_price and ref_price else None
        delta_usdc = (ref_usdc - buy_usdc) if ref_usdc is not None and buy_usdc is not None else None

        rows.append({
            "symbol":     sym,
            "buy_ts":     buy_tx["_ts"].strftime("%H:%M") if buy_tx  else None,
            "buy_price":  buy_price,
            "buy_usdc":   buy_usdc,
            "sell_ts":    sell_tx["_ts"].strftime("%H:%M") if sell_tx else None,
            "sell_price": sell_price,
            "sell_usdc":  sell_usdc,
            "cur_price":  cur_price if not sell_tx else None,
            "cur_usdc":   cur_usdc,
            "delta_usdc": delta_usdc,
            "pnl_pct":    pnl_pct,
            "open":       sell_tx is None,
        })

    rows.sort(key=lambda r: r["delta_usdc"] if r["delta_usdc"] is not None else float("-inf"), reverse=True)

    # ── Render table ──────────────────────────────────────────────────────────
    table = Table(
        title=f"[bold cyan]Récapitulatif sim {date_str} (GMT+2)[/]",
        box=box.ROUNDED, title_justify="left",
    )
    table.add_column("Symbole",    style="bold")
    table.add_column("Entrée",     justify="right", style="dim")
    table.add_column("Prix achat", justify="right")
    table.add_column("Investi",    justify="right")
    table.add_column("Sortie",     justify="right", style="dim")
    table.add_column("Prix vente", justify="right")
    table.add_column("Récupéré",   justify="right")
    table.add_column("Delta",      justify="right")
    table.add_column("Perf",       justify="right")

    for r in rows:
        buy_p  = f"{r['buy_price']:.6g}"  if r["buy_price"]  else "—"
        buy_u  = f"{r['buy_usdc']:.2f}"   if r["buy_usdc"]   else "—"
        buy_ts = r["buy_ts"] or "—"

        if r["open"]:
            sell_ts = "[dim]ouvert[/]"
            sell_p  = f"[dim]{r['cur_price']:.6g}[/]" if r["cur_price"] else "—"
            sell_u  = f"[dim]{r['cur_usdc']:.2f}[/]"  if r["cur_usdc"]  else "—"
        else:
            sell_ts = r["sell_ts"] or "?"
            sell_p  = f"{r['sell_price']:.6g}" if r["sell_price"] else "—"
            sell_u  = f"{r['sell_usdc']:.2f}"  if r["sell_usdc"]  else "—"

        if r["delta_usdc"] is not None:
            dc      = "green" if r["delta_usdc"] >= 0 else "red"
            delta_s = f"[{dc}]{r['delta_usdc']:+.2f}[/]"
        else:
            delta_s = "—"

        if r["pnl_pct"] is not None:
            c       = "green" if r["pnl_pct"] >= 0 else "red"
            pnl_str = f"[{c}]{r['pnl_pct']:+.2f}%[/]"
            if r["open"]:
                pnl_str += " [dim](live)[/]"
        else:
            pnl_str = "—"

        table.add_row(r["symbol"], buy_ts, buy_p, buy_u, sell_ts, sell_p, sell_u, delta_s, pnl_str)

    console.print(table)

    # ── Summary footer ────────────────────────────────────────────────────────
    from rich.rule import Rule

    closed = [r for r in rows if not r["open"] and r["pnl_pct"] is not None]
    open_  = [r for r in rows if r["open"]]

    if not closed and not open_:
        return

    def _c(v):        return "green" if v >= 0 else "red"
    def _pct(d, ref): return (d / ref * 100) if ref > 0 else 0.0

    in_closed  = sum(r["buy_usdc"]  for r in closed if r["buy_usdc"])
    out_closed = sum(r["sell_usdc"] for r in closed if r["sell_usdc"])
    d_closed   = out_closed - in_closed

    in_open  = sum(r["buy_usdc"] for r in open_ if r["buy_usdc"])
    cur_open = sum(r["cur_usdc"] for r in open_ if r["cur_usdc"])
    d_open   = cur_open - in_open

    total_in  = in_closed + in_open
    total_val = out_closed + cur_open
    total_d   = total_val - total_in

    console.print(Rule("[bold dim]Bilan journée[/bold dim]", style="dim"))
    if closed:
        avg_pct = sum(r["pnl_pct"] for r in closed) / len(closed)
        console.print(
            f"  Clôturés ({len(closed)})   "
            f"investi {in_closed:.2f} → récupéré {out_closed:.2f} USDC   "
            f"[{_c(d_closed)}]{d_closed:+.2f} ({_pct(d_closed, in_closed):+.1f}%)[/]   "
            f"moy. [{_c(avg_pct)}]{avg_pct:+.1f}%[/]"
        )
    if open_:
        console.print(
            f"  Ouverts  ({len(open_)})   "
            f"investi {in_open:.2f} → valeur   {cur_open:.2f} USDC   "
            f"[{_c(d_open)}]{d_open:+.2f} ({_pct(d_open, in_open):+.1f}%)[/]   [dim](latent)[/]"
        )
    console.print(
        f"  Total        investi {total_in:.2f} → valeur   {total_val:.2f} USDC   "
        f"[bold {_c(total_d)}]{total_d:+.2f} ({_pct(total_d, total_in):+.1f}%)[/]"
    )

    # Portfolio value evolution from cycles + current state
    all_cycles   = sim_get_cycles(limit=2000)
    pre_day      = [cy for cy in all_cycles if _parse(cy["timestamp"]) < day_start]
    day_cycs     = [cy for cy in all_cycles if _on_day(_parse(cy["timestamp"]))]
    start_val    = (pre_day[0]["total_value"]   if pre_day  else
                    day_cycs[-1]["total_value"]  if day_cycs else None)

    usdc_now     = sim_get_usdc()
    holdings_now = sim_get_holdings()
    hold_syms    = [h.symbol for h in holdings_now]
    hold_prices  = get_prices(hold_syms) if hold_syms else {}
    crypto_now   = sum(h.quantity * hold_prices.get(h.symbol, 0.0) for h in holdings_now)
    total_now    = usdc_now + crypto_now

    console.print()
    if start_val is not None:
        pf_d = total_now - start_val
        pf_p = _pct(pf_d, start_val)
        console.print(
            f"  Portefeuille   début {start_val:.2f} → actuel {total_now:.2f} USDC   "
            f"[bold {_c(pf_d)}]{pf_d:+.2f} USDC ({pf_p:+.1f}%)[/]"
        )
    else:
        console.print(f"  Portefeuille   {total_now:.2f} USDC (valeur actuelle)")


# ── Register ──────────────────────────────────────────────────────────────────

def register(sub):
    p = sub.add_parser("sim-reset",
                       help="Initialiser / réinitialiser le portefeuille virtuel")
    p.add_argument("--balance", type=float, default=1000.0,
                   metavar="USDC", help="Capital de départ (défaut : 1000 USDC)")

    sub.add_parser("sim-status", help="Afficher le portefeuille virtuel")

    p = sub.add_parser("sim-run", help="Un cycle sim (scan pump Tier 2)")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Afficher le JSON envoye a l'API")

    p = sub.add_parser("sim-loop", help="Boucle sim : scan pump Tier-2 toutes les N min")
    p.add_argument("--interval",  type=int, default=15,
                   help="Minutes entre chaque scan Tier-2 (defaut : 15)")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Afficher le JSON envoye a l'API (chaque cycle)")

    p = sub.add_parser("sim-history",
                       help="Historique des transactions et cycles virtuels")
    p.add_argument("--limit",  type=int, default=30,
                   help="Nombre de transactions a afficher (defaut : 30)")
    p.add_argument("--cycles", type=int, default=10,
                   help="Nombre de cycles a afficher (defaut : 10)")

    p = sub.add_parser("sim-recap",
                       help="Récapitulatif des trades d'une journée (dd-MM-YYYY, défaut : aujourd'hui)")
    p.add_argument("date", nargs="?", default=None,
                   help="Date au format dd-MM-YYYY (ex: 04-05-2026, défaut : aujourd'hui)")

    return {
        "sim-reset":   cmd_sim_reset,
        "sim-status":  cmd_sim_status,
        "sim-run":     cmd_sim_run,
        "sim-loop":    cmd_sim_loop,
        "sim-history": cmd_sim_history,
        "sim-recap":   cmd_sim_recap,
    }
