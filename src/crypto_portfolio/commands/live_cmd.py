"""
live-run     : one trading cycle on real Binance account
live-loop    : automated loop — main cycle every N min + watchlist sub-cycle every 5 min
live-status  : show portfolio tracked by this tool + real Binance USDC balance
live-history : recent live transactions and cycle performance
"""
import math
import sys
import time

from ..binance import (get_account_balances, get_earn_balances, get_lot_size,
                       get_my_trades, get_prices, get_spot_balance, has_api_keys,
                       place_market_order, redeem_earn)
from ..config import QUOTE_CURRENCY
from ..display import console

_IGNORED_ASSETS  = frozenset({QUOTE_CURRENCY, "USDT", "BUSD", "FDUSD", "TUSD", "DAI"})
_MIN_MANUAL_USDC = 1.0  # ignore dust < $1 when detecting manual buys
from ..storage import (init_db, live_add_cycle, live_add_transaction,
                       live_clear_holdings, live_get_cycles, live_get_holdings,
                       live_get_state, live_get_transactions, live_set_state,
                       live_upsert_holding)
from ._market import (PortfolioBackend, _now_iso,
                      _print_portfolio_snapshot, run_combined_cycle,
                      run_watchlist_cycle)


# ── Live helpers ──────────────────────────────────────────────────────────────

def _floor_qty(qty: float, step: float) -> float:
    """Floor qty to the nearest valid stepSize, avoid float precision drift."""
    if step <= 0:
        return qty
    precision = max(0, round(-math.log10(step))) if step < 1 else 0
    return round(math.floor(qty / step) * step, precision)


def _live_usdc() -> float:
    """Spot USDC + Simple Earn USDC (for display only)."""
    spot = get_spot_balance(QUOTE_CURRENCY)
    earn = get_earn_balances().get(QUOTE_CURRENCY, 0.0)
    return spot + earn


def _current_total() -> float:
    """Current total portfolio value: USDC + crypto holdings at market price."""
    usdc     = _live_usdc()
    holdings = live_get_holdings()
    prices   = get_prices([h.symbol for h in holdings]) if holdings else {}
    crypto   = sum(h.quantity * prices.get(h.symbol, 0) for h in holdings)
    return usdc + crypto


def _record_live_pnl(pnl_pct: float) -> None:
    import json
    raw  = live_get_state("recent_pnls")
    pnls = json.loads(raw) if raw else []
    pnls.insert(0, round(pnl_pct, 2))
    live_set_state("recent_pnls", json.dumps(pnls[:10]))


def _check_api_keys() -> None:
    if not has_api_keys():
        console.print("[red]Clés API Binance non configurées. Lance : crypto-portfolio setup[/]")
        sys.exit(1)


def _ensure_spot(asset: str, needed: float) -> None:
    spot = get_spot_balance(asset)
    if spot >= needed - 1e-8:
        return
    deficit = needed - spot
    console.print(f"[yellow]Spot {asset} : {spot:.6g}. Rachat {deficit:.6g} depuis Earn…[/]")
    redeemed = redeem_earn(asset, deficit)
    if redeemed < deficit - 1e-8:
        console.print(f"[red]Rachat insuffisant ({redeemed:.6g} / {deficit:.6g}).[/]")
        raise RuntimeError(f"Solde {asset} insuffisant après rachat Earn")
    time.sleep(3)


# ── External movement detection ───────────────────────────────────────────────

def _sync_external_movements(dry_run: bool = False) -> int:
    """
    Reconcile live_holdings against actual Binance balances (spot + earn).

    Manual SELL detected (tracked holding no longer on Binance):
      → log a manual_sell transaction, remove from tracking.

    Manual BUY detected (asset on Binance not tracked, or significantly more
    than tracked):
      → log a manual_buy transaction, add/update tracking.
      The brain will evaluate these positions in the next cycle and may sell
      if it deems them a bad idea.

    Returns total number of discrepancies handled.
    """
    holdings = {h.symbol: h for h in live_get_holdings()}

    spot_all = get_account_balances()
    earn_all = get_earn_balances()

    all_actual: dict[str, float] = {}
    for asset, qty in spot_all.items():
        all_actual[asset] = all_actual.get(asset, 0.0) + qty
    for asset, qty in earn_all.items():
        all_actual[asset] = all_actual.get(asset, 0.0) + qty

    assets_to_price = set(holdings.keys()) | {
        a for a in all_actual
        if a not in _IGNORED_ASSETS and not a.startswith("LD") and all_actual[a] > 1e-10
    }
    prices = get_prices(list(assets_to_price)) if assets_to_price else {}

    now = _now_iso()
    thirty_days_ms = int((time.time() - 30 * 86400) * 1000)
    changes = 0

    # ── 1. Manual sells: tracked holding vanished from Binance ────────────────
    for sym, h in holdings.items():
        actual = all_actual.get(sym, 0.0)
        if actual >= h.quantity * 0.1:
            continue  # still mostly present

        qty_sold   = h.quantity - max(actual, 0.0)
        sell_price = prices.get(sym) or h.avg_buy_price

        try:
            raw_trades = get_my_trades(sym, start_ms=thirty_days_ms)
            sells = [t for t in raw_trades if not t["isBuyer"]]
            if sells:
                total_qty  = sum(float(t["qty"])      for t in sells)
                total_usdc = sum(float(t["quoteQty"]) for t in sells)
                if total_qty > 0:
                    sell_price = total_usdc / total_qty
                    qty_sold   = min(h.quantity, total_qty)
        except Exception:
            pass

        pnl_pct = ((sell_price / h.avg_buy_price) - 1) * 100 if h.avg_buy_price > 0 else 0.0
        color   = "green" if pnl_pct >= 0 else "red"
        console.print(
            f"[yellow]↩ Mouvement externe :[/] [bold]{sym}[/] vendu manuellement "
            f"({h.quantity:.6g} → {actual:.6g}, ~{sell_price:.4g} USDC/u) "
            f"[{color}]{pnl_pct:+.1f}%[/]"
        )
        if not dry_run:
            remaining = actual if actual > 1e-8 else 0.0
            live_upsert_holding(sym, remaining, h.avg_buy_price if remaining > 0 else 0)
            live_add_transaction(now, sym, "SELL", qty_sold, sell_price, "manual_sell")
        changes += 1

    # ── 2. Manual buys: new or increased positions on Binance ─────────────────
    for asset, actual_qty in all_actual.items():
        if asset in _IGNORED_ASSETS or asset.startswith("LD"):
            continue

        price = prices.get(asset, 0.0)
        value = actual_qty * price
        tracked = holdings.get(asset)

        if tracked is None:
            # Completely unknown asset
            if value < _MIN_MANUAL_USDC:
                continue

            buy_price  = price
            qty_bought = actual_qty
            try:
                raw_trades = get_my_trades(asset, start_ms=thirty_days_ms)
                buys = [t for t in raw_trades if t["isBuyer"]]
                if buys:
                    total_qty  = sum(float(t["qty"])      for t in buys)
                    total_usdc = sum(float(t["quoteQty"]) for t in buys)
                    if total_qty > 0:
                        buy_price  = total_usdc / total_qty
                        qty_bought = min(actual_qty, total_qty)
            except Exception:
                pass

            console.print(
                f"[bold yellow]↑ Mouvement externe :[/] [bold]{asset}[/] acheté manuellement "
                f"({qty_bought:.6g} @ ~{buy_price:.4g} ≈ {value:.2f} USDC) — ajouté au tracking"
            )
            if not dry_run:
                live_upsert_holding(asset, actual_qty, buy_price)
                live_add_transaction(now, asset, "BUY", qty_bought, buy_price, "manual_buy")
            changes += 1

        else:
            # Already tracked — check for significant top-up
            delta       = actual_qty - tracked.quantity
            delta_value = delta * price
            if delta <= tracked.quantity * 0.1 or delta_value < _MIN_MANUAL_USDC:
                continue

            buy_price = price
            try:
                raw_trades = get_my_trades(asset, start_ms=thirty_days_ms)
                buys = [t for t in raw_trades if t["isBuyer"]]
                if buys:
                    total_qty  = sum(float(t["qty"])      for t in buys)
                    total_usdc = sum(float(t["quoteQty"]) for t in buys)
                    if total_qty > 0:
                        buy_price = total_usdc / total_qty
            except Exception:
                pass

            console.print(
                f"[bold yellow]↑ Mouvement externe :[/] [bold]{asset}[/] augmenté manuellement "
                f"(+{delta:.6g} @ ~{buy_price:.4g} = +{delta_value:.2f} USDC)"
            )
            if not dry_run:
                new_total = tracked.quantity + delta
                new_avg   = (tracked.quantity * tracked.avg_buy_price + delta * buy_price) / new_total
                live_upsert_holding(asset, new_total, new_avg)
                live_add_transaction(now, asset, "BUY", delta, buy_price, "manual_buy")
            changes += 1

    return changes


def _full_reset_from_binance(dry_run: bool = False) -> int:
    """
    Wipe live_holdings and rebuild from actual Binance balances (spot + earn).
    Called at startup of live-run / live-loop to guarantee the tracking matches
    reality before the first cycle runs.

    avg_buy_price is reconstructed from the last 90 days of trade history.
    Falls back to current price when no trades are found (old/external position).

    Returns the number of positions imported.
    """
    console.print("[cyan]Réinitialisation du tracking depuis Binance…[/]")

    spot_all = get_account_balances()
    earn_all = get_earn_balances()

    all_actual: dict[str, float] = {}
    for asset, qty in spot_all.items():
        all_actual[asset] = all_actual.get(asset, 0.0) + qty
    for asset, qty in earn_all.items():
        all_actual[asset] = all_actual.get(asset, 0.0) + qty

    assets_to_price = {
        a for a, q in all_actual.items()
        if a not in _IGNORED_ASSETS and not a.startswith("LD") and q > 1e-10
    }
    prices = get_prices(list(assets_to_price)) if assets_to_price else {}

    now = _now_iso()
    ninety_days_ms = int((time.time() - 90 * 86400) * 1000)

    if not dry_run:
        live_clear_holdings()

    imported = 0
    for asset, actual_qty in sorted(all_actual.items()):
        if asset in _IGNORED_ASSETS or asset.startswith("LD"):
            continue

        price = prices.get(asset, 0.0)
        value = actual_qty * price
        if value < _MIN_MANUAL_USDC:
            continue

        buy_price = price  # fallback: current market price
        try:
            raw_trades = get_my_trades(asset, start_ms=ninety_days_ms)
            buys = [t for t in raw_trades if t["isBuyer"]]
            if buys:
                total_qty  = sum(float(t["qty"])      for t in buys)
                total_usdc = sum(float(t["quoteQty"]) for t in buys)
                if total_qty > 0:
                    buy_price = total_usdc / total_qty
        except Exception:
            pass

        console.print(
            f"  [green]↑[/] [bold]{asset}[/] {actual_qty:.6g}"
            f" @ ~{buy_price:.4g} USDC ≈ {value:.2f} USDC total"
        )
        if not dry_run:
            live_upsert_holding(asset, actual_qty, buy_price)
            live_add_transaction(now, asset, "BUY", actual_qty, buy_price, "startup_import")
        imported += 1

    console.print(
        f"[green]{imported} position(s) importée(s) depuis Binance.[/]"
        if imported else "[dim]Aucune position trouvée sur Binance.[/]"
    )
    return imported


# ── Live order execution ──────────────────────────────────────────────────────

def _live_buy(sym: str, usdc_amount: float, reason: str, now: str,
              dry_run: bool = False) -> dict | None:
    if usdc_amount < 5:
        return None

    usdc = _live_usdc()

    if usdc_amount > usdc + 0.01:
        usdc_amount = usdc
    if usdc_amount < 5:
        return None

    if dry_run:
        price = (get_prices([sym]) or {}).get(sym, 0.0)
        qty   = usdc_amount / price if price > 0 else 0.0
        return {"action": "BUY", "symbol": sym, "qty": qty, "price": price,
                "usdc_spent": usdc_amount, "reason": reason, "dry_run": True}

    try:
        _ensure_spot(QUOTE_CURRENCY, usdc_amount)
    except RuntimeError:
        console.print(f"[yellow]Earn inaccessible — BUY {sym} limité au spot disponible[/]")

    # Always cap to actual spot balance after any Earn redemption (redemption may take >3s)
    actual_spot = get_spot_balance(QUOTE_CURRENCY)
    if usdc_amount > actual_spot - 0.01:
        usdc_amount = actual_spot - 0.01
    if usdc_amount < 5:
        console.print(f"[yellow]BUY {sym} annulé : solde spot insuffisant ({actual_spot:.2f} USDC)[/]")
        return None

    try:
        avg_price, filled_qty = place_market_order(sym, "BUY", quote_qty=usdc_amount)
    except Exception as e:
        console.print(f"[red]BUY {sym} échoué : {e}[/]")
        return None

    holdings = {h.symbol: h for h in live_get_holdings()}
    if sym in holdings:
        h = holdings[sym]
        total_qty = h.quantity + filled_qty
        new_avg   = (h.quantity * h.avg_buy_price + filled_qty * avg_price) / total_qty
        live_upsert_holding(sym, total_qty, new_avg)
    else:
        live_upsert_holding(sym, filled_qty, avg_price)

    live_add_transaction(now, sym, "BUY", filled_qty, avg_price, reason)
    return {"action": "BUY", "symbol": sym, "qty": filled_qty, "price": avg_price,
            "usdc_spent": filled_qty * avg_price, "reason": reason}


def _live_sell(sym: str, reason: str, now: str, dry_run: bool = False) -> dict | None:
    holdings = {h.symbol: h for h in live_get_holdings()}
    if sym not in holdings:
        return None

    h = holdings[sym]
    step, min_qty = get_lot_size(sym)

    if dry_run:
        qty     = _floor_qty(h.quantity, step)
        price   = (get_prices([sym]) or {}).get(sym, 0.0)
        pnl_pct = ((price / h.avg_buy_price) - 1) * 100 if h.avg_buy_price > 0 else 0.0
        return {"action": "SELL", "symbol": sym, "qty": qty, "price": price,
                "proceeds": qty * price, "pnl_pct": pnl_pct, "reason": reason,
                "dry_run": True}

    # Check actual availability before attempting any Earn redemption
    spot_qty = get_spot_balance(sym)
    earn_qty = get_earn_balances().get(sym, 0.0)
    total_qty = spot_qty + earn_qty
    if total_qty * h.avg_buy_price < 1.0:
        live_upsert_holding(sym, 0, 0)
        return None
    if total_qty < h.quantity * 0.1:
        # Position was manually closed on Binance — clean up the ghost holding
        console.print(
            f"[yellow]SELL {sym} annulé : position fermée manuellement "
            f"(Binance: {spot_qty + earn_qty:.6g}, tracking: {h.quantity:.6g}). "
            f"Tracking mis à jour.[/]"
        )
        live_upsert_holding(sym, 0, 0)
        return None

    if spot_qty < h.quantity - 1e-8:
        deficit = h.quantity - spot_qty
        # Skip dust-level redemptions — Binance rejects amounts below product minimum
        if deficit / max(h.quantity, 1e-10) > 1e-4:
            console.print(f"[yellow]Spot {sym} : {spot_qty:.6g}. Rachat {deficit:.6g} depuis Earn…[/]")
            redeem_earn(sym, deficit)
            time.sleep(3)

    # Use actual spot balance after any redemption — avoids strict-equality failures
    qty = _floor_qty(get_spot_balance(sym), step)
    if qty < max(min_qty, 1e-8):
        console.print(f"[red]SELL {sym} échoué : solde spot insuffisant après rachat ({qty:.6g})[/]")
        live_upsert_holding(sym, 0, 0)
        return None

    try:
        avg_price, filled_qty = place_market_order(sym, "SELL", quantity=qty)
    except Exception as e:
        console.print(f"[red]SELL {sym} échoué : {e}[/]")
        return None

    live_upsert_holding(sym, 0, 0)
    live_add_transaction(now, sym, "SELL", filled_qty, avg_price, reason)
    pnl_pct  = ((avg_price / h.avg_buy_price) - 1) * 100 if h.avg_buy_price > 0 else 0.0
    proceeds = filled_qty * avg_price
    _record_live_pnl(pnl_pct)

    return {"action": "SELL", "symbol": sym, "qty": filled_qty, "price": avg_price,
            "proceeds": proceeds, "pnl_pct": pnl_pct, "reason": reason}


def _live_execute(actions: list, context: dict, now: str, dry_run: bool) -> list[dict]:
    executed = []
    for action in actions:
        sym    = action["symbol"].upper()
        act    = action["action"]
        reason = action.get("reason", "")
        result = None
        if act == "SELL":
            result = _live_sell(sym, reason, now, dry_run)
        elif act == "BUY":
            usdc_amount = float(action.get("usdc_amount", 0.0))
            result = _live_buy(sym, usdc_amount, reason, now, dry_run)
        if result:
            executed.append(result)
    return executed


_LIVE_BACKEND = PortfolioBackend(
    label="LIVE",
    get_usdc=_live_usdc,
    get_holdings=live_get_holdings,
    get_transactions=live_get_transactions,
    get_state=live_get_state,
    set_state=live_set_state,
    add_cycle=live_add_cycle,
    execute=_live_execute,
)


def _print_live_portfolio() -> None:
    holdings = live_get_holdings()
    usdc     = _live_usdc()
    initial  = float(live_get_state("initial_balance") or "0")
    prices   = get_prices([h.symbol for h in holdings]) if holdings else {}
    _print_portfolio_snapshot(holdings, prices, usdc, initial)


# ── Commands ──────────────────────────────────────────────────────────────────

def cmd_live_status(args) -> None:
    from rich import box
    from rich.table import Table

    init_db()
    _check_api_keys()

    holdings = live_get_holdings()
    usdc     = _live_usdc()
    initial  = float(live_get_state("initial_balance") or "0")
    started  = live_get_state("started_at", "?")
    prices   = get_prices([h.symbol for h in holdings]) if holdings else {}

    table = Table(title="[bold cyan]Live — Portefeuille réel (tracking interne)[/]",
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
        val     = h.quantity * price
        pnl_pct = ((price / h.avg_buy_price) - 1) * 100 if h.avg_buy_price > 0 and price > 0 else 0.0
        total_crypto += val
        c       = "green" if pnl_pct >= 0 else "red"
        label   = f"[bold cyan]{h.symbol} [dim](réserve)[/][/]" if h.symbol in RESERVE_CANDIDATES else h.symbol
        table.add_row(
            label, f"{h.quantity:.6g}", f"{h.avg_buy_price:.4g}",
            f"{price:.4g}", f"{val:.2f}", f"[{c}]{pnl_pct:+.1f}%[/]",
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
    if initial > 0:
        console.print(f"[dim]Référence initiale : {initial:.2f} USDC | Démarré : {started}[/]")

    cycles = live_get_cycles(limit=1)
    if cycles:
        cy = cycles[0]
        console.print(f"[dim]Dernier cycle : {cy['timestamp']} — {cy['actions_taken']} action(s)[/]")


def cmd_live_run(args) -> None:
    init_db()
    _check_api_keys()

    if not args.yes and not args.dry_run:
        console.print("\n[bold red]ATTENTION — TRADING REEL sur Binance[/]")
        console.print("[dim]Les ordres seront executés avec de l'argent réel.[/]")
        console.print("[dim]Utilise --dry-run pour simuler sans exécuter.[/]\n")
        if input("Confirmer ? [o/N] ").strip().lower() != "o":
            console.print("[dim]Annulé.[/]")
            return

    _full_reset_from_binance(dry_run=args.dry_run)

    now = _now_iso()
    live_set_state("started_at", now)
    usdc     = _live_usdc()
    holdings = live_get_holdings()
    prices   = get_prices([h.symbol for h in holdings]) if holdings else {}
    crypto   = sum(h.quantity * prices.get(h.symbol, 0) for h in holdings)
    balance  = round(usdc + crypto, 2)
    if balance > 0:
        live_set_state("initial_balance", str(balance))

    run_combined_cycle(_LIVE_BACKEND, dry_run=args.dry_run, verbose=args.verbose)


def cmd_live_loop(args) -> None:
    init_db()
    _check_api_keys()

    if not args.yes:
        console.print("\n[bold red]ATTENTION — TRADING REEL sur Binance en boucle automatique[/]")
        console.print(f"[dim]Cycle principal toutes les {args.interval} min + sous-cycle watchlist toutes les 5 min.[/]")
        console.print("[dim]Utilise --dry-run pour simuler sans exécuter.[/]\n")
        if input("Démarrer la boucle live ? [o/N] ").strip().lower() != "o":
            console.print("[dim]Annulé.[/]")
            return

    _full_reset_from_binance(dry_run=args.dry_run)

    now = _now_iso()
    live_set_state("started_at", now)
    usdc     = _live_usdc()
    holdings = live_get_holdings()
    prices   = get_prices([h.symbol for h in holdings]) if holdings else {}
    crypto   = sum(h.quantity * prices.get(h.symbol, 0) for h in holdings)
    balance  = round(usdc + crypto, 2)
    if balance > 0:
        live_set_state("initial_balance", str(balance))

    # Capture reference balance at loop launch for per-session P&L display
    _ref_usdc     = _live_usdc()
    _ref_holdings = live_get_holdings()
    _ref_prices   = get_prices([h.symbol for h in _ref_holdings]) if _ref_holdings else {}
    _ref_crypto   = sum(h.quantity * _ref_prices.get(h.symbol, 0) for h in _ref_holdings)
    _ref_total    = round(_ref_usdc + _ref_crypto, 2)
    if _ref_total > 0:
        live_set_state("loop_ref_balance", str(_ref_total))

    _SUB_INTERVAL_MIN = 5
    main_interval_secs = args.interval * 60
    sub_interval_secs  = _SUB_INTERVAL_MIN * 60

    stop_str = f" | Stop : [bold red]-{args.stop_loss:.1f}%[/]" if args.stop_loss > 0 else ""
    console.print(
        f"[bold cyan]Live loop[/] — sous-cycle [bold]{_SUB_INTERVAL_MIN} min[/] (watchlist)"
        f" + cycle principal [bold]{args.interval} min[/]. "
        f"Référence boucle : [bold]{_ref_total:.2f} USDC[/]{stop_str}. Ctrl+C pour arrêter.\n"
    )

    main_cycle = 0
    sub_cycle  = 0
    last_main_ts = 0.0  # force le cycle principal dès la première itération

    while True:
        sub_cycle += 1
        _sync_external_movements(dry_run=args.dry_run)

        if time.time() - last_main_ts >= main_interval_secs:
            last_main_ts = time.time()
            main_cycle  += 1
            console.print(f"[bold]━━ Cycle principal {main_cycle} / sous-cycle {sub_cycle} — {_now_iso()} ━━[/]")
            try:
                run_combined_cycle(_LIVE_BACKEND, verbose=args.verbose, dry_run=args.dry_run)
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                console.print(f"[red]Erreur cycle principal {main_cycle} : {exc}[/]")
        else:
            console.print(f"[bold]━━ Sous-cycle {sub_cycle} — {_now_iso()} ━━[/]")
            try:
                run_watchlist_cycle(_LIVE_BACKEND, dry_run=args.dry_run)
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                console.print(f"[red]Erreur sous-cycle {sub_cycle} : {exc}[/]")

        if args.stop_loss > 0 and balance > 0:
            current = _current_total()
            pnl_pct = (current / balance - 1) * 100
            if pnl_pct < -args.stop_loss:
                console.print(
                    f"\n[bold red]⛔ Stop journalier atteint : {pnl_pct:+.1f}% "
                    f"(seuil -{args.stop_loss:.1f}%). Loop arrêtée.[/]"
                )
                break

        console.print(f"\n[dim]Prochain sous-cycle dans {_SUB_INTERVAL_MIN} min…[/]\n")
        try:
            time.sleep(sub_interval_secs)
        except KeyboardInterrupt:
            console.print("\n[yellow]Loop arrêtée.[/]")
            break


def cmd_live_sync(args) -> None:
    """Detect and record external (manual) movements, then display portfolio."""
    init_db()
    _check_api_keys()
    n = _sync_external_movements()
    if n == 0:
        console.print("[green]Tracking cohérent avec Binance — aucun mouvement externe détecté.[/]")
    else:
        console.print(f"[yellow]{n} mouvement(s) externe(s) traité(s).[/]")
    _print_live_portfolio()


def cmd_live_history(args) -> None:
    from rich import box
    from rich.table import Table

    init_db()

    cycles = live_get_cycles(limit=args.cycles)
    if cycles:
        initial = float(live_get_state("initial_balance") or "0")
        ctable  = Table(title=f"[bold cyan]Cycles live (derniers {len(cycles)})[/]",
                        box=box.SIMPLE, title_justify="left")
        ctable.add_column("Date",           style="dim")
        ctable.add_column("Total USDC",     justify="right")
        ctable.add_column("P&L vs ref.",    justify="right")
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

    txs = live_get_transactions(limit=args.limit)
    if not txs:
        console.print("[dim]Aucune transaction live.[/]")
        return

    table = Table(title=f"[bold cyan]Transactions live (dernières {len(txs)})[/]",
                  box=box.ROUNDED, title_justify="left")
    table.add_column("Date",   style="dim")
    table.add_column("Action", justify="center")
    table.add_column("Actif",  style="bold")
    table.add_column("Qté",    justify="right")
    table.add_column("Prix",   justify="right")
    table.add_column("Raison")

    for tx in txs:
        color  = "green" if tx["tx_type"] == "BUY" else "red"
        reason = tx["reason"] or ""
        if reason in ("manual_buy", "manual_sell"):
            reason_disp = f"[yellow]{reason}[/]"
        else:
            reason_disp = reason[:70]
        table.add_row(
            tx["timestamp"][:19],
            f"[{color}]{tx['tx_type']}[/]",
            tx["symbol"], f"{tx['quantity']:.6g}", f"{tx['price']:.4g}",
            reason_disp,
        )
    console.print(table)


def cmd_live_recap(args) -> None:
    from collections import defaultdict
    from datetime import datetime, timedelta, timezone

    from rich import box
    from rich.table import Table

    init_db()
    _check_api_keys()

    tz_p2    = timezone(timedelta(hours=2))
    date_str = args.date or datetime.now(tz_p2).strftime("%d-%m-%Y")

    try:
        day_naive = datetime.strptime(date_str, "%d-%m-%Y")
    except ValueError:
        console.print(f"[red]Format invalide : {date_str}. Utilise dd-MM-YYYY (ex: 04-05-2026)[/]")
        return
    day_start = day_naive.replace(tzinfo=tz_p2)
    day_end   = day_start + timedelta(days=1)
    start_ms  = int(day_start.timestamp() * 1000)
    end_ms    = int(day_end.timestamp()   * 1000)

    def _parse(ts_str: str) -> datetime:
        dt = datetime.fromisoformat(ts_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(tz_p2)

    def _on_day(dt: datetime) -> bool:
        return day_start <= dt < day_end

    # ── Program transactions (FIFO pairing) ───────────────────────────────────
    txs = list(reversed(live_get_transactions(limit=5000)))
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
                for buy_tx in open_buys[sym]:
                    pairs.append({
                        "symbol":     sym,
                        "buy":        buy_tx,
                        "sell":       tx,
                        "sell_qty":   buy_tx["quantity"],
                        "manual":     False,
                        "manual_buy": buy_tx.get("reason") == "manual_buy",
                    })
                open_buys[sym] = []
            else:
                pairs.append({
                    "symbol":     sym,
                    "buy":        None,
                    "sell":       tx,
                    "sell_qty":   tx["quantity"],
                    "manual":     False,
                    "manual_buy": False,
                })

    for sym, buys in open_buys.items():
        for b in buys:
            pairs.append({
                "symbol": sym, "buy": b, "sell": None, "sell_qty": None,
                "manual": False, "manual_buy": b.get("reason") == "manual_buy",
            })

    filtered = [
        p for p in pairs
        if (p["buy"]  and _on_day(p["buy"]["_ts"]))
        or (p["sell"] and _on_day(p["sell"]["_ts"]))
    ]

    # ── Detect manually closed positions and fetch real Binance trades ─────────
    spot_all = get_account_balances()
    earn_all = get_earn_balances()

    open_prog = [p for p in filtered if p["sell"] is None and p["buy"] is not None]
    for p in open_prog:
        sym      = p["symbol"]
        buy_tx   = p["buy"]
        actual   = spot_all.get(sym, 0.0) + earn_all.get(sym, 0.0)

        if actual >= buy_tx["quantity"] * 0.1:
            continue  # still open, nothing to do

        # Position was manually closed — fetch Binance trade history
        raw_trades = get_my_trades(sym, start_ms=int(buy_tx["_ts"].timestamp() * 1000), end_ms=end_ms)
        sells = [t for t in raw_trades if not t["isBuyer"]]

        if sells:
            # Aggregate all SELL trades for this symbol (could be split orders)
            total_qty  = sum(float(t["qty"])      for t in sells)
            total_usdc = sum(float(t["quoteQty"]) for t in sells)
            avg_price  = total_usdc / total_qty if total_qty else 0.0
            last_ts    = datetime.fromtimestamp(
                max(t["time"] for t in sells) / 1000, tz=timezone.utc
            ).astimezone(tz_p2)
            p["sell"]   = {
                "price":    avg_price,
                "quantity": total_qty,
                "_ts":      last_ts,
            }
        else:
            # No sell trade found — mark as manually closed at unknown price
            p["sell"] = {"price": None, "quantity": buy_tx["quantity"], "_ts": None}

        p["manual"] = True

    # ── Build display rows ────────────────────────────────────────────────────
    cur_prices = get_prices(
        list({p["symbol"] for p in filtered if p["sell"] is None})
    )

    rows = []
    for p in filtered:
        buy_tx  = p["buy"]
        sell_tx = p["sell"]
        sym     = p["symbol"]
        manual  = p["manual"]

        buy_price  = buy_tx["price"]    if buy_tx  else None
        sell_price = sell_tx["price"]   if sell_tx else None
        cur_price  = cur_prices.get(sym)

        buy_qty      = buy_tx["quantity"] if buy_tx else None
        # sell_qty is the proportional slice (= buy_qty when sell covered multiple buys)
        eff_sell_qty = p.get("sell_qty")
        buy_usdc   = buy_qty      * buy_price  if buy_qty      and buy_price  else None
        sell_usdc  = eff_sell_qty * sell_price if eff_sell_qty and sell_price else None
        cur_usdc   = buy_qty      * cur_price  if buy_qty      and cur_price and not sell_tx else None

        ref_price  = sell_price if sell_tx else cur_price
        ref_usdc   = sell_usdc  if sell_tx else cur_usdc
        pnl_pct    = ((ref_price / buy_price) - 1) * 100 if buy_price and ref_price else None
        delta_usdc = (ref_usdc - buy_usdc) if ref_usdc is not None and buy_usdc is not None else None

        rows.append({
            "symbol":     sym,
            "buy_ts":     buy_tx["_ts"].strftime("%H:%M") if buy_tx  else None,
            "buy_price":  buy_price,
            "buy_usdc":   buy_usdc,
            "sell_ts":    (sell_tx["_ts"].strftime("%H:%M") if sell_tx["_ts"] else "?") if sell_tx else None,
            "sell_price": sell_price,
            "sell_usdc":  sell_usdc,
            "cur_price":  cur_price if not sell_tx else None,
            "cur_usdc":   cur_usdc,
            "delta_usdc": delta_usdc,
            "pnl_pct":    pnl_pct,
            "open":       sell_tx is None,
            "manual":     manual,
            "manual_buy": p.get("manual_buy", False),
        })

    rows.sort(key=lambda r: r["delta_usdc"] if r["delta_usdc"] is not None else float("-inf"), reverse=True)

    # ── Render table ─────────────────────────────────────────────────────────
    table = Table(
        title=f"[bold cyan]Récapitulatif {date_str} (GMT+2)[/]",
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
        sym_label = r["symbol"]
        markers   = []
        if r.get("manual_buy"):  markers.append("[yellow]↑[/]")
        if r["manual"]:          markers.append("[dim]✎[/]")
        if markers:
            sym_label = sym_label + " " + " ".join(markers)
        buy_p  = f"{r['buy_price']:.6g}"  if r["buy_price"]  else "—"
        buy_u  = f"{r['buy_usdc']:.2f}"   if r["buy_usdc"]   else "—"
        buy_ts = r["buy_ts"] or "—"

        if r["open"]:
            sell_ts = "[dim]ouvert[/]"
            if r["cur_price"]:
                sell_p = f"[dim]{r['cur_price']:.6g}[/]"
                sell_u = f"[dim]{r['cur_usdc']:.2f}[/]"
            else:
                sell_p = sell_u = "—"
        elif r["sell_price"]:
            tag    = " [dim]✎[/]" if r["manual"] else ""
            sell_ts = (r["sell_ts"] or "?") + tag
            sell_p = f"{r['sell_price']:.6g}"
            sell_u = f"{r['sell_usdc']:.2f}" if r["sell_usdc"] else "—"
        else:
            sell_ts = "[dim]✎ inconnu[/]"
            sell_p  = sell_u = "—"

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

        table.add_row(sym_label, buy_ts, buy_p, buy_u, sell_ts, sell_p, sell_u, delta_s, pnl_str)

    console.print(table)
    legend = []
    if any(r.get("manual_buy") for r in rows): legend.append("[yellow]↑[/] achat manuel")
    if any(r["manual"]         for r in rows): legend.append("[dim]✎[/] clôture manuelle")
    if legend:
        console.print("[dim]" + " | ".join(legend) + "[/]")

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
        avg_pct   = sum(r["pnl_pct"] for r in closed) / len(closed)
        n_manual  = sum(1 for r in closed if r["manual"] or r.get("manual_buy"))
        man_s     = f"  [dim]{n_manual} manuels[/]" if n_manual else ""
        console.print(
            f"  Clôturés ({len(closed)}){man_s}   "
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
    all_cycles   = live_get_cycles(limit=2000)
    pre_day      = [cy for cy in all_cycles if _parse(cy["timestamp"]) < day_start]
    day_cycs     = [cy for cy in all_cycles if _on_day(_parse(cy["timestamp"]))]
    start_val    = (pre_day[0]["total_value"]   if pre_day  else
                    day_cycs[-1]["total_value"]  if day_cycs else None)

    usdc_now     = spot_all.get(QUOTE_CURRENCY, 0.0) + earn_all.get(QUOTE_CURRENCY, 0.0)
    holdings_now = live_get_holdings()
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
    sub.add_parser("live-status", help="Afficher le portefeuille live (tracking interne)")

    p = sub.add_parser("live-run", help="Un cycle live (scan pump Tier 2)")
    p.add_argument("--verbose",  action="store_true")
    p.add_argument("--dry-run",  action="store_true", dest="dry_run",
                   help="Simuler les ordres sans les exécuter")
    p.add_argument("--yes", "-y", action="store_true",
                   help="Ignorer la confirmation interactive")

    p = sub.add_parser("live-loop", help="Boucle automatique live (pump Tier 2 toutes les N min)")
    p.add_argument("--interval",  type=int, default=15, metavar="MIN",
                   help="Intervalle pump en minutes (defaut : 15)")
    p.add_argument("--stop-loss", type=float, default=5.0, metavar="PCT", dest="stop_loss",
                   help="Arrêter si P&L session < -PCT%% (défaut : 5.0 ; 0 = désactivé)")
    p.add_argument("--verbose",   action="store_true")
    p.add_argument("--dry-run",   action="store_true", dest="dry_run")
    p.add_argument("--yes", "-y", action="store_true",
                   help="Ignorer la confirmation de démarrage")

    p = sub.add_parser("live-history", help="Historique des transactions et cycles live")
    p.add_argument("--limit",  type=int, default=30)
    p.add_argument("--cycles", type=int, default=10)

    p = sub.add_parser("live-recap",
                       help="Récapitulatif des trades d'une journée (dd-MM-YYYY, défaut : aujourd'hui)")
    p.add_argument("date", nargs="?", default=None,
                   help="Date au format dd-MM-YYYY (ex: 04-05-2026, défaut : aujourd'hui)")

    sub.add_parser("live-sync",
                   help="Réconcilier le tracking interne avec les balances réelles Binance")

    return {
        "live-status":  cmd_live_status,
        "live-run":     cmd_live_run,
        "live-loop":    cmd_live_loop,
        "live-sync":    cmd_live_sync,
        "live-history": cmd_live_history,
        "live-recap":   cmd_live_recap,
    }
