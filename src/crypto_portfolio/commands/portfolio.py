import sys

from ..binance import get_prices as binance_get_prices
from ..config import QUOTE_CURRENCY
from ..display import build_prices_table, console, render_portfolio
from ..portfolio import buy, get_portfolio, sell
from ..storage import get_transactions, init_db

import time
from ..config import REFRESH_INTERVAL


def cmd_watch(_args) -> None:
    init_db()
    try:
        holdings = get_portfolio()
        render_portfolio(holdings)
    except Exception as e:
        console.print(f"[red]Erreur : {e}[/]")


def cmd_prices(args) -> None:
    from rich.live import Live

    symbols = [s.upper() for s in args.symbols]
    prev_prices: dict[str, float] = {}

    console.print("[dim]Appuie sur Ctrl+C pour arrêter[/]\n")
    with Live(console=console, refresh_per_second=4, screen=False) as live:
        try:
            while True:
                try:
                    prices = binance_get_prices(symbols)
                    live.update(build_prices_table(prices, prev_prices))
                    prev_prices = prices
                except Exception as e:
                    live.update(f"[red]Erreur Binance : {e}[/]")
                time.sleep(REFRESH_INTERVAL)
        except KeyboardInterrupt:
            pass


def cmd_buy(args) -> None:
    init_db()
    buy(args.symbol, args.quantity, args.price)
    console.print(
        f"[green]Achat enregistré : {args.quantity} {args.symbol.upper()} "
        f"@ {args.price:,.4f} {QUOTE_CURRENCY}[/]"
    )


def cmd_sell(args) -> None:
    init_db()
    try:
        sell(args.symbol, args.quantity, args.price)
        console.print(
            f"[green]Vente enregistrée : {args.quantity} {args.symbol.upper()} "
            f"@ {args.price:,.4f} {QUOTE_CURRENCY}[/]"
        )
    except ValueError as e:
        console.print(f"[red]{e}[/]")
        sys.exit(1)


def cmd_sync(args) -> None:
    from ..binance import get_account_balances, get_earn_balances, has_api_keys
    from ..models import Holding
    from ..storage import delete_holding, get_holding, upsert_holding

    init_db()
    if not has_api_keys():
        console.print("[red]Aucune clé API configurée. Lance d'abord : crypto-portfolio setup-keys[/]")
        sys.exit(1)

    ignore = {args.quote or QUOTE_CURRENCY, "USDT", "USDC", "BUSD", "EUR"}
    spot = get_account_balances()
    earn = get_earn_balances()

    merged: dict[str, float] = {}
    for asset, qty in spot.items():
        if asset in ignore or qty < 1e-8:
            continue
        underlying = asset[2:] if asset.startswith("LD") else None
        if underlying and underlying in earn:
            delete_holding(asset)
            continue
        merged[asset] = qty

    for asset, qty in earn.items():
        if asset in ignore or qty < 1e-8:
            continue
        merged[asset] = merged.get(asset, 0.0) + qty

    updated = 0
    for asset, qty in merged.items():
        existing = get_holding(asset)
        avg = existing.avg_buy_price if existing else 0.0
        upsert_holding(Holding(symbol=asset, quantity=qty, avg_buy_price=avg))
        updated += 1

    console.print(f"[green]{updated} position(s) synchronisée(s) depuis Binance.[/]")
    if updated:
        console.print("[yellow]Note : les prix d'achat moyens ne sont pas disponibles via l'API Binance.[/]")


def cmd_history(args) -> None:
    from rich import box
    from rich.table import Table

    init_db()
    txs = get_transactions(args.symbol)
    if not txs:
        console.print("[yellow]Aucune transaction trouvée.[/]")
        return

    table = Table(box=box.SIMPLE)
    table.add_column("Date")
    table.add_column("Actif", style="bold")
    table.add_column("Type")
    table.add_column("Quantité", justify="right")
    table.add_column(f"Prix ({QUOTE_CURRENCY})", justify="right")
    table.add_column(f"Total ({QUOTE_CURRENCY})", justify="right")

    for tx in txs:
        color = "green" if tx.tx_type == "buy" else "red"
        table.add_row(
            tx.timestamp.strftime("%Y-%m-%d %H:%M"),
            tx.symbol,
            f"[{color}]{tx.tx_type.upper()}[/]",
            f"{tx.quantity:.8f}".rstrip("0").rstrip("."),
            f"{tx.price:,.4f}",
            f"{tx.quantity * tx.price:,.2f}",
        )
    console.print(table)


def register(sub):
    sub.add_parser("watch", help="Afficher le portefeuille (one-shot)")

    p = sub.add_parser("prices", help="Cours live de symboles Binance")
    p.add_argument("symbols", nargs="+", help="Symboles (ex: BTC ETH SOL)")

    p = sub.add_parser("buy", help="Enregistrer un achat manuel")
    p.add_argument("symbol")
    p.add_argument("quantity", type=float)
    p.add_argument("price", type=float, help=f"Prix en {QUOTE_CURRENCY}")

    p = sub.add_parser("sell", help="Enregistrer une vente manuelle")
    p.add_argument("symbol")
    p.add_argument("quantity", type=float)
    p.add_argument("price", type=float, help=f"Prix en {QUOTE_CURRENCY}")

    p = sub.add_parser("sync", help="Synchroniser les balances depuis Binance")
    p.add_argument("--quote", help="Stablecoin à exclure")

    p = sub.add_parser("history", help="Historique des transactions")
    p.add_argument("symbol", nargs="?", help="Filtrer par symbole")

    return {
        "watch": cmd_watch,
        "prices": cmd_prices,
        "buy": cmd_buy,
        "sell": cmd_sell,
        "sync": cmd_sync,
        "history": cmd_history,
    }
