from datetime import datetime

from rich import box
from rich.console import Console
from rich.table import Table
from rich.text import Text

from .config import QUOTE_CURRENCY
from .models import PricedHolding

console = Console()


def render_portfolio(holdings: list[PricedHolding], last_error: str | None = None) -> None:
    console.print(
        f"[bold cyan]Crypto Portfolio[/] — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
        f"| Référence : [bold]{QUOTE_CURRENCY}[/]\n"
    )
    if last_error:
        console.print(f"[red]Erreur : {last_error}[/]\n")

    if not holdings:
        console.print("[yellow]Aucune position. Utilise la commande 'buy' pour en ajouter.[/]")
        return

    table = Table(box=box.ROUNDED, show_footer=True, footer_style="bold")
    table.add_column("Actif", style="bold")
    table.add_column("Quantité", justify="right")
    table.add_column(f"Achat ({QUOTE_CURRENCY})", justify="right")
    table.add_column(f"Prix actuel ({QUOTE_CURRENCY})", justify="right")
    table.add_column(f"Valeur ({QUOTE_CURRENCY})", justify="right")
    table.add_column("P&L", justify="right")
    table.add_column("P&L %", justify="right")

    total_value = sum(h.current_value for h in holdings)
    total_pnl = sum(h.pnl for h in holdings)

    for h in sorted(holdings, key=lambda x: x.current_value, reverse=True):
        pnl_color = "green" if h.pnl >= 0 else "red"
        qty_str = f"{h.holding.quantity:.8f}".rstrip("0").rstrip(".")
        table.add_row(
            h.holding.symbol,
            qty_str,
            f"{h.holding.avg_buy_price:,.4f}",
            f"{h.current_price:,.4f}",
            f"{h.current_value:,.2f}",
            f"[{pnl_color}]{h.pnl:+,.2f}[/]",
            f"[{pnl_color}]{h.pnl_pct:+.1f}%[/]",
        )

    pnl_color = "green" if total_pnl >= 0 else "red"
    table.columns[4].footer = f"{total_value:,.2f}"
    table.columns[5].footer = f"[{pnl_color}]{total_pnl:+,.2f}[/]"

    console.print(table)


def build_prices_table(prices: dict[str, float], prev_prices: dict[str, float]) -> Table:
    table = Table(
        box=box.ROUNDED,
        title=f"[bold cyan]Cours Binance[/] — {datetime.now().strftime('%H:%M:%S')} | {QUOTE_CURRENCY}",
        title_justify="left",
    )
    table.add_column("Actif", style="bold", min_width=8)
    table.add_column(f"Prix ({QUOTE_CURRENCY})", justify="right", min_width=16)
    table.add_column("Variation (tick)", justify="right", min_width=14)

    for symbol in sorted(prices):
        price = prices[symbol]
        prev = prev_prices.get(symbol, price)
        diff_pct = ((price - prev) / prev * 100) if prev else 0.0
        if diff_pct > 0:
            trend = Text(f"+{diff_pct:.4f}%", style="green")
        elif diff_pct < 0:
            trend = Text(f"{diff_pct:.4f}%", style="red")
        else:
            trend = Text("—", style="dim")
        table.add_row(symbol, f"{price:,.4f}", trend)

    return table
