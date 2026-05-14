import argparse

from .commands import build_parser_and_handlers
from .display import console
from .portfolio import get_portfolio
from .storage import init_db


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="crypto-portfolio",
        description="Gestionnaire de portefeuille crypto — Binance",
    )
    sub = parser.add_subparsers(dest="command")
    handlers = build_parser_and_handlers(sub)

    args = parser.parse_args()

    if args.command is None:
        init_db()
        try:
            from .display import render_portfolio
            render_portfolio(get_portfolio())
        except Exception as e:
            console.print(f"[red]Erreur : {e}[/]")
        return

    handler = handlers.get(args.command)
    if handler is None:
        parser.print_help()
        return

    handler(args)


if __name__ == "__main__":
    main()
