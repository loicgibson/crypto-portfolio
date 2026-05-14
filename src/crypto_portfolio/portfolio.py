from .binance import get_prices
from .models import Holding, PricedHolding, Transaction
from .storage import add_transaction, get_holding, get_holdings, upsert_holding


def get_portfolio() -> list[PricedHolding]:
    holdings = get_holdings()
    if not holdings:
        return []
    prices = get_prices([h.symbol for h in holdings])
    return [
        PricedHolding(holding=h, current_price=prices.get(h.symbol, 0.0))
        for h in holdings
    ]


def buy(symbol: str, quantity: float, price: float) -> None:
    symbol = symbol.upper()
    existing = get_holding(symbol)
    if existing:
        total_qty = existing.quantity + quantity
        avg_price = (existing.quantity * existing.avg_buy_price + quantity * price) / total_qty
        holding = Holding(symbol=symbol, quantity=total_qty, avg_buy_price=avg_price)
    else:
        holding = Holding(symbol=symbol, quantity=quantity, avg_buy_price=price)
    upsert_holding(holding)
    add_transaction(Transaction(symbol=symbol, tx_type="buy", quantity=quantity, price=price))


def sell(symbol: str, quantity: float, price: float) -> None:
    symbol = symbol.upper()
    existing = get_holding(symbol)
    if not existing:
        raise ValueError(f"Aucune position ouverte pour {symbol}")
    if quantity > existing.quantity:
        raise ValueError(f"Impossible de vendre {quantity}, tu n'as que {existing.quantity} {symbol}")
    new_qty = existing.quantity - quantity
    upsert_holding(Holding(symbol=symbol, quantity=new_qty, avg_buy_price=existing.avg_buy_price))
    add_transaction(Transaction(symbol=symbol, tx_type="sell", quantity=quantity, price=price))
