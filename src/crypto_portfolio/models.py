from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Holding:
    symbol: str
    quantity: float
    avg_buy_price: float


@dataclass
class PricedHolding:
    holding: Holding
    current_price: float

    @property
    def current_value(self) -> float:
        return self.holding.quantity * self.current_price

    @property
    def cost_basis(self) -> float:
        return self.holding.quantity * self.holding.avg_buy_price

    @property
    def pnl(self) -> float:
        return self.current_value - self.cost_basis

    @property
    def pnl_pct(self) -> float:
        if self.cost_basis == 0:
            return 0.0
        return (self.pnl / self.cost_basis) * 100


@dataclass
class Transaction:
    symbol: str
    tx_type: str  # "buy" | "sell"
    quantity: float
    price: float
    timestamp: datetime = field(default_factory=datetime.now)
    id: int | None = None
