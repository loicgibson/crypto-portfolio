import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Iterator

from .config import DB_PATH
from .models import Holding, Transaction


@contextmanager
def _conn() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with _conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS holdings (
                symbol TEXT PRIMARY KEY,
                quantity REAL NOT NULL,
                avg_buy_price REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                tx_type TEXT NOT NULL,
                quantity REAL NOT NULL,
                price REAL NOT NULL,
                timestamp TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS excluded_symbols (
                symbol TEXT PRIMARY KEY
            );
            CREATE TABLE IF NOT EXISTS klines (
                symbol TEXT NOT NULL,
                interval TEXT NOT NULL,
                open_time INTEGER NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL,
                close_time INTEGER NOT NULL,
                quote_volume REAL NOT NULL,
                num_trades INTEGER NOT NULL,
                PRIMARY KEY (symbol, interval, open_time)
            );
            CREATE TABLE IF NOT EXISTS funding_rates (
                symbol TEXT NOT NULL,
                funding_time INTEGER NOT NULL,
                rate REAL NOT NULL,
                PRIMARY KEY (symbol, funding_time)
            );
            CREATE TABLE IF NOT EXISTS sim_state (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sim_holdings (
                symbol        TEXT PRIMARY KEY,
                quantity      REAL NOT NULL,
                avg_buy_price REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sim_transactions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp   TEXT NOT NULL,
                symbol      TEXT NOT NULL,
                tx_type     TEXT NOT NULL,
                quantity    REAL NOT NULL,
                price       REAL NOT NULL,
                usdc_before REAL NOT NULL,
                usdc_after  REAL NOT NULL,
                reason      TEXT
            );
            CREATE TABLE IF NOT EXISTS sim_cycles (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp      TEXT NOT NULL,
                usdc_balance   REAL NOT NULL,
                total_value    REAL NOT NULL,
                actions_taken  INTEGER NOT NULL,
                market_summary TEXT
            );
            CREATE TABLE IF NOT EXISTS inactive_symbols (
                symbol     TEXT PRIMARY KEY,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS trading_symbols (
                symbol     TEXT PRIMARY KEY,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS app_state (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS live_state (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS live_holdings (
                symbol        TEXT PRIMARY KEY,
                quantity      REAL NOT NULL,
                avg_buy_price REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS live_transactions (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol    TEXT NOT NULL,
                tx_type   TEXT NOT NULL,
                quantity  REAL NOT NULL,
                price     REAL NOT NULL,
                reason    TEXT
            );
            CREATE TABLE IF NOT EXISTS live_cycles (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp      TEXT NOT NULL,
                usdc_balance   REAL NOT NULL,
                total_value    REAL NOT NULL,
                actions_taken  INTEGER NOT NULL,
                market_summary TEXT
            );
        """)


def get_holdings() -> list[Holding]:
    with _conn() as conn:
        rows = conn.execute("SELECT * FROM holdings WHERE quantity > 0").fetchall()
        return [Holding(symbol=r["symbol"], quantity=r["quantity"], avg_buy_price=r["avg_buy_price"]) for r in rows]


def get_holding(symbol: str) -> Holding | None:
    with _conn() as conn:
        row = conn.execute("SELECT * FROM holdings WHERE symbol = ?", (symbol.upper(),)).fetchone()
        return Holding(symbol=row["symbol"], quantity=row["quantity"], avg_buy_price=row["avg_buy_price"]) if row else None


def upsert_holding(holding: Holding) -> None:
    with _conn() as conn:
        conn.execute("""
            INSERT INTO holdings (symbol, quantity, avg_buy_price)
            VALUES (?, ?, ?)
            ON CONFLICT(symbol) DO UPDATE SET
                quantity = excluded.quantity,
                avg_buy_price = excluded.avg_buy_price
        """, (holding.symbol, holding.quantity, holding.avg_buy_price))


def delete_holding(symbol: str) -> None:
    with _conn() as conn:
        conn.execute("DELETE FROM holdings WHERE symbol = ?", (symbol.upper(),))


def add_transaction(tx: Transaction) -> None:
    with _conn() as conn:
        conn.execute("""
            INSERT INTO transactions (symbol, tx_type, quantity, price, timestamp)
            VALUES (?, ?, ?, ?, ?)
        """, (tx.symbol, tx.tx_type, tx.quantity, tx.price, tx.timestamp.isoformat()))


def add_excluded(symbols: list[str]) -> None:
    with _conn() as conn:
        conn.executemany(
            "INSERT OR IGNORE INTO excluded_symbols (symbol) VALUES (?)",
            [(s.upper(),) for s in symbols],
        )


def remove_excluded(symbols: list[str]) -> None:
    with _conn() as conn:
        conn.executemany(
            "DELETE FROM excluded_symbols WHERE symbol = ?",
            [(s.upper(),) for s in symbols],
        )


def get_excluded() -> set[str]:
    with _conn() as conn:
        rows = conn.execute("SELECT symbol FROM excluded_symbols").fetchall()
        return {r["symbol"] for r in rows}


def upsert_klines(symbol: str, interval: str, rows: list[list]) -> int:
    with _conn() as conn:
        conn.executemany("""
            INSERT OR IGNORE INTO klines
                (symbol, interval, open_time, open, high, low, close, volume, close_time, quote_volume, num_trades)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            (symbol.upper(), interval,
             int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]),
             float(r[5]), int(r[6]), float(r[7]), int(r[8]))
            for r in rows
        ])
        return conn.total_changes


def get_last_kline_time(symbol: str, interval: str) -> int | None:
    with _conn() as conn:
        row = conn.execute(
            "SELECT MAX(open_time) FROM klines WHERE symbol = ? AND interval = ?",
            (symbol.upper(), interval),
        ).fetchone()
        return row[0] if row and row[0] is not None else None


def upsert_funding_rates(symbol: str, rates: list[dict]) -> int:
    with _conn() as conn:
        conn.executemany(
            "INSERT OR IGNORE INTO funding_rates (symbol, funding_time, rate) VALUES (?, ?, ?)",
            [(symbol.upper(), int(r["fundingTime"]), float(r["fundingRate"])) for r in rates],
        )
        return conn.total_changes


def get_last_funding_time(symbol: str) -> int | None:
    with _conn() as conn:
        row = conn.execute(
            "SELECT MAX(funding_time) FROM funding_rates WHERE symbol = ?",
            (symbol.upper(),),
        ).fetchone()
        return row[0] if row and row[0] is not None else None


def get_funding_df(symbol: str):
    import pandas as pd
    with _conn() as conn:
        rows = conn.execute(
            "SELECT funding_time, rate FROM funding_rates WHERE symbol = ? ORDER BY funding_time",
            (symbol.upper(),),
        ).fetchall()
    if not rows:
        return pd.DataFrame(columns=["funding_time", "rate"])
    return pd.DataFrame(rows, columns=["funding_time", "rate"])


# ── Simulation (paper trading) ────────────────────────────────────────────────

def sim_get_state(key: str, default: str | None = None) -> str | None:
    with _conn() as conn:
        row = conn.execute("SELECT value FROM sim_state WHERE key = ?", (key,)).fetchone()
        return row[0] if row else default


def sim_set_state(key: str, value: str) -> None:
    with _conn() as conn:
        conn.execute(
            "INSERT INTO sim_state (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )


def sim_get_usdc() -> float:
    v = sim_get_state("usdc_balance")
    return float(v) if v is not None else 0.0


def sim_set_usdc(amount: float) -> None:
    sim_set_state("usdc_balance", str(round(amount, 8)))


def sim_get_holdings() -> list[Holding]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM sim_holdings WHERE quantity > 0"
        ).fetchall()
        return [Holding(symbol=r["symbol"], quantity=r["quantity"],
                        avg_buy_price=r["avg_buy_price"]) for r in rows]


def sim_upsert_holding(symbol: str, quantity: float, avg_buy_price: float) -> None:
    with _conn() as conn:
        if quantity <= 1e-8:
            conn.execute("DELETE FROM sim_holdings WHERE symbol = ?", (symbol.upper(),))
        else:
            conn.execute(
                "INSERT INTO sim_holdings (symbol, quantity, avg_buy_price) VALUES (?, ?, ?) "
                "ON CONFLICT(symbol) DO UPDATE SET "
                "quantity = excluded.quantity, avg_buy_price = excluded.avg_buy_price",
                (symbol.upper(), quantity, avg_buy_price),
            )


def sim_add_transaction(timestamp: str, symbol: str, tx_type: str,
                        quantity: float, price: float,
                        usdc_before: float, usdc_after: float,
                        reason: str = "") -> None:
    with _conn() as conn:
        conn.execute(
            "INSERT INTO sim_transactions "
            "(timestamp, symbol, tx_type, quantity, price, usdc_before, usdc_after, reason) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (timestamp, symbol.upper(), tx_type, quantity, price,
             usdc_before, usdc_after, reason),
        )


def sim_add_cycle(timestamp: str, usdc_balance: float, total_value: float,
                  actions_taken: int, market_summary: str = "") -> None:
    with _conn() as conn:
        conn.execute(
            "INSERT INTO sim_cycles "
            "(timestamp, usdc_balance, total_value, actions_taken, market_summary) "
            "VALUES (?, ?, ?, ?, ?)",
            (timestamp, usdc_balance, total_value, actions_taken, market_summary),
        )


def sim_get_transactions(limit: int = 50) -> list[dict]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM sim_transactions ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


def sim_get_cycles(limit: int = 20) -> list[dict]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM sim_cycles ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


def app_get_state(key: str, default: str | None = None) -> str | None:
    with _conn() as conn:
        row = conn.execute("SELECT value FROM app_state WHERE key = ?", (key,)).fetchone()
        return row[0] if row else default


def app_set_state(key: str, value: str) -> None:
    with _conn() as conn:
        conn.execute(
            "INSERT INTO app_state (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )


def get_inactive_symbols() -> set[str]:
    with _conn() as conn:
        rows = conn.execute("SELECT symbol FROM inactive_symbols").fetchall()
        return {r[0] for r in rows}


def set_inactive_symbols(symbols: set[str], updated_at: str) -> None:
    with _conn() as conn:
        conn.execute("DELETE FROM inactive_symbols")
        conn.executemany(
            "INSERT INTO inactive_symbols (symbol, updated_at) VALUES (?, ?)",
            [(s.upper(), updated_at) for s in symbols],
        )


def get_trading_symbols() -> set[str]:
    with _conn() as conn:
        rows = conn.execute("SELECT symbol FROM trading_symbols").fetchall()
        return {r[0] for r in rows}


def set_trading_symbols(symbols: set[str], updated_at: str) -> None:
    with _conn() as conn:
        conn.execute("DELETE FROM trading_symbols")
        conn.executemany(
            "INSERT INTO trading_symbols (symbol, updated_at) VALUES (?, ?)",
            [(s.upper(), updated_at) for s in symbols],
        )


# ── Live trading ──────────────────────────────────────────────────────────────

def live_get_state(key: str, default: str | None = None) -> str | None:
    with _conn() as conn:
        row = conn.execute("SELECT value FROM live_state WHERE key = ?", (key,)).fetchone()
        return row[0] if row else default


def live_set_state(key: str, value: str) -> None:
    with _conn() as conn:
        conn.execute(
            "INSERT INTO live_state (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )


def live_get_holdings() -> list[Holding]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM live_holdings WHERE quantity > 0"
        ).fetchall()
        return [Holding(symbol=r["symbol"], quantity=r["quantity"],
                        avg_buy_price=r["avg_buy_price"]) for r in rows]


def live_upsert_holding(symbol: str, quantity: float, avg_buy_price: float) -> None:
    with _conn() as conn:
        if quantity <= 1e-8:
            conn.execute("DELETE FROM live_holdings WHERE symbol = ?", (symbol.upper(),))
        else:
            conn.execute(
                "INSERT INTO live_holdings (symbol, quantity, avg_buy_price) VALUES (?, ?, ?) "
                "ON CONFLICT(symbol) DO UPDATE SET "
                "quantity = excluded.quantity, avg_buy_price = excluded.avg_buy_price",
                (symbol.upper(), quantity, avg_buy_price),
            )


def live_add_transaction(timestamp: str, symbol: str, tx_type: str,
                         quantity: float, price: float, reason: str = "") -> None:
    with _conn() as conn:
        conn.execute(
            "INSERT INTO live_transactions "
            "(timestamp, symbol, tx_type, quantity, price, reason) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (timestamp, symbol.upper(), tx_type, quantity, price, reason),
        )


def live_get_transactions(limit: int = 50) -> list[dict]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM live_transactions ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


def live_add_cycle(timestamp: str, usdc_balance: float, total_value: float,
                   actions_taken: int, market_summary: str = "") -> None:
    with _conn() as conn:
        conn.execute(
            "INSERT INTO live_cycles "
            "(timestamp, usdc_balance, total_value, actions_taken, market_summary) "
            "VALUES (?, ?, ?, ?, ?)",
            (timestamp, usdc_balance, total_value, actions_taken, market_summary),
        )


def live_get_cycles(limit: int = 20) -> list[dict]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM live_cycles ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


def sim_reset() -> None:
    with _conn() as conn:
        conn.executescript("""
            DELETE FROM sim_state;
            DELETE FROM sim_holdings;
            DELETE FROM sim_transactions;
            DELETE FROM sim_cycles;
        """)


# ── Real transactions ──────────────────────────────────────────────────────────

def get_transactions(symbol: str | None = None) -> list[Transaction]:
    with _conn() as conn:
        if symbol:
            rows = conn.execute(
                "SELECT * FROM transactions WHERE symbol = ? ORDER BY timestamp DESC",
                (symbol.upper(),),
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM transactions ORDER BY timestamp DESC").fetchall()
        return [
            Transaction(
                symbol=r["symbol"], tx_type=r["tx_type"],
                quantity=r["quantity"], price=r["price"],
                timestamp=datetime.fromisoformat(r["timestamp"]), id=r["id"],
            )
            for r in rows
        ]
