import hashlib
import hmac
import json
import math
import time

import requests

from .config import BINANCE_API_KEY, BINANCE_API_SECRET, BINANCE_BASE_URL, QUOTE_CURRENCY


def _raise(resp: requests.Response) -> None:
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        try:
            detail = resp.json()
            msg = detail.get("msg", resp.text)
            code = detail.get("code", "")
            raise requests.HTTPError(f"{e} — Binance [{code}] {msg}", response=resp) from None
        except (ValueError, KeyError):
            raise


def _get(endpoint: str, params: dict | None = None) -> dict | list:
    resp = requests.get(f"{BINANCE_BASE_URL}{endpoint}", params=params, timeout=10)
    _raise(resp)
    return resp.json()


def _signed_post(endpoint: str, params: dict | None = None) -> dict:
    p = dict(params or {})
    p["timestamp"] = int(time.time() * 1000)
    query = "&".join(f"{k}={v}" for k, v in p.items())
    sig = hmac.new(BINANCE_API_SECRET.encode(), query.encode(), hashlib.sha256).hexdigest()
    p["signature"] = sig
    resp = requests.post(
        f"{BINANCE_BASE_URL}{endpoint}",
        data=p,
        headers={"X-MBX-APIKEY": BINANCE_API_KEY},
        timeout=10,
    )
    _raise(resp)
    return resp.json()


def _signed_get(endpoint: str, params: dict | None = None) -> dict | list:
    p = dict(params or {})
    p["timestamp"] = int(time.time() * 1000)
    query = "&".join(f"{k}={v}" for k, v in p.items())
    sig = hmac.new(BINANCE_API_SECRET.encode(), query.encode(), hashlib.sha256).hexdigest()
    p["signature"] = sig
    resp = requests.get(
        f"{BINANCE_BASE_URL}{endpoint}",
        params=p,
        headers={"X-MBX-APIKEY": BINANCE_API_KEY},
        timeout=10,
    )
    _raise(resp)
    return resp.json()


def has_api_keys() -> bool:
    return bool(BINANCE_API_KEY and BINANCE_API_SECRET)


def get_prices(symbols: list[str]) -> dict[str, float]:
    if not symbols:
        return {}
    pairs = [f"{s}{QUOTE_CURRENCY}" for s in symbols]
    try:
        data = _get("/api/v3/ticker/price", {"symbols": json.dumps(pairs, separators=(",", ":"))})
        return {
            item["symbol"].removesuffix(QUOTE_CURRENCY): float(item["price"])
            for item in data
        }
    except Exception:
        result = {}
        for symbol, pair in zip(symbols, pairs):
            try:
                data = _get("/api/v3/ticker/price", {"symbol": pair})
                result[symbol] = float(data["price"])
            except Exception:
                pass
        return result


def get_account_balances() -> dict[str, float]:
    data = _signed_get("/api/v3/account")
    return {
        b["asset"]: float(b["free"]) + float(b["locked"])
        for b in data["balances"]
        if float(b["free"]) + float(b["locked"]) > 0
    }


def get_spot_balance(asset: str) -> float:
    data = _signed_get("/api/v3/account")
    for b in data["balances"]:
        if b["asset"] == asset.upper():
            return float(b["free"])
    return 0.0


def redeem_earn(asset: str, amount: float) -> float:
    """Rachète `amount` depuis Simple Earn (flexible d'abord, puis locked). Retourne le total racheté."""
    redeemed = 0.0
    asset = asset.upper()

    # Flexible earn (rachat partiel possible)
    try:
        data = _signed_get("/sapi/v1/simple-earn/flexible/position", {"asset": asset})
        for row in data.get("rows", []):
            if redeemed >= amount:
                break
            to_redeem = min(amount - redeemed, float(row["totalAmount"]))
            if to_redeem < 1e-8:
                continue
            try:
                _signed_post("/sapi/v1/simple-earn/flexible/redeem", {
                    "productId": row["productId"],
                    "amount": to_redeem,
                })
                redeemed += to_redeem
            except Exception as e:
                import logging
                logging.warning("Earn flexible redeem %s productId=%s : %s",
                                asset, row.get("productId"), e)
    except Exception as e:
        import logging
        logging.warning("Earn flexible position fetch %s : %s", asset, e)

    if redeemed >= amount:
        return redeemed

    # Locked earn (rachat de la position entière uniquement)
    try:
        data = _signed_get("/sapi/v1/simple-earn/locked/position", {"asset": asset})
        for row in data.get("rows", []):
            if redeemed >= amount:
                break
            try:
                _signed_post("/sapi/v1/simple-earn/locked/redeem", {
                    "positionId": row["positionId"],
                })
                redeemed += float(row["amount"])
            except Exception as e:
                import logging
                logging.warning("Earn locked redeem %s positionId=%s : %s",
                                asset, row.get("positionId"), e)
    except Exception as e:
        import logging
        logging.warning("Earn locked position fetch %s : %s", asset, e)

    return redeemed


def place_market_order(
    symbol: str,
    side: str,
    quantity: float | None = None,
    quote_qty: float | None = None,
) -> tuple[float, float]:
    """Places a market order. Returns (avg_price, filled_qty)."""
    if quantity is None and quote_qty is None:
        raise ValueError("quantity ou quote_qty requis")
    pair = f"{symbol.upper()}{QUOTE_CURRENCY}"
    params: dict = {"symbol": pair, "side": side.upper(), "type": "MARKET"}
    if quote_qty is not None:
        params["quoteOrderQty"] = math.floor(quote_qty * 100) / 100
    else:
        params["quantity"] = quantity
    data = _signed_post("/api/v3/order", params)
    filled_qty = float(data["executedQty"])
    fills = data.get("fills", [])
    if fills:
        avg_price = sum(float(f["price"]) * float(f["qty"]) for f in fills) / filled_qty
    else:
        avg_price = 0.0
    return avg_price, filled_qty


def get_all_tickers_24h() -> list[dict]:
    return _get("/api/v3/ticker/24hr")


def get_all_usdc_pairs() -> list[str]:
    """Return base assets of all TRADING pairs quoted in USDC."""
    trading, _ = get_usdc_pairs_by_status()
    return trading


def get_usdc_pairs_by_status() -> tuple[list[str], set[str]]:
    """Single exchangeInfo call — returns (trading_symbols, inactive_symbols).
    Pairs with empty permissions lists are treated as inactive (restricted/frozen)."""
    data = _get("/api/v3/exchangeInfo")
    trading, inactive = [], set()
    for s in data["symbols"]:
        if s["quoteAsset"] != QUOTE_CURRENCY:
            continue
        # permissionSets replaced permissions in newer API versions
        psets = s.get("permissionSets") or s.get("permissions") or []
        has_spot = any("SPOT" in pset for pset in psets) if psets else False
        if s["status"] == "TRADING" and has_spot:
            trading.append(s["baseAsset"])
        else:
            inactive.add(s["baseAsset"])
    return trading, inactive


def get_funding_rates(symbol: str, start_ms: int | None = None, limit: int = 1000) -> list[dict]:
    """
    Fetch perpetual futures funding rates (USDT-margined) for symbol.
    Returns [] if no perpetual contract exists for that symbol.
    """
    params: dict = {"symbol": f"{symbol.upper()}USDT", "limit": limit}
    if start_ms is not None:
        params["startTime"] = start_ms
    try:
        resp = requests.get(
            "https://fapi.binance.com/fapi/v1/fundingRate",
            params=params,
            timeout=10,
        )
        if resp.status_code in (400, 404):
            return []
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return []


def get_lot_size(symbol: str) -> tuple[float, float]:
    """Return (stepSize, minQty) for symbol/QUOTE pair from LOT_SIZE filter."""
    try:
        data = _get("/api/v3/exchangeInfo", {"symbol": f"{symbol.upper()}{QUOTE_CURRENCY}"})
        for s in data.get("symbols", []):
            for f in s.get("filters", []):
                if f["filterType"] == "LOT_SIZE":
                    return float(f["stepSize"]), float(f["minQty"])
    except Exception:
        pass
    return 1.0, 0.0


def get_recent_klines(symbol: str, interval: str, limit: int = 60) -> list[list]:
    return _get("/api/v3/klines", {
        "symbol": f"{symbol.upper()}{QUOTE_CURRENCY}",
        "interval": interval,
        "limit": limit,
    })


def get_klines(
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int | None = None,
) -> list[list]:
    """Fetches up to 1000 klines starting at start_ms (epoch ms)."""
    params: dict = {
        "symbol": f"{symbol.upper()}{QUOTE_CURRENCY}",
        "interval": interval,
        "startTime": start_ms,
        "limit": 1000,
    }
    if end_ms is not None:
        params["endTime"] = end_ms
    return _get("/api/v3/klines", params)


def get_earn_aprs() -> dict[str, float]:
    """
    Return Simple Earn flexible APRs keyed by asset, as percentage.
    e.g. {"BTC": 5.44, "ETH": 3.21}
    Returns {} if API keys are absent or endpoint fails.
    """
    result: dict[str, float] = {}
    page = 1
    while True:
        try:
            data = _signed_get("/sapi/v1/simple-earn/flexible/list",
                               {"current": page, "size": 100})
            rows = data.get("rows", [])
            for row in rows:
                asset = row["asset"]
                apr   = float(row.get("latestAnnualPercentageRate", 0)) * 100
                result[asset] = round(apr, 2)
            if len(rows) < 100:
                break
            page += 1
        except Exception:
            break
    return result


def get_my_trades(symbol: str, start_ms: int | None = None,
                  end_ms: int | None = None) -> list[dict]:
    """
    Fetch executed trades for one symbol from Binance (/api/v3/myTrades).
    Returns a list of dicts with keys: time, isBuyer, price, qty, quoteQty.
    """
    params: dict = {"symbol": f"{symbol}{QUOTE_CURRENCY}", "limit": 1000}
    if start_ms:
        params["startTime"] = start_ms
    if end_ms:
        params["endTime"] = end_ms
    try:
        return _signed_get("/api/v3/myTrades", params)  # type: ignore[return-value]
    except Exception:
        return []


def get_earn_balances() -> dict[str, float]:
    """Returns total Simple Earn balances (flexible + locked) keyed by underlying asset."""
    result: dict[str, float] = {}

    endpoints = [
        ("/sapi/v1/simple-earn/flexible/position", "totalAmount"),
        ("/sapi/v1/simple-earn/locked/position", "amount"),
    ]
    for endpoint, amount_key in endpoints:
        page = 1
        while True:
            try:
                data = _signed_get(endpoint, {"current": page, "size": 100})
                rows = data.get("rows", [])
                for row in rows:
                    asset = row["asset"]
                    result[asset] = result.get(asset, 0.0) + float(row[amount_key])
                if len(rows) < 100:
                    break
                page += 1
            except Exception:
                break

    return result
