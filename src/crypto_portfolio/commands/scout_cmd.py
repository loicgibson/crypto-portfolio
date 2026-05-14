"""
scout — single-symbol deep-dive analysis with Claude AI verdict.

Usage:
    crypto-portfolio scout BTC
    crypto-portfolio scout LUNC --capital 500
"""
import json

from ..binance import get_all_tickers_24h, get_earn_aprs, get_recent_klines
from ..config import QUOTE_CURRENCY
from ..display import console
from ..metrics.compute import compute_context, compute_metrics
from ..ml.predictor import predict_symbol
from ..storage import init_db

from ._market import _condensed_candles, _funding_summary


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fmt_float(v, decimals=2, suffix="") -> str:
    if v is None:
        return "[dim]—[/]"
    return f"{v:.{decimals}f}{suffix}"


def _chg_color(v) -> str:
    if v is None:
        return "[dim]—[/]"
    color = "green" if v >= 0 else "red"
    return f"[{color}]{v:+.2f}%[/]"


def _display_metrics_panel(symbol: str, ctx: dict) -> None:
    from rich import box
    from rich.panel import Panel
    from rich.table import Table

    m   = ctx.get("metrics") or {}
    c1h = ctx.get("context_1h") or {}
    mkt = ctx.get("market") or {}

    # ── Price / changes ───────────────────────────────────────────────────────
    price_table = Table(box=None, show_header=False, padding=(0, 1))
    price_table.add_column(style="dim")
    price_table.add_column()

    price_table.add_row("Prix",    _fmt_float(ctx.get("price"), decimals=8))
    price_table.add_row("1h",      _chg_color(ctx.get("change_1h")))
    price_table.add_row("3h",      _chg_color(ctx.get("change_3h")))
    price_table.add_row("6h",      _chg_color(ctx.get("change_6h")))
    price_table.add_row("24h",     _chg_color(ctx.get("change_24h")))
    price_table.add_row("Spike15m", _fmt_float(ctx.get("vol_spike_15m"), 2, "x"))

    # ── Momentum indicators ───────────────────────────────────────────────────
    mom_table = Table(box=None, show_header=False, padding=(0, 1))
    mom_table.add_column(style="dim")
    mom_table.add_column()

    rsi = m.get("rsi_14")
    rsi_color = "red" if rsi and rsi > 70 else "green" if rsi and rsi < 35 else "white"
    mom_table.add_row("RSI",        f"[{rsi_color}]{_fmt_float(rsi, 1)}[/]")
    mom_table.add_row("RSI trend",  _fmt_float(m.get("rsi_trend_val"), 1))
    mom_table.add_row("RSI div",    str(m.get("rsi_divergence") or "—"))
    mom_table.add_row("MACD dir",   str(m.get("macd_hist_direction") or "—"))
    mom_table.add_row("Stoch K/D",  f"{_fmt_float(m.get('stoch_k'), 1)} / {_fmt_float(m.get('stoch_d'), 1)}")

    # ── Volume / BB ───────────────────────────────────────────────────────────
    vol_table = Table(box=None, show_header=False, padding=(0, 1))
    vol_table.add_column(style="dim")
    vol_table.add_column()

    vol = m.get("volume_ratio")
    vol_color = "green" if vol and vol > 1.5 else "red" if vol and vol < 0.7 else "white"
    vol_table.add_row("Vol ratio",    f"[{vol_color}]{_fmt_float(vol, 2)}x[/]")
    vol_table.add_row("Vol trend",    str(m.get("volume_trend_5") or "—"))
    vol_table.add_row("Buy/Sell",     _fmt_float(m.get("buy_sell_ratio"), 2))
    vol_table.add_row("BB position",  _fmt_float(m.get("bb_position"), 3))
    vol_table.add_row("BB squeeze",   str(m.get("bb_squeeze_active") or False))

    # ── Trend / structure ─────────────────────────────────────────────────────
    trend_table = Table(box=None, show_header=False, padding=(0, 1))
    trend_table.add_column(style="dim")
    trend_table.add_column()

    trend_table.add_row("MA align",     str(m.get("ma_alignment") or "—"))
    trend_table.add_row("MA25 dist",    _fmt_float(m.get("price_distance_ma25_pct"), 2, "%"))
    trend_table.add_row("MA99 (1h)",    "[green]✓[/]" if c1h.get("price_above_ma99_1h") else "[red]✗[/]")
    trend_table.add_row("Phase 1h",     str(c1h.get("market_phase_1h") or "—"))
    trend_table.add_row("Pump phase",   str(m.get("pump_phase") or "—"))
    trend_table.add_row("Extension",    _fmt_float(m.get("extension_atr"), 2, " ATR"))
    trend_table.add_row("Parabolic",    _fmt_float(m.get("parabolic_score"), 0))

    # ── ML + funding + market ─────────────────────────────────────────────────
    ml_table = Table(box=None, show_header=False, padding=(0, 1))
    ml_table.add_column(style="dim")
    ml_table.add_column()

    ml_prob = ctx.get("ml_prob_up")
    ml_ap   = ctx.get("ml_ap")
    ml_color = "green" if ml_prob and ml_prob >= 0.60 else "red" if ml_prob and ml_prob < 0.40 else "white"
    ml_table.add_row("ML prob",  f"[{ml_color}]{_fmt_float(ml_prob, 3)}[/]")
    ml_table.add_row("ML AP",    _fmt_float(ml_ap, 3))

    funding = ctx.get("funding") or {}
    if funding:
        fr_color = "red" if (funding.get("avg_7d_pct") or 0) > 0.03 else "green"
        ml_table.add_row("Funding 7d", f"[{fr_color}]{_fmt_float(funding.get('avg_7d_pct'), 4)}%[/]")
        ml_table.add_row("Fund trend", str(funding.get("trend") or "—"))

    earn_apr = ctx.get("earn_apr")
    if earn_apr is not None:
        ml_table.add_row("Earn APR",  _fmt_float(earn_apr, 2, "%"))

    for sym in ("BTC", "ETH"):
        if sym in mkt:
            ml_table.add_row(sym, _chg_color(mkt[sym].get("change_24h")))

    # ── Candles sparkline ─────────────────────────────────────────────────────
    candles = ctx.get("candles_1h") or []
    sparks = []
    for c in candles:
        if len(c) >= 2:
            sparks.append("[green]▲[/]" if c[1] >= c[0] else "[red]▼[/]")
    candle_str = " ".join(sparks) if sparks else "[dim]—[/]"

    # ── Assemble with a grid ──────────────────────────────────────────────────
    from rich.columns import Columns

    console.rule(f"[bold cyan]{symbol} — analyse technique[/]")
    console.print(Columns([price_table, mom_table, vol_table, trend_table, ml_table], equal=False, expand=False))
    console.print(f"  [dim]Bougies 1h (12h) :[/] {candle_str}")
    console.print()


def _display_verdict(symbol: str, result: dict) -> None:
    from rich.panel import Panel

    verdict = result.get("verdict", "WAIT")
    conf    = result.get("confidence", 0)
    alloc   = result.get("suggested_allocation_pct", 0)
    sl      = result.get("stop_loss_pct", -8)
    tp      = result.get("take_profit_pct", 20)

    color_map = {
        "STRONG_BUY": "bold green",
        "BUY":        "green",
        "WAIT":       "yellow",
        "AVOID":      "bold red",
    }
    color = color_map.get(verdict, "white")

    lines = [
        f"[{color}]▶  {verdict}[/]  [dim]confiance {conf}%[/]",
        "",
    ]

    analysis = result.get("analysis", "")
    if analysis:
        lines.append(f"[white]{analysis}[/]")
        lines.append("")

    signals = result.get("key_signals") or []
    if signals:
        lines.append("[bold]Signaux clés :[/]")
        for s in signals:
            lines.append(f"  [green]✦[/] {s}")
        lines.append("")

    risks = result.get("risks") or []
    if risks:
        lines.append("[bold]Risques :[/]")
        for r in risks:
            lines.append(f"  [red]⚠[/] {r}")
        lines.append("")

    if verdict in ("STRONG_BUY", "BUY"):
        lines.append(
            f"[bold]Allocation suggérée :[/] {alloc:.0f}%  |  "
            f"[bold]Stop-loss :[/] {sl:.1f}%  |  "
            f"[bold]Take-profit :[/] +{tp:.1f}%"
        )

    console.print(Panel("\n".join(lines), title=f"[bold]Verdict Claude — {symbol}[/]", border_style=color.split()[-1]))


# ── Command ───────────────────────────────────────────────────────────────────

def cmd_scout(args) -> None:
    init_db()
    symbol = args.symbol.upper()

    ctx: dict = {"symbol": symbol}

    with console.status(f"[cyan]Collecte des données {symbol}…[/]") as status:
        # klines
        try:
            klines_1h = get_recent_klines(symbol, "1h", limit=100)
        except Exception as e:
            console.print(f"[red]Impossible de récupérer les klines {symbol} : {e}[/]")
            return

        if len(klines_1h) < 30:
            console.print(f"[red]{symbol} : données insuffisantes ({len(klines_1h)} bougies).[/]")
            return

        closes = [float(k[4]) for k in klines_1h]

        def _chg(n: int) -> float | None:
            return round((closes[-1] / closes[-n] - 1) * 100, 2) if len(closes) >= n and closes[-n] > 0 else None

        # 15m spike
        klines_15m = None
        try:
            klines_15m = get_recent_klines(symbol, "15m", limit=10)
        except Exception:
            pass

        vol_spike_15m = None
        if klines_15m and len(klines_15m) >= 4:
            vols = [float(k[5]) for k in klines_15m]
            avg  = sum(vols[-4:-1]) / 3
            vol_spike_15m = round(vols[-1] / avg, 2) if avg > 0 else None

        # indicators
        status.update(f"[cyan]Calcul des indicateurs {symbol}…[/]")
        try:
            metrics    = compute_metrics(klines_1h)
            context_1h = compute_context(klines_1h)
            candles    = _condensed_candles(klines_1h, n=12)
            funding    = _funding_summary(symbol)
        except Exception as e:
            console.print(f"[yellow]Avertissement indicateurs : {e}[/]")
            metrics = context_1h = {}
            candles = []
            funding = None

        # BTC/ETH context + 24h change
        status.update("[cyan]Contexte marché…[/]")
        try:
            all_tickers = get_all_tickers_24h()
            tickers_map = {
                t["symbol"].removesuffix(QUOTE_CURRENCY): t
                for t in all_tickers
                if t["symbol"].endswith(QUOTE_CURRENCY)
            }
        except Exception:
            tickers_map = {}

        market: dict = {}
        for ctx_sym in ("BTC", "ETH"):
            if ctx_sym in tickers_map:
                t = tickers_map[ctx_sym]
                market[ctx_sym] = {
                    "price":      round(float(t["lastPrice"]), 2),
                    "change_24h": round(float(t["priceChangePercent"]), 2),
                }

        change_24h = None
        if symbol in tickers_map:
            change_24h = round(float(tickers_map[symbol]["priceChangePercent"]), 2)

        # Earn APR
        earn_apr = None
        try:
            earn_aprs = get_earn_aprs()
            earn_apr  = earn_aprs.get(symbol)
        except Exception:
            pass

        # ML prediction
        status.update(f"[cyan]Prédiction ML {symbol}…[/]")
        try:
            ml = predict_symbol(symbol)
        except Exception:
            ml = {}

        ctx.update({
            "price":         round(closes[-1], 8),
            "change_1h":     _chg(2),
            "change_3h":     _chg(4),
            "change_6h":     _chg(7),
            "change_24h":    change_24h,
            "vol_spike_15m": vol_spike_15m,
            "metrics":       metrics,
            "context_1h":    context_1h,
            "candles_1h":    candles,
            "ml_prob_up":    ml.get("ml_prob"),
            "ml_ap":         ml.get("ap"),
            "funding":       funding,
            "earn_apr":      earn_apr,
            "market":        market,
        })

    # Display metrics
    _display_metrics_panel(symbol, ctx)

    # Call Claude
    with console.status("[cyan]Analyse Claude en cours…[/]"):
        from ..ml.brain import get_scout_decision
        try:
            result = get_scout_decision(ctx)
        except ValueError as e:
            console.print(f"[red]{e}[/]")
            return
        except Exception as e:
            console.print(f"[red]Erreur API Claude : {e}[/]")
            return

    _display_verdict(symbol, result)


def register(sub):
    p = sub.add_parser(
        "scout",
        help="Analyse approfondie d'un symbole avec verdict d'achat par Claude AI",
    )
    p.add_argument("symbol", help="Symbole crypto à analyser (ex: BTC, LUNC)")
    return {"scout": cmd_scout}
