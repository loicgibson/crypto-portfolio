"""
Training pipeline — multi-timeframe pump detection.

For each symbol:
  1. Load 1h / 4h / 15m klines from SQLite
  2. Compute multi-TF features + path-dependent target
  3. Chronological 80/20 split
  4. Walk-forward CV (5 folds) to compare LGBM / RF / LR
  5. Best model retrained on full train set, evaluated on test set
  6. Saved to  ML_MODELS_DIR/{symbol}.pkl

Target: y = 1  if  max(high[t+1..t+4]) / close[t] - 1  >= ML_THRESHOLD
              AND  min(low[t+1..t+4])  / close[t] - 1  > -STOP_LOSS_PCT_TIER2/100
Aligns with HARD_TAKE_PROFIT_PCT (default 4%) and STOP_LOSS_PCT_TIER2 (default 5%).
"""
import sqlite3
import time
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np

from ..config import (DB_PATH, ML_HORIZON, ML_MODELS_DIR, ML_THRESHOLD,
                      STOP_LOSS_PCT_TIER2)
from .features import (FEATURE_COLS, TARGET_HORIZON, TARGET_THRESHOLD,
                       add_funding_features, add_market_context, add_target,
                       compute_features, klines_to_df)

warnings.filterwarnings("ignore", category=UserWarning)

MIN_TRAIN_ROWS = 500
MODELS_DIR     = Path(ML_MODELS_DIR)


# ── Model definitions ─────────────────────────────────────────────────────────

def _build_candidates():
    from lightgbm import LGBMClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    imputer = SimpleImputer(strategy="constant", fill_value=0)

    return {
        "lgbm": LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            random_state=42,
            verbose=-1,
        ),
        "rf": Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
            ("clf", RandomForestClassifier(
                n_estimators=300,
                max_depth=8,
                min_samples_leaf=20,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )),
        ]),
        "lr": Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=42,
                C=0.1,
            )),
        ]),
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_klines_df(symbol: str, interval: str) -> "pd.DataFrame":
    """Read klines from SQLite. Returns empty DataFrame if none found."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT open_time, open, high, low, close, volume, close_time, "
            "quote_volume, num_trades FROM klines "
            "WHERE symbol=? AND interval=? ORDER BY open_time",
            (symbol.upper(), interval),
        ).fetchall()
    return klines_to_df(rows)


def _cv_score(model, X, y, n_splits: int = 5) -> float:
    from sklearn.metrics import average_precision_score
    from sklearn.model_selection import TimeSeriesSplit

    tscv   = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    for train_idx, val_idx in tscv.split(X):
        if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[val_idx])) < 2:
            continue
        m = _clone(model)
        m.fit(X[train_idx], y[train_idx])
        proba = m.predict_proba(X[val_idx])
        if proba.shape[1] < 2:
            continue
        scores.append(average_precision_score(y[val_idx], proba[:, 1]))
    return float(np.mean(scores)) if scores else 0.0


def _clone(model):
    from sklearn.base import clone
    return clone(model)


def _eval_metrics(model, X_test, y_test) -> dict:
    from sklearn.metrics import (average_precision_score, f1_score,
                                  precision_score, recall_score, roc_auc_score)

    prob  = model.predict_proba(X_test)[:, 1]
    pred  = (prob >= 0.5).astype(int)
    ap    = average_precision_score(y_test, prob)
    auc   = roc_auc_score(y_test, prob)
    prec  = precision_score(y_test, pred, zero_division=0)
    rec   = recall_score(y_test, pred, zero_division=0)
    f1    = f1_score(y_test, pred, zero_division=0)
    pred65 = (prob >= 0.65).astype(int)
    prec65 = precision_score(y_test, pred65, zero_division=0) if pred65.sum() > 0 else 0.0
    n_buy  = int(pred65.sum())

    return {
        "ap":           round(ap, 4),
        "auc":          round(auc, 4),
        "precision_50": round(prec, 4),
        "recall_50":    round(rec, 4),
        "f1_50":        round(f1, 4),
        "precision_65": round(prec65, 4),
        "n_buy_65":     n_buy,
        "pos_rate":     round(float(y_test.mean()), 4),
    }


# ── Public API ────────────────────────────────────────────────────────────────

def get_model_path(symbol: str) -> Path:
    return MODELS_DIR / f"{symbol.upper()}.pkl"


def load_model(symbol: str, interval: str = "1h") -> dict | None:
    """Return saved model bundle or None if not found."""
    p = get_model_path(symbol)
    if not p.exists():
        return None
    return joblib.load(p)


def train_symbol(symbol: str, interval: str = "1h",
                 progress_cb=None) -> dict:
    """
    Train models for one symbol using 1h + 4h + 15m data.

    Returns a result dict with keys:
      symbol, best_model, cv_scores, test_metrics,
      n_train, n_test, pos_rate_train, trained_at, error (if failed)
    """
    sym = symbol.upper()

    raw_1h  = _load_klines_df(sym, "1h")
    if raw_1h.empty:
        return {"symbol": sym, "error": "no 1h klines in DB"}

    raw_4h  = _load_klines_df(sym, "4h")
    raw_15m = _load_klines_df(sym, "15m")
    btc_1h  = _load_klines_df("BTC", "1h") if sym != "BTC" else None

    from ..storage import get_funding_df
    funding_df = get_funding_df(sym)

    df_4h  = raw_4h  if not raw_4h.empty  else None
    df_15m = raw_15m if not raw_15m.empty else None

    df = compute_features(raw_1h, df_4h=df_4h, df_15m=df_15m)
    df = add_market_context(df, btc_1h, is_btc=(sym == "BTC"))
    df = add_funding_features(df, funding_df)
    df = add_target(df, sl_frac=STOP_LOSS_PCT_TIER2 / 100)

    if len(df) < MIN_TRAIN_ROWS:
        return {"symbol": sym, "error": f"only {len(df)} labelled rows (need {MIN_TRAIN_ROWS})"}

    import numpy as np
    X = df[FEATURE_COLS].fillna(0).values.astype(np.float32)
    y = df["target"].values.astype(int)

    split = int(len(X) * 0.80)
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]

    if len(np.unique(y_te)) < 2:
        return {"symbol": sym, "error": "test set contains only one class"}

    candidates = _build_candidates()

    if progress_cb:
        progress_cb(f"[cyan]  {sym} — CV…[/]")
    cv_scores = {}
    for name, model in candidates.items():
        cv_scores[name] = _cv_score(model, X_tr, y_tr)

    best_name  = max(cv_scores, key=lambda k: cv_scores[k])
    best_model = _clone(candidates[best_name])

    if progress_cb:
        progress_cb(f"[cyan]  {sym} — fit {best_name}…[/]")
    best_model.fit(X_tr, y_tr)
    metrics = _eval_metrics(best_model, X_te, y_te)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model":          best_model,
        "model_name":     best_name,
        "feature_cols":   FEATURE_COLS,
        "symbol":         sym,
        "intervals":      ["1h", "4h", "15m"],
        "horizon":        TARGET_HORIZON,
        "threshold":      TARGET_THRESHOLD,
        "sl_frac":        round(STOP_LOSS_PCT_TIER2 / 100, 4),
        "cv_scores":      {k: round(v, 4) for k, v in cv_scores.items()},
        "test_metrics":   metrics,
        "n_train":        int(split),
        "n_test":         len(X_te),
        "pos_rate_train": round(float(y_tr.mean()), 4),
        "trained_at":     datetime.utcnow().isoformat(timespec="seconds"),
        "has_4h":         df_4h is not None,
        "has_15m":        df_15m is not None,
    }
    joblib.dump(bundle, get_model_path(sym))

    return {
        "symbol":     sym,
        "best_model": best_name,
        "cv_scores":  bundle["cv_scores"],
        "metrics":    metrics,
        "n_train":    int(split),
        "n_test":     len(X_te),
        "has_4h":     df_4h is not None,
        "has_15m":    df_15m is not None,
    }


def train_all(symbols: list[str] | None = None,
              interval: str = "1h",
              progress_cb=None) -> list[dict]:
    """Train models for all given symbols (or all with 1h data in DB)."""
    if symbols is None:
        with sqlite3.connect(DB_PATH) as conn:
            rows = conn.execute(
                "SELECT DISTINCT symbol FROM klines WHERE interval='1h'"
            ).fetchall()
        symbols = [r[0] for r in rows]

    results = []
    for sym in symbols:
        try:
            r = train_symbol(sym, interval, progress_cb=progress_cb)
        except Exception as exc:
            r = {"symbol": sym, "error": str(exc)}
        results.append(r)

    return results
