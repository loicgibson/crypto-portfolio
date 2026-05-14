# Commandes disponibles — crypto-portfolio

Syntaxe de base : `crypto-portfolio <commande> [options]`

---

## Configuration

| Commande | Description |
|---|---|
| `setup-keys` | Configurer les clés API Binance dans le Credential Manager |
| `setup-anthropic` | Configurer la clé API Anthropic dans le Credential Manager |
| `setup-grok` | Configurer la clé API Grok (xAI) pour le sentiment X/Twitter |

> **Grok (xAI)** — optionnel. Quand configuré, chaque cycle de trading injecte automatiquement le sentiment X en temps réel dans le contexte envoyé à Claude. Voir aussi la commande `sentiment`.

---

## Portefeuille

| Commande | Description |
|---|---|
| `watch` | Afficher le portefeuille et les cours live (one-shot) |
| `prices <SYM...>` | Cours live de symboles Binance |
| `buy <SYM> <qty> <prix>` | Enregistrer un achat manuel |
| `sell <SYM> <qty> <prix>` | Enregistrer une vente manuelle |
| `sync [--quote STABLE]` | Synchroniser les balances depuis Binance |
| `history [SYM]` | Historique des transactions (optionnellement filtré par symbole) |

---

## Trading réel

| Commande | Description |
|---|---|
| `order buy\|sell <SYM> <qty>` | Passer un ordre marché réel |
| `order buy\|sell <SYM> --for <USDC>` | Ordre en montant USDC plutôt qu'en quantité |
| `rebalance [options]` | Suggère et exécute les rééquilibrages |

**Options `rebalance` :**
- `--interval 1h` — intervalle klines (défaut : 1h)
- `--stop-loss PCT` — seuil stop-loss (défaut : config)
- `--tp1 PCT / --tp2 PCT` — take-profit paliers 1 et 2
- `--position-size PCT` — taille de position (% du capital)
- `--max-positions N` — nombre max de positions simultanées
- `--reserve PCT` — réserve USDC minimum

---

## Live (trading automatique réel)

| Commande | Description |
|---|---|
| `live-status` | Afficher le portefeuille live (tracking interne) |
| `live-run [options]` | Un cycle de trading sur le compte Binance réel |
| `live-loop [options]` | Boucle automatique : cycle principal toutes les N min + sous-cycle watchlist toutes les 5 min |
| `live-history [--limit N] [--cycles N]` | Historique des transactions et cycles live |
| `live-recap [dd-MM-YYYY]` | Récapitulatif P&L d'une journée (défaut : aujourd'hui) |
| `live-sync` | Réconcilier le tracking interne avec les balances réelles Binance |

**Options communes `live-run` / `live-loop` :**
- `--interval MIN` — intervalle principal en minutes (loop, défaut : 15)
- `--dry-run` — simuler les ordres sans les exécuter
- `--yes / -y` — ignorer la confirmation interactive
- `--verbose` — afficher le JSON envoyé à l'API

---

## Simulation (paper trading)

| Commande | Description |
|---|---|
| `sim-reset [--balance USDC]` | Initialiser / réinitialiser le portefeuille virtuel (défaut : 1000 USDC) |
| `sim-status` | Afficher le portefeuille virtuel |
| `sim-run [options]` | Un cycle complet de simulation |
| `sim-loop [options]` | Boucle automatique simulation |
| `sim-history [--limit N] [--cycles N]` | Historique transactions et cycles virtuels |
| `sim-recap [dd-MM-YYYY]` | Récapitulatif P&L simulation d'une journée (défaut : aujourd'hui) |

**Options communes `sim-run` / `sim-loop` :**
- `--interval MIN` — intervalle entre chaque scan en minutes (loop, défaut : 15)
- `--verbose / -v` — afficher le JSON envoyé à l'API

---

## Scan du marché

| Commande | Description |
|---|---|
| `scan [--top N] [--interval 1h] [--min-volume USDC]` | Scanner les paires en début de progression |
| `dip [--top N] [--interval 1h] [--min-volume USDC]` | Scanner les paires en dip dans une tendance haussière |

**Valeurs par défaut :** `--top 15`, `--min-volume 50000`

---

## Analyse du portefeuille

| Commande | Description |
|---|---|
| `analyze [--interval 1h]` | Analyser les positions pour identifier les sorties |
| `sentiment [SYM...]` | Afficher le sentiment X/Twitter en temps réel via Grok (défaut : positions du portefeuille) |
| `scout <SYM>` | Analyse approfondie d'un symbole avec verdict d'achat par Claude AI |
| `exclude add\|remove\|list [SYM...]` | Gérer la liste des symboles exclus des scans |
| `fetch <SYM> [--interval 15m] [--since YYYY-MM-DD]` | Télécharger l'historique klines localement |

> **`sentiment`** — nécessite `setup-grok`. Les données sont mises en cache 5 min. Sans argument, affiche les positions du portefeuille ; avec arguments, affiche les symboles spécifiés (ex: `sentiment BTC ETH SOL`).

---

## Machine Learning

| Commande | Description |
|---|---|
| `ml-fetch [options]` | Télécharger l'historique 1h pour toutes les paires USDC |
| `ml-train [--symbol SYM...] [--interval 1h]` | Entraîner un modèle ML par crypto (LGBM/RF/LR) |
| `ml-scan [options]` | Scanner les opportunités avec scores technique + ML |
| `ml-analyze [options]` | Analyser le portefeuille avec scores technique + ML |

**Options `ml-fetch` :**
- `--interval 1h` — intervalle klines (défaut : 1h)
- `--min-volume USDC` — volume 24h minimum (défaut : 1 000 000)
- `--years N` — profondeur historique en années (défaut : 5)

**Options `ml-scan` :**
- `--top N` — nombre de résultats affichés (défaut : 20)
- `--interval 1h` — intervalle klines pour le score tech
- `--ml-interval` — intervalle des modèles ML
- `--min-volume USDC` — volume minimum (défaut : 500 000)
- `--pool N` — candidats à analyser avant filtrage (défaut : 100)
- `--min-tech N` — score tech minimum (défaut : 3)
- `--min-ml PROB` — probabilité ML minimum (défaut : 0.45)
- `--tech-weight W` — poids du score technique dans le combined (défaut : 0.4)

---

## Logs

Les logs sont écrits automatiquement à chaque run dans `log/` :

| Fichier | Contenu |
|---|---|
| `log/YYYY-MM-DD_candidates.jsonl` | Un enregistrement par symbole scanné (candidats + filtrés) par cycle |
| `log/YYYY-MM-DD_trades.jsonl` | Un enregistrement par trade exécuté (BUY ou SELL) |

```python
# Exemple de lecture pour backtester une règle
import json
from pathlib import Path
rows = [json.loads(l) for l in Path("log/2026-05-14_trades.jsonl").read_text().splitlines()]
buys = [r for r in rows if r["type"] == "BUY"]
```
