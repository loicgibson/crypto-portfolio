# Commandes disponibles — crypto-portfolio

Syntaxe de base : `crypto-portfolio <commande> [options]`

---

## Configuration

| Commande | Description |
|---|---|
| `setup-keys` | Configurer les clés API Binance dans le Credential Manager |
| `setup-anthropic` | Configurer la clé API Anthropic dans le Credential Manager |

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
| `live-run [options]` | Un cycle : Tier-1 (si 24h écoulées) + Tier-2 pump scan |
| `live-loop [options]` | Boucle automatique : pump toutes les 15 min, classique toutes les 24h |
| `live-history [--limit N] [--cycles N]` | Historique des transactions et cycles live |
| `live-recap <dd-MM-YYYY>` | Récapitulatif P&L d'une journée (ex: `04-05-2026`) |

**Options communes `live-run` / `live-loop` :**
- `--interval MIN` — intervalle pump en minutes (loop, défaut : 15)
- `--interval-klines 1h` — intervalle klines Tier-1 (défaut : 1h)
- `--ml-interval` — intervalle des modèles ML (défaut : valeur config)
- `--pool N` — candidats Tier-1 à analyser (défaut : 10)
- `--force-classic` — forcer le cycle classique même si < 24h (run seulement)
- `--dry-run` — simuler les ordres sans les exécuter
- `--yes / -y` — ignorer la confirmation interactive
- `--verbose` — afficher le JSON envoyé à l'API

---

## Simulation (paper trading)

| Commande | Description |
|---|---|
| `sim-reset [--balance USDC]` | Initialiser / réinitialiser le portefeuille virtuel (défaut : 1000 USDC) |
| `sim-status` | Afficher le portefeuille virtuel |
| `sim-run [options]` | Un cycle complet (Tier-1 + Tier-2 pump scan) |
| `sim-loop [options]` | Boucle automatique simulation |
| `sim-history [--limit N] [--cycles N]` | Historique transactions et cycles virtuels |

**Options communes `sim-run` / `sim-loop` :**
- `--interval MIN` — intervalle pump en minutes (loop, défaut : 15)
- `--interval-klines 1h` — intervalle klines Tier-1 (défaut : 1h)
- `--ml-interval` — intervalle des modèles ML
- `--pool N` — candidats Tier-1 à analyser (défaut : 10)
- `--force-classic` — forcer le cycle classique même si < 24h (run seulement)
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
| `exclude add\|remove\|list [SYM...]` | Gérer la liste des symboles exclus des scans |
| `fetch <SYM> [--interval 15m] [--since YYYY-MM-DD]` | Télécharger l'historique klines localement |

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
| `log/YYYY-MM-DD_candidates.jsonl` | Un enregistrement par symbole scanné (candidats + filtrés) par cycle pump |
| `log/YYYY-MM-DD_trades.jsonl` | Un enregistrement par trade exécuté (BUY ou SELL, Tier-1 et Tier-2) |

```python
# Exemple de lecture pour backtester une règle
import json
from pathlib import Path
rows = [json.loads(l) for l in Path("log/2026-05-04_trades.jsonl").read_text().splitlines()]
p2_buys = [r for r in rows if r["type"] == "BUY" and r.get("signal") == "p2"]
```
