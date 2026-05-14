import os
import keyring
from dotenv import load_dotenv

load_dotenv()

KEYRING_SERVICE = "crypto-portfolio"
QUOTE_CURRENCY = os.getenv("QUOTE_CURRENCY", "USDC")
REFRESH_INTERVAL = int(os.getenv("REFRESH_INTERVAL", "5"))
DB_PATH = os.getenv("DB_PATH", "portfolio.db")
BINANCE_BASE_URL = "https://api.binance.com"

ML_MODELS_DIR = os.getenv("ML_MODELS_DIR", "data/models")
ML_INTERVAL   = os.getenv("ML_INTERVAL", "1h")
ML_HORIZON    = int(os.getenv("ML_HORIZON", "4"))      # candles ahead (4h window)
ML_THRESHOLD  = float(os.getenv("ML_THRESHOLD", "0.04"))  # +4% peak gain (aligns with HARD_TAKE_PROFIT_PCT)

POSITION_SIZE_PCT   = float(os.getenv("POSITION_SIZE_PCT",   "7"))
STOP_LOSS_PCT       = float(os.getenv("STOP_LOSS_PCT",       "7"))   # Tier-1 classic
STOP_LOSS_PCT_TIER2 = float(os.getenv("STOP_LOSS_PCT_TIER2", "5"))   # Tier-2 pump
TAKE_PROFIT_1_PCT   = float(os.getenv("TAKE_PROFIT_1_PCT",   "30"))
TAKE_PROFIT_2_PCT   = float(os.getenv("TAKE_PROFIT_2_PCT",   "60"))
HARD_TAKE_PROFIT_PCT = float(os.getenv("HARD_TAKE_PROFIT_PCT", "4.0"))  # exit mécanique sans brain
DAILY_STOP_PCT      = float(os.getenv("DAILY_STOP_PCT",      "3.0"))   # stopper les BUY si P&L jour < -X%
DAILY_TARGET_PCT    = float(os.getenv("DAILY_TARGET_PCT",    "2.0"))   # objectif journalier (info pour brain)
MAX_POSITIONS    = int(os.getenv("MAX_POSITIONS",    "8"))
USDC_RESERVE_PCT = float(os.getenv("USDC_RESERVE_PCT", "25"))


def _get_secret(key: str, env_var: str) -> str:
    value = keyring.get_password(KEYRING_SERVICE, key)
    if value:
        return value
    return os.getenv(env_var, "") or ""


BINANCE_API_KEY = _get_secret("api_key", "BINANCE_API_KEY")
BINANCE_API_SECRET = _get_secret("api_secret", "BINANCE_API_SECRET")

ANTHROPIC_API_KEY = _get_secret("anthropic_api_key", "ANTHROPIC_API_KEY")
BRAIN_MODEL = os.getenv("BRAIN_MODEL", "claude-sonnet-4-6")

GROK_API_KEY = _get_secret("grok_api_key", "GROK_API_KEY")
GROK_MODEL   = os.getenv("GROK_MODEL", "grok-4-1-fast")
