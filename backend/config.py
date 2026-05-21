"""
Configuration — edit this file before running.
All values can be overridden with environment variables.
"""
import os

# ── GitHub ────────────────────────────────────────────────────────────────
GITHUB_REPO   = os.environ.get("GITHUB_REPO",   "chadallen/polymarket-insider-detection")
GITHUB_BRANCH = os.environ.get("GITHUB_BRANCH", "main")
GITHUB_TOKEN  = os.environ.get("GITHUB_TOKEN",  "")

# ── Dune Analytics ────────────────────────────────────────────────────────
DUNE_API_KEY  = os.environ.get("DUNE_API_KEY", "")

# ── Polygonscan (free tier: 5 req/sec, optional API key for better limits) ─
POLYGONSCAN_API_KEY = os.environ.get("POLYGONSCAN_API_KEY", "")

# ── Pipeline settings ─────────────────────────────────────────────────────
TOP_N_MARKETS      = int(os.environ.get("TOP_N_MARKETS", "50"))  # emergency override; not applied by default
MIN_VOLUME_USD     = 1_000_000    # Markets below this are excluded (lowered from $10M to capture lower-volume labeled cases)
MIN_END_DATE       = "2025-01-01"
FETCH_TAG_IDS = [
    2,       # Politics
    100265,  # Geopolitics
    596,     # Culture
    1401,    # Tech
    101999,  # Big Tech
    107,     # Business
    120,     # Finance
    101970,  # World
    100328,  # Economy
]
POLITICS_TAG_ID    = 2  # alias kept for any code that references it directly
MARKETS_PER_PAGE   = 100
MAX_PAGES          = 10
PRICE_HOURS_BEFORE = 48  # Hours of price history to fetch before resolution

# ── Question content filter ────────────────────────────────────────────────
# Markets whose question matches any of these patterns (case-insensitive) are
# excluded from the pipeline. Add patterns here to suppress noisy market types.
import re as _re
QUESTION_BLOCK_PATTERNS = [
    # Elon Musk tweet/post-count markets ("Will Elon Musk post 120-139 tweets...")
    _re.compile(r"elon musk", _re.IGNORECASE),
    # Andrew Tate post-count markets ("Andrew Tate total posts March 10 - March 17?")
    _re.compile(r"andrew tate", _re.IGNORECASE),
    # Generic election winner markets ("will X win the Y election", "election winner YYYY")
    _re.compile(r"win\b.{0,40}\belection\b", _re.IGNORECASE),
    _re.compile(r"\belection\b.{0,40}\bwinner\b", _re.IGNORECASE),
]

def question_is_blocked(question: str) -> bool:
    return any(p.search(question) for p in QUESTION_BLOCK_PATTERNS)

# ── Local data directory (replaces Google Drive) ──────────────────────────
DATA_DIR = os.environ.get(
    "DATA_DIR",
    os.path.join(os.path.dirname(__file__), "..", "data")
)
