"""
config.py
---------
Central configuration for Argus.
All API keys are read from environment variables — never hardcoded.
"""

import os

# --- Gemini ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-2.0-flash"

if not GEMINI_API_KEY:
    raise EnvironmentError(
        "GEMINI_API_KEY not found in environment. "
        "Run: export GEMINI_API_KEY='your_key_here' in your terminal."
    )

# --- Paths ---
DATA_DIR = "data/filings"
CHARTS_DIR = "charts"
LOGS_DIR = "logs"

# --- Sentiment Agent ---
SENTIMENT_SEARCH_RESULTS = 10          # Number of DuckDuckGo results to fetch
SENTIMENT_CONFIDENCE_THRESHOLD = 0.60  # Below this, orchestrator escalates

# --- Market Agent ---
DEFAULT_DATE_RANGE_DAYS = 180          # Fallback if orchestrator doesn't set a range
CHART_FIGSIZE = (12, 7)

# --- RAG Agent ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
RAG_TOP_K = 4                          # Number of chunks retrieved per question
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # sentence-transformers model, runs locally

# --- SEC EDGAR ---
EDGAR_BASE_URL = "https://efts.sec.gov/LATEST/search-index"
EDGAR_FILING_URL = "https://www.sec.gov"
EDGAR_HEADERS = {
    "User-Agent": "Argus Research Tool salilborkar@gmail.com",  # SEC requires this
    "Accept-Encoding": "gzip, deflate",
}

# --- Orchestrator ---
ORCHESTRATOR_TEMPERATURE = 0.2         # Low temp for structured reasoning
MAX_ORCHESTRATOR_RETRIES = 2           # Retry once if agent returns low confidence
