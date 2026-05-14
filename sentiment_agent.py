"""
agents/sentiment_agent.py
--------------------------
Sentiment Agent for Argus.

Responsibility:
  - Search the web for recent news and analyst commentary on a ticker
  - Send results to Gemini for structured sentiment extraction
  - Write results into ResearchSession
  - Set escalation flag if confidence is below threshold

Input (reads from session):
  - session.ticker
  - session.user_query

Output (writes to session):
  - session.sentiment_score
  - session.sentiment_summary
  - session.sentiment_themes
  - session.sentiment_confidence
  - session.sentiment_sources
"""

import json
import google.generativeai as genai
from duckduckgo_search import DDGS

from core.session import ResearchSession
from config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    SENTIMENT_SEARCH_RESULTS,
    SENTIMENT_CONFIDENCE_THRESHOLD,
    ORCHESTRATOR_TEMPERATURE,
)

genai.configure(api_key=GEMINI_API_KEY)


def _search_web(ticker: str) -> list[dict]:
    """Fetch recent news headlines and snippets for the ticker."""
    queries = [
        f"{ticker} stock analyst sentiment 2025",
        f"{ticker} earnings outlook risks 2025",
    ]
    results = []
    with DDGS() as ddgs:
        for query in queries:
            hits = list(ddgs.text(query, max_results=SENTIMENT_SEARCH_RESULTS // 2))
            results.extend(hits)
    return results


def _extract_sentiment(ticker: str, search_results: list[dict]) -> dict:
    """
    Send search results to Gemini and extract structured sentiment.
    Returns a dict with score, summary, themes, confidence, sources.
    """
    snippets = "\n".join(
        f"- [{r.get('title', '')}]: {r.get('body', '')}"
        for r in search_results[:SENTIMENT_SEARCH_RESULTS]
    )
    sources = [r.get("href", "") for r in search_results if r.get("href")][:5]

    prompt = f"""
You are a financial analyst. Based on the following recent news and analyst commentary 
about {ticker}, provide a structured sentiment analysis.

NEWS SNIPPETS:
{snippets}

Respond ONLY with a valid JSON object in exactly this format (no markdown, no extra text):
{{
  "sentiment_score": "positive" | "negative" | "mixed" | "neutral",
  "confidence": <float between 0.0 and 1.0>,
  "summary": "<2-3 sentence summary of the overall sentiment>",
  "themes": ["<theme1>", "<theme2>", "<theme3>"],
  "key_risks": ["<risk1>", "<risk2>"],
  "recommended_date_range_days": <integer: how many days back is most relevant given this sentiment, e.g. 90, 180, 365>
}}
"""

    model = genai.GenerativeModel(
        GEMINI_MODEL,
        generation_config={"temperature": ORCHESTRATOR_TEMPERATURE},
    )
    response = model.generate_content(prompt)
    raw = response.text.strip()

    # Strip markdown fences if Gemini wraps in ```json
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    return json.loads(raw), sources


def run(session: ResearchSession):
    """
    Main entry point called by the orchestrator.
    Reads session.ticker, writes all sentiment fields back to session.
    """
    print(f"  [sentiment_agent] Searching web for {session.ticker} sentiment...")

    search_results = _search_web(session.ticker)

    if not search_results:
        session.sentiment_score = "neutral"
        session.sentiment_summary = "No recent news found."
        session.sentiment_confidence = 0.0
        session.escalation_flags.append("sentiment_agent: no search results returned")
        return

    print(f"  [sentiment_agent] Retrieved {len(search_results)} results. Extracting sentiment via Gemini...")

    extracted, sources = _extract_sentiment(session.ticker, search_results)

    session.sentiment_score = extracted.get("sentiment_score", "neutral")
    session.sentiment_confidence = float(extracted.get("confidence", 0.0))
    session.sentiment_summary = extracted.get("summary", "")
    session.sentiment_themes = extracted.get("themes", [])
    session.sentiment_sources = sources

    # Store recommended date range as a hint for the orchestrator
    # Orchestrator will use this to set date_range_start/end for market agent
    session._sentiment_recommended_days = int(
        extracted.get("recommended_date_range_days", 180)
    )

    # Escalate if confidence is too low
    if session.sentiment_confidence < SENTIMENT_CONFIDENCE_THRESHOLD:
        session.escalation_flags.append(
            f"sentiment_agent: low confidence ({session.sentiment_confidence:.2f}) — "
            f"results may be sparse or contradictory"
        )

    print(
        f"  [sentiment_agent] Done. Score: {session.sentiment_score}, "
        f"Confidence: {session.sentiment_confidence:.2f}"
    )
