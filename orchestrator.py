"""
orchestrator.py
---------------
The Argus Orchestrator.

This is the brain of the system. It:
  1. Decomposes the user query into a research plan
  2. Dispatches agents in the right order via the registry
  3. Reads each agent's output and reasons about what to do next
  4. Dynamically sets inputs for downstream agents based on prior agent results
  5. Synthesizes a final answer from all agent outputs
  6. Records every decision in the audit trail

Agents never call each other. The orchestrator mediates everything through ResearchSession.
"""

import json
from datetime import datetime, timedelta
import google.generativeai as genai

from core.session import ResearchSession
from core.audit import log_agent_call
from core.registry import AgentRegistry
from config import GEMINI_API_KEY, GEMINI_MODEL, ORCHESTRATOR_TEMPERATURE

genai.configure(api_key=GEMINI_API_KEY)
_model = genai.GenerativeModel(
    GEMINI_MODEL,
    generation_config={"temperature": ORCHESTRATOR_TEMPERATURE},
)


# ─── Step 1: Planning ──────────────────────────────────────────────────────────

def _plan(session: ResearchSession) -> list[str]:
    """
    Ask Gemini to decompose the user query into an ordered research plan.
    Returns a list of agent names to invoke, in order.
    """
    prompt = f"""
You are the orchestrator of a multi-agent financial research system called Argus.
Available agents: sentiment_agent, market_agent, rag_agent

User query: "{session.user_query}"
Ticker: {session.ticker}

Decide which agents to invoke and in what order to best answer the query.
Always run sentiment_agent first — its output informs the date range for market_agent
and the questions for rag_agent.

Respond ONLY with a JSON object (no markdown):
{{
  "plan": ["sentiment_agent", "market_agent", "rag_agent"],
  "reasoning": "<one sentence explaining the plan>"
}}
"""
    response = _model.generate_content(prompt)
    raw = response.text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    parsed = json.loads(raw)
    session.orchestrator_reasoning = parsed.get("reasoning", "")
    return parsed.get("plan", ["sentiment_agent", "market_agent", "rag_agent"])


# ─── Step 2: Post-sentiment reasoning ─────────────────────────────────────────

def _reason_after_sentiment(session: ResearchSession):
    """
    After sentiment agent runs, orchestrator decides:
      - What date range to give the market agent
      - What questions to generate for the RAG agent
    This is the key dynamic handoff that makes Argus genuinely multi-agent.
    """
    recommended_days = getattr(session, "_sentiment_recommended_days", 180)
    end_date = datetime.today()
    start_date = end_date - timedelta(days=recommended_days)
    session.date_range_start = start_date.strftime("%Y-%m-%d")
    session.date_range_end = end_date.strftime("%Y-%m-%d")

    print(
        f"  [orchestrator] Sentiment: {session.sentiment_score} "
        f"({session.sentiment_confidence:.2f} confidence). "
        f"Setting chart window to {recommended_days} days."
    )

    # Generate targeted RAG questions based on sentiment themes
    themes_str = ", ".join(session.sentiment_themes) if session.sentiment_themes else "general financials"

    prompt = f"""
You are a financial research orchestrator. Based on the following sentiment analysis 
of {session.ticker}, generate 4 targeted questions to ask a RAG system 
that has access to the company's most recent 10-K SEC filing.

Sentiment: {session.sentiment_score}
Key themes identified: {themes_str}
User's original query: "{session.user_query}"

Generate questions that directly address the themes and the user's concern.
Focus on what the 10-K would actually contain: risk factors, revenue breakdown, 
debt obligations, segment performance, management discussion.

Respond ONLY with a JSON object (no markdown):
{{
  "questions": [
    "<question 1>",
    "<question 2>",
    "<question 3>",
    "<question 4>"
  ],
  "reasoning": "<one sentence explaining why these questions address the sentiment themes>"
}}
"""
    response = _model.generate_content(prompt)
    raw = response.text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    parsed = json.loads(raw)
    session.filing_questions = parsed.get("questions", [])

    print(f"  [orchestrator] Generated {len(session.filing_questions)} RAG questions from sentiment themes.")
    for i, q in enumerate(session.filing_questions, 1):
        print(f"    Q{i}: {q}")


# ─── Step 3: Synthesis ─────────────────────────────────────────────────────────

def _synthesize(session: ResearchSession):
    """
    Final orchestrator step: combine all agent outputs into a structured answer.
    """
    rag_summary = "\n".join(
        f"  Q: {a.question}\n  A: {a.answer} (confidence: {a.confidence:.2f})"
        for a in session.rag_answers
    ) if session.rag_answers else "No RAG answers available."

    prompt = f"""
You are a senior financial analyst. Synthesize the following research findings 
into a clear, structured investment research summary for {session.ticker}.

USER QUERY: "{session.user_query}"

SENTIMENT ANALYSIS:
- Score: {session.sentiment_score} (confidence: {session.sentiment_confidence:.2f})
- Summary: {session.sentiment_summary}
- Key themes: {', '.join(session.sentiment_themes)}

MARKET DATA:
- Period: {session.date_range_start} to {session.date_range_end}
- Trend: {session.price_trend}
- Price change: {session.price_change_pct:+.1f}%

10-K FILING INSIGHTS ({session.filing_year}):
{rag_summary}

Provide a structured synthesis. Respond ONLY with a JSON object (no markdown):
{{
  "final_answer": "<3-5 paragraph synthesis directly addressing the user's query>",
  "risk_flags": ["<specific risk 1>", "<specific risk 2>", "<specific risk 3>"],
  "confidence_overall": <float 0.0-1.0>,
  "follow_up_questions": ["<suggested follow-up 1>", "<suggested follow-up 2>"]
}}
"""
    response = _model.generate_content(prompt)
    raw = response.text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    parsed = json.loads(raw)
    session.final_answer = parsed.get("final_answer", "")
    session.risk_flags = parsed.get("risk_flags", [])
    session.confidence_overall = float(parsed.get("confidence_overall", 0.5))
    session.follow_up_questions = parsed.get("follow_up_questions", [])


# ─── Main Orchestration Loop ───────────────────────────────────────────────────

def run(session: ResearchSession, registry: AgentRegistry):
    """
    Main orchestration loop.
    Called from main.py or app.py with a populated session and loaded registry.
    """
    print(f"\n[orchestrator] Starting research for {session.ticker}...")
    print(f"[orchestrator] Query: {session.user_query}")

    # Step 1: Plan
    print("\n[orchestrator] Planning research sequence...")
    plan = _plan(session)
    session.research_plan = plan
    print(f"[orchestrator] Plan: {' → '.join(plan)}")
    print(f"[orchestrator] Reasoning: {session.orchestrator_reasoning}")

    # Step 2: Execute plan
    for agent_name in plan:
        agent_fn = registry.get(agent_name)

        print(f"\n[orchestrator] Dispatching {agent_name}...")

        # Pre-dispatch reasoning for market and rag agents
        if agent_name == "market_agent" and session.sentiment_score:
            task = f"Generate price chart for {session.ticker} from {session.date_range_start} to {session.date_range_end}"
            reasoning = (
                f"Sentiment was {session.sentiment_score} with key themes: "
                f"{', '.join(session.sentiment_themes[:2])}. "
                f"Date range set to {getattr(session, '_sentiment_recommended_days', 180)} days "
                f"to capture the relevant price action."
            )
        elif agent_name == "rag_agent":
            task = f"Answer {len(session.filing_questions)} targeted questions from {session.ticker} 10-K"
            reasoning = (
                f"Questions generated from sentiment themes: "
                f"{', '.join(session.sentiment_themes[:2])}. "
                f"Filing will provide fundamental context to validate/contrast market sentiment."
            )
        else:
            task = f"Analyze market sentiment for {session.ticker}"
            reasoning = "Sentiment runs first — its output drives date range and RAG question generation."

        log_agent_call(
            session,
            agent_name,
            task,
            reasoning,
            agent_fn,
            session,
        )

        # Post-dispatch reasoning: after sentiment, set downstream inputs
        if agent_name == "sentiment_agent":
            _reason_after_sentiment(session)

        # Check for escalations after each agent
        if session.escalation_flags:
            new_flags = [
                f for f in session.escalation_flags
                if f.startswith(agent_name)
            ]
            if new_flags:
                print(f"  [orchestrator] ⚠ Escalation from {agent_name}: {new_flags[-1]}")

    # Step 3: Synthesize
    print("\n[orchestrator] Synthesizing final answer...")
    log_agent_call(
        session,
        "orchestrator",
        f"Synthesize research findings for {session.ticker}",
        "All agents complete. Combining sentiment, market, and filing data into final answer.",
        _synthesize,
        session,
    )

    print(f"\n[orchestrator] Research complete. Overall confidence: {session.confidence_overall:.2f}")
    return session
