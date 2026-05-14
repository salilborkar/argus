"""
core/audit.py
-------------
Structured audit logger for Argus.
Every agent invocation is wrapped by the orchestrator using log_agent_call().
Produces a timestamped JSON log per session in logs/.
"""

import json
import time
import os
from datetime import datetime
from functools import wraps
from core.session import ResearchSession, AuditEntry


LOGS_DIR = "logs"


def log_agent_call(
    session: ResearchSession,
    agent_name: str,
    task_assigned: str,
    orchestrator_reasoning: str,
    fn,
    *args,
    **kwargs,
):
    """
    Execute an agent function and record the full audit entry.

    Usage in orchestrator:
        result = log_agent_call(
            session=session,
            agent_name="sentiment_agent",
            task_assigned="Analyze market sentiment for NVDA",
            orchestrator_reasoning="Running sentiment first to inform downstream date range",
            fn=sentiment_agent.run,
            session,
        )
    """
    start = time.time()
    escalated = False
    escalation_reason = ""
    output_summary = ""

    try:
        result = fn(*args, **kwargs)
        output_summary = _summarize_result(agent_name, session)
    except Exception as e:
        escalated = True
        escalation_reason = str(e)
        output_summary = f"ERROR: {str(e)}"
        session.escalation_flags.append(f"{agent_name}: {str(e)}")
        result = None

    latency_ms = int((time.time() - start) * 1000)

    entry = AuditEntry(
        timestamp=datetime.now().isoformat(),
        agent=agent_name,
        task_assigned=task_assigned,
        orchestrator_reasoning=orchestrator_reasoning,
        output_summary=output_summary,
        latency_ms=latency_ms,
        escalated=escalated,
        escalation_reason=escalation_reason,
    )

    session.audit_trail.append(entry)
    _persist_log(session)

    return result


def _summarize_result(agent_name: str, session: ResearchSession) -> str:
    """Generate a one-line summary of agent output for the audit log."""
    if agent_name == "sentiment_agent":
        return (
            f"{session.sentiment_score.upper()} sentiment, "
            f"{session.sentiment_confidence:.2f} confidence, "
            f"themes: {', '.join(session.sentiment_themes[:3])}"
        )
    elif agent_name == "market_agent":
        return (
            f"{session.price_trend} trend, "
            f"{session.price_change_pct:+.1f}% over period, "
            f"chart saved to {session.chart_path}"
        )
    elif agent_name == "rag_agent":
        answered = len([a for a in session.rag_answers if a.answer])
        return f"Answered {answered}/{len(session.filing_questions)} questions from {session.filing_year} 10-K"
    elif agent_name == "orchestrator":
        return f"Synthesized final answer, {len(session.risk_flags)} risk flags identified"
    return "Completed"


def _persist_log(session: ResearchSession):
    """Write the current session state to a JSON log file."""
    os.makedirs(LOGS_DIR, exist_ok=True)
    log_path = os.path.join(LOGS_DIR, f"{session.session_id}_{session.ticker}.json")
    with open(log_path, "w") as f:
        json.dump(session.to_dict(), f, indent=2)


def print_audit_summary(session: ResearchSession):
    """Print a formatted audit trail to the terminal."""
    print("\n" + "=" * 60)
    print(f"  ARGUS AUDIT TRAIL — Session {session.session_id}")
    print("=" * 60)
    for i, entry in enumerate(session.audit_trail, 1):
        status = "⚠ ESCALATED" if entry.escalated else "✓"
        print(f"\n[{i}] {status} {entry.agent.upper()}")
        print(f"    Task     : {entry.task_assigned}")
        print(f"    Reasoning: {entry.orchestrator_reasoning}")
        print(f"    Output   : {entry.output_summary}")
        print(f"    Latency  : {entry.latency_ms}ms")
        if entry.escalated:
            print(f"    Reason   : {entry.escalation_reason}")
    print("\n" + "=" * 60)
