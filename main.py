"""
main.py
-------
Argus CLI entry point.
Usage: python main.py
"""

import sys
from core.session import ResearchSession
from core.registry import register_all_agents
from core.audit import print_audit_summary
import orchestrator


def main():
    print("=" * 60)
    print("  ARGUS — Multi-Agent Financial Research System")
    print("=" * 60)

    ticker = input("\nEnter stock ticker (e.g. NVDA, AAPL, MSFT): ").upper().strip()
    if not ticker:
        print("No ticker entered. Exiting.")
        sys.exit(1)

    query = input(f"What do you want to know about {ticker}? : ").strip()
    if not query:
        query = f"Give me a comprehensive financial health assessment of {ticker}."

    # Initialize shared context
    session = ResearchSession(ticker=ticker, user_query=query)

    # Register all agents
    registry = register_all_agents()

    # Run orchestration loop
    orchestrator.run(session, registry)

    # Print results
    print("\n" + "=" * 60)
    print(f"  ARGUS RESEARCH REPORT — {ticker}")
    print("=" * 60)

    print(f"\n📊 SENTIMENT: {session.sentiment_score.upper()} "
          f"(confidence: {session.sentiment_confidence:.2f})")
    print(f"   {session.sentiment_summary}")

    print(f"\n📈 MARKET ({session.date_range_start} → {session.date_range_end}):")
    print(f"   Trend: {session.price_trend} | Change: {session.price_change_pct:+.1f}%")
    if session.chart_path:
        print(f"   Chart saved: {session.chart_path}")

    if session.rag_answers:
        print(f"\n📄 10-K INSIGHTS ({session.filing_year}):")
        for a in session.rag_answers:
            print(f"   Q: {a.question}")
            print(f"   A: {a.answer[:200]}{'...' if len(a.answer) > 200 else ''}")
            print(f"   Confidence: {a.confidence:.2f} | Page: {a.source_page}\n")

    print(f"\n🔍 SYNTHESIS (confidence: {session.confidence_overall:.2f}):")
    print(f"{session.final_answer}")

    if session.risk_flags:
        print(f"\n⚠ RISK FLAGS:")
        for flag in session.risk_flags:
            print(f"   • {flag}")

    if session.follow_up_questions:
        print(f"\n💡 SUGGESTED FOLLOW-UPS:")
        for q in session.follow_up_questions:
            print(f"   • {q}")

    # Print audit trail
    print_audit_summary(session)


if __name__ == "__main__":
    main()
