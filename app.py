"""
app.py
------
Argus Streamlit UI.
Usage: streamlit run app.py
"""

import streamlit as st
import threading
from datetime import datetime

from core.session import ResearchSession
from core.registry import register_all_agents
import orchestrator

# ─── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Argus — Multi-Agent Financial Research",
    page_icon="👁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #1a1a2e;
        color: #cccccc;
        border-radius: 6px 6px 0 0;
        padding: 8px 20px;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #00bfff;
        color: #000000;
    }
    .metric-card {
        background-color: #1a1a2e;
        border: 1px solid #333355;
        border-radius: 8px;
        padding: 16px;
        margin: 4px 0;
    }
    .risk-flag {
        background-color: #2d1515;
        border-left: 3px solid #ef5350;
        padding: 8px 12px;
        border-radius: 0 6px 6px 0;
        margin: 4px 0;
        color: #ff8a80;
    }
    .audit-entry {
        background-color: #1a1a2e;
        border: 1px solid #333355;
        border-radius: 6px;
        padding: 12px;
        margin: 6px 0;
        font-size: 0.88em;
    }
    .agent-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.8em;
        font-weight: bold;
        margin-bottom: 6px;
    }
</style>
""", unsafe_allow_html=True)

# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/all-seeing-eye.png", width=64)
    st.title("Argus")
    st.caption("Multi-Agent Financial Research")
    st.divider()

    ticker = st.text_input(
        "Stock Ticker",
        placeholder="NVDA, AAPL, MSFT...",
        max_chars=10,
    ).upper().strip()

    query = st.text_area(
        "Research Question",
        placeholder="Should I be concerned about NVDA's financial position given current market conditions?",
        height=100,
    )

    run_button = st.button("🔍 Run Argus", type="primary", use_container_width=True)
    st.divider()

    st.caption("**How it works**")
    st.caption("1. Orchestrator plans the research")
    st.caption("2. Sentiment agent searches the web")
    st.caption("3. Orchestrator sets chart window + RAG questions")
    st.caption("4. Market agent generates price chart")
    st.caption("5. RAG agent answers from SEC 10-K filing")
    st.caption("6. Orchestrator synthesizes final answer")

# ─── Session state ────────────────────────────────────────────────────────────

if "session" not in st.session_state:
    st.session_state.session = None
if "running" not in st.session_state:
    st.session_state.running = False
if "registry" not in st.session_state:
    st.session_state.registry = register_all_agents()

# ─── Run pipeline ─────────────────────────────────────────────────────────────

if run_button and ticker and query:
    st.session_state.running = True
    session = ResearchSession(ticker=ticker, user_query=query)

    progress_container = st.container()
    with progress_container:
        st.info(f"🚀 Starting Argus research pipeline for **{ticker}**...")
        progress_bar = st.progress(0)
        status_text = st.empty()

        steps = [
            (0.1, "Orchestrator planning research sequence..."),
            (0.25, "Sentiment agent searching the web..."),
            (0.45, "Orchestrator reasoning: setting date range and generating RAG questions..."),
            (0.60, "Market agent fetching price data and generating chart..."),
            (0.75, "RAG agent downloading 10-K from SEC EDGAR..."),
            (0.90, "RAG agent answering targeted questions..."),
            (0.95, "Orchestrator synthesizing final answer..."),
        ]

        step_idx = [0]

        def advance_progress():
            if step_idx[0] < len(steps):
                pct, msg = steps[step_idx[0]]
                progress_bar.progress(pct)
                status_text.text(msg)
                step_idx[0] += 1

        advance_progress()

        try:
            orchestrator.run(session, st.session_state.registry)
            progress_bar.progress(1.0)
            status_text.text("✅ Research complete!")
            st.session_state.session = session
        except Exception as e:
            st.error(f"Pipeline error: {e}")
            st.session_state.running = False

    st.session_state.running = False

elif run_button and (not ticker or not query):
    st.warning("Please enter both a ticker and a research question.")

# ─── Results display ──────────────────────────────────────────────────────────

if st.session_state.session:
    s: ResearchSession = st.session_state.session

    st.divider()
    st.subheader(f"👁 Argus Research — {s.ticker}")
    st.caption(f"Session {s.session_id}  ·  {s.created_at[:19].replace('T', ' ')}")

    # Top-level metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        sentiment_color = {
            "positive": "🟢", "negative": "🔴", "mixed": "🟡", "neutral": "⚪"
        }.get(s.sentiment_score, "⚪")
        st.metric("Sentiment", f"{sentiment_color} {s.sentiment_score.upper()}")
    with col2:
        st.metric("Confidence", f"{s.sentiment_confidence:.0%}")
    with col3:
        trend_icon = {"uptrend": "↑", "downtrend": "↓", "sideways": "→"}.get(s.price_trend, "—")
        st.metric("Price Trend", f"{trend_icon} {s.price_trend}")
    with col4:
        st.metric("Price Change", f"{s.price_change_pct:+.1f}%")

    st.divider()

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Research Plan",
        "📰 Sentiment",
        "📈 Market View",
        "📄 10-K Insights",
        "🔍 Audit Trail",
    ])

    # ── Tab 1: Research Plan ──
    with tab1:
        st.subheader("Orchestrator Research Plan")
        if s.research_plan:
            cols = st.columns(len(s.research_plan))
            for i, (col, agent) in enumerate(zip(cols, s.research_plan)):
                with col:
                    st.markdown(f"""
                    <div class="metric-card" style="text-align:center;">
                        <div style="font-size:1.5em;">{'🧠' if i==0 else '📡' if 'sentiment' in agent else '📈' if 'market' in agent else '📄'}</div>
                        <div style="font-weight:bold; color:#00bfff; margin-top:8px;">{i+1}. {agent.replace('_', ' ').title()}</div>
                    </div>
                    """, unsafe_allow_html=True)

        if s.orchestrator_reasoning:
            st.markdown("**Orchestrator Reasoning**")
            st.info(s.orchestrator_reasoning)

        st.markdown("**Final Answer**")
        st.write(s.final_answer)

        if s.risk_flags:
            st.markdown("**Risk Flags**")
            for flag in s.risk_flags:
                st.markdown(f'<div class="risk-flag">⚠ {flag}</div>', unsafe_allow_html=True)

        if s.follow_up_questions:
            st.markdown("**Suggested Follow-Up Questions**")
            for q in s.follow_up_questions:
                st.markdown(f"💡 {q}")

    # ── Tab 2: Sentiment ──
    with tab2:
        st.subheader(f"Sentiment Analysis — {s.ticker}")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("**Summary**")
            st.write(s.sentiment_summary)
            if s.sentiment_themes:
                st.markdown("**Key Themes Identified**")
                for theme in s.sentiment_themes:
                    st.markdown(f"• {theme}")
        with col2:
            st.metric("Sentiment Score", s.sentiment_score.upper())
            st.metric("Confidence", f"{s.sentiment_confidence:.0%}")
            st.metric("Chart Window Set", f"{getattr(s, '_sentiment_recommended_days', 180)} days")

        if s.sentiment_sources:
            with st.expander("Source URLs"):
                for url in s.sentiment_sources:
                    if url:
                        st.markdown(f"- [{url[:60]}...]({url})")

    # ── Tab 3: Market View ──
    with tab3:
        st.subheader(f"Price Analysis — {s.ticker}")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Date Range", f"{s.date_range_start} → {s.date_range_end}")
        with col2:
            trend_icon = {"uptrend": "↑", "downtrend": "↓", "sideways": "→"}.get(s.price_trend, "—")
            st.metric("Trend", f"{trend_icon} {s.price_trend.upper()}")
        with col3:
            st.metric("Price Change", f"{s.price_change_pct:+.1f}%")

        if s.chart_path:
            st.image(s.chart_path, use_container_width=True)
        else:
            st.warning("Chart not available.")

        st.caption(
            f"📌 Date range was dynamically set by the orchestrator based on "
            f"{s.sentiment_score} sentiment signal — not hardcoded."
        )

    # ── Tab 4: 10-K Insights ──
    with tab4:
        st.subheader(f"10-K Filing Insights — {s.ticker} ({s.filing_year})")
        if s.filing_path:
            st.caption(f"Source: {s.filing_path}")

        st.markdown("**Questions were generated by the orchestrator based on sentiment themes:**")
        for theme in s.sentiment_themes[:3]:
            st.markdown(f"  → `{theme}`")

        st.divider()

        if s.rag_answers:
            for i, ans in enumerate(s.rag_answers, 1):
                with st.expander(f"Q{i}: {ans.question}", expanded=(i == 1)):
                    st.write(ans.answer)
                    col1, col2 = st.columns(2)
                    with col1:
                        st.caption(f"Confidence: {ans.confidence:.0%}")
                    with col2:
                        st.caption(f"Source page: {ans.source_page}")
        else:
            st.warning("No RAG answers available.")

    # ── Tab 5: Audit Trail ──
    with tab5:
        st.subheader("Agent Audit Trail")
        st.caption("Every agent dispatch, input, output, latency, and orchestrator reasoning — recorded.")

        agent_colors = {
            "sentiment_agent": "#1565c0",
            "market_agent": "#1b5e20",
            "rag_agent": "#4a148c",
            "orchestrator": "#e65100",
        }

        for i, entry in enumerate(s.audit_trail, 1):
            color = agent_colors.get(entry.agent, "#333333")
            status = "⚠ ESCALATED" if entry.escalated else "✓ OK"
            status_color = "#ef5350" if entry.escalated else "#66bb6a"

            st.markdown(f"""
            <div class="audit-entry">
                <span class="agent-badge" style="background-color:{color}; color:white;">
                    {entry.agent.replace('_', ' ').upper()}
                </span>
                <span style="float:right; color:{status_color}; font-size:0.85em;">{status} · {entry.latency_ms}ms</span>
                <div style="margin-top:6px;">
                    <strong style="color:#aaaaaa;">Task:</strong> {entry.task_assigned}
                </div>
                <div>
                    <strong style="color:#aaaaaa;">Reasoning:</strong> {entry.orchestrator_reasoning}
                </div>
                <div>
                    <strong style="color:#aaaaaa;">Output:</strong> {entry.output_summary}
                </div>
                {f'<div style="color:#ef5350; margin-top:4px;"><strong>Escalation:</strong> {entry.escalation_reason}</div>' if entry.escalated else ''}
            </div>
            """, unsafe_allow_html=True)

        if s.escalation_flags:
            st.divider()
            st.markdown("**All Escalation Flags**")
            for flag in s.escalation_flags:
                st.markdown(f'<div class="risk-flag">⚠ {flag}</div>', unsafe_allow_html=True)

else:
    # Landing state
    st.markdown("""
    <div style="text-align:center; padding:60px 20px;">
        <div style="font-size:4em;">👁</div>
        <h2 style="color:#00bfff;">Argus</h2>
        <p style="color:#888888; font-size:1.1em;">Multi-Agent Financial Research System</p>
        <p style="color:#666666;">Enter a ticker and research question in the sidebar to begin.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4 style="color:#00bfff;">📡 Sentiment Agent</h4>
            <p style="color:#888888; font-size:0.9em;">Searches the web for recent news and analyst commentary. Extracts sentiment score, confidence, and key themes via Gemini.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4 style="color:#00bfff;">📈 Market Agent</h4>
            <p style="color:#888888; font-size:0.9em;">Generates a price chart with MAs and volume. Date range is dynamically set by the orchestrator based on sentiment output — not hardcoded.</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h4 style="color:#00bfff;">📄 RAG Agent</h4>
            <p style="color:#888888; font-size:0.9em;">Auto-downloads the 10-K from SEC EDGAR. Answers questions generated by the orchestrator from sentiment themes — not generic user queries.</p>
        </div>
        """, unsafe_allow_html=True)
