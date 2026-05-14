"""
core/registry.py
----------------
Agent registry for Argus.
The orchestrator dispatches agents by name through this registry.
Keeps orchestrator.py clean — it never imports agents directly.
"""

from typing import Callable
from core.session import ResearchSession


class AgentRegistry:
    """
    Lightweight registry that maps agent names to their run() functions.
    All agents must accept a single argument: ResearchSession.
    """

    def __init__(self):
        self._agents: dict[str, Callable] = {}

    def register(self, name: str, fn: Callable):
        self._agents[name] = fn

    def get(self, name: str) -> Callable:
        if name not in self._agents:
            raise ValueError(f"Agent '{name}' not found in registry. Registered: {list(self._agents.keys())}")
        return self._agents[name]

    def list_agents(self) -> list[str]:
        return list(self._agents.keys())


# Global registry instance — imported by orchestrator and main.py
registry = AgentRegistry()


def register_all_agents():
    """
    Import and register all agents.
    Called once at startup in main.py and app.py.
    Keeps imports lazy — agents aren't loaded until needed.
    """
    from agents.sentiment_agent import run as sentiment_run
    from agents.market_agent import run as market_run
    from agents.rag_agent import run as rag_run

    registry.register("sentiment_agent", sentiment_run)
    registry.register("market_agent", market_run)
    registry.register("rag_agent", rag_run)

    return registry
