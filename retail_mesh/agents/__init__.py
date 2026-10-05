"""Sub-agents. Each one is its own compiled LangGraph graph with its own state.

The orchestrator only knows how to call `GRAPH.invoke(...)` and read back an
AgentResult. That boundary is what lets you later move an agent into its own
service (A2A, MCP, a separate team's deployment) without touching the router.
"""
