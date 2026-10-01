"""synthetic stub: LLM CLI runner interface (no implementation shipped).

``run_llm_cli`` is called by the orderer/auditor steps. Plug in your own LLM backend.
"""
from __future__ import annotations


def run_llm_cli(system: str, prompt: str, *, model: str | None = None,
                timeout: int = 600, cwd: str | None = None) -> str:
    """Return the model's text output. Not implemented in this public repo."""
    raise NotImplementedError("synthetic stub: connect your own LLM backend")
