# -*- coding: utf-8 -*-
"""
Bridge to real DJ execution (v1 stub).

Future: run :class:`data_juicer.core.executor` on a sampled subset and collect
token usage from tracer / LLM ops.
"""

from __future__ import annotations

from typing import Any, Dict, List

from data_juicer.planner.contracts.recipe import DJExecutableConfig


def run_pipeline_sample_stub(
    cfg: DJExecutableConfig,
    *,
    sample_size: int = 10,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """Placeholder for sample execution; returns empty metrics."""
    return {
        "ok": True,
        "sample_size": sample_size,
        "random_seed": random_seed,
        "outputs": [],
    }


def collect_usage_stub() -> List[Dict[str, Any]]:
    """Placeholder token usage records."""
    return []
