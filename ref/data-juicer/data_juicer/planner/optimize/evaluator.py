# -*- coding: utf-8 -*-
"""Sample-based evaluation: cost + quality (pluggable)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from data_juicer.planner.contracts.cost import CostBreakdown
from data_juicer.planner.contracts.eval_protocol import EvalConfig
from data_juicer.planner.contracts.recipe import DJExecutableConfig


@runtime_checkable
class PipelineEvaluator(Protocol):
    """Evaluate a candidate config on a sample (implementation may run DJ executor or mock)."""

    def evaluate(self, cfg: DJExecutableConfig) -> tuple[CostBreakdown, float]:
        """Return cost breakdown and mean quality in ``[0, 1]`` (or unconstrained float)."""


class StubPipelineEvaluator:
    """Deterministic stub for tests: quality from hash, zero cost."""

    def __init__(self, eval_config: EvalConfig | None = None) -> None:
        self.eval_config = eval_config or EvalConfig()

    def evaluate(self, cfg: DJExecutableConfig) -> tuple[CostBreakdown, float]:
        blob = str(cfg.get("process", [])).encode()
        q = (hash(blob) % 10000) / 10000.0
        cost = CostBreakdown(llm_token_cost=0.0, wall_time_sec=0.0)
        return cost, q
