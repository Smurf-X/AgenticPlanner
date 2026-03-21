# -*- coding: utf-8 -*-
"""Beam search over directive expansions (v1 simplified local search)."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from pydantic import BaseModel, Field

from data_juicer.planner.contracts.cost import CostBreakdown
from data_juicer.planner.contracts.eval_protocol import EvalConfig
from data_juicer.planner.contracts.recipe import DJExecutableConfig, validate_executable_config
from data_juicer.planner.optimize.directives.base import DirectiveResult
from data_juicer.planner.optimize.directives.registry import DIRECTIVE_REGISTRY
from data_juicer.planner.optimize.evaluator import PipelineEvaluator, StubPipelineEvaluator


class BeamSearchConfig(BaseModel):
    beam_width: int = Field(default=4, ge=1, le=64)
    max_iterations: int = Field(default=3, ge=1, le=50)
    expansion_directives: List[str] = Field(
        default_factory=list,
        description="Directive keys tried as one-step neighbors from each beam.",
    )

    model_config = {"extra": "allow"}


@dataclass
class CandidateRecord:
    """One scored configuration."""

    config: DJExecutableConfig
    cost: CostBreakdown
    quality: float
    origin: str = ""
    trace: List[DirectiveResult] = field(default_factory=list)


class BeamSearchOptimizer:
    """
    Expand neighbors by applying each expansion directive once; keep top-k by quality.

    This is a lightweight local search, not full MOAR/MCTS.
    """

    def __init__(
        self,
        beam_config: BeamSearchConfig,
        evaluator: PipelineEvaluator | None = None,
        eval_config: EvalConfig | None = None,
    ) -> None:
        self._beam = beam_config
        self._evaluator = evaluator or StubPipelineEvaluator(eval_config)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BeamSearchOptimizer":
        return cls(BeamSearchConfig.model_validate(data))

    def search(self, root: DJExecutableConfig) -> List[CandidateRecord]:
        errors = validate_executable_config(root)
        if errors:
            raise ValueError("root config invalid: " + "; ".join(errors))

        beams: List[Tuple[DJExecutableConfig, str]] = [(deepcopy(root), "root")]
        scored: List[CandidateRecord] = []

        root_cost, root_q = self._evaluator.evaluate(root)
        scored.append(CandidateRecord(config=deepcopy(root), cost=root_cost, quality=root_q, origin="root"))

        for _ in range(self._beam_config.max_iterations):
            next_beams: List[Tuple[DJExecutableConfig, str, float]] = []
            for cfg, label in beams:
                for dname in self._beam_config.expansion_directives:
                    directive = DIRECTIVE_REGISTRY.get(dname)
                    if directive is None:
                        continue
                    res = directive.apply(cfg)
                    if not res.ok or res.config_after is None:
                        continue
                    child = res.config_after
                    if validate_executable_config(child):
                        continue
                    c, q = self._evaluator.evaluate(child)
                    scored.append(
                        CandidateRecord(
                            config=child,
                            cost=c,
                            quality=q,
                            origin=f"{label}+{dname}",
                            trace=[res],
                        )
                    )
                    next_beams.append((child, f"{label}+{dname}", q))

            if not next_beams:
                break
            next_beams.sort(key=lambda x: x[2], reverse=True)
            beams = [(x[0], x[1]) for x in next_beams[: self._beam_config.beam_width]]

        scored.sort(key=lambda r: r.quality, reverse=True)
        return scored
