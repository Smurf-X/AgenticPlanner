# -*- coding: utf-8 -*-
"""
Beam search strategy for pipeline optimization.

Beam search maintains top-k candidates at each iteration.
"""

from __future__ import annotations

import random
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field

from agentic_planner.contracts.cost import CostBreakdown
from agentic_planner.contracts.recipe import DJExecutableConfig, validate_executable_config
from agentic_planner.optimizer.directives.base import DirectiveResult
from agentic_planner.optimizer.directives.registry import DIRECTIVE_REGISTRY
from agentic_planner.optimizer.search.base import (
    BaseSearchStrategy,
    SearchConfig,
    SearchReport,
    SearchResult,
    SearchStrategyType,
)


class BeamSearchConfig(BaseModel):
    """Configuration for beam search."""

    beam_width: int = Field(default=4, ge=1, le=64)
    max_iterations: int = Field(default=3, ge=1, le=50)
    expansion_directives: List[str] = Field(
        default_factory=list,
        description="Directive keys tried as one-step neighbors from each beam.",
    )
    track_pareto: bool = Field(default=True, description="Track Pareto frontier during search.")
    seed: int = Field(default=42)

    model_config = {"extra": "allow"}


@dataclass
class BeamCandidate:
    """Internal representation of a beam candidate."""

    config: DJExecutableConfig
    cost: CostBreakdown
    quality: float
    origin: str
    trace: List[DirectiveResult]
    config_hash: str = ""

    def __post_init__(self):
        if not self.config_hash:
            self.config_hash = self._hash_config(self.config)

    @staticmethod
    def _hash_config(cfg: DJExecutableConfig) -> str:
        """Generate a hash for configuration deduplication."""
        import hashlib
        import json
        content = json.dumps(cfg.get("process", []), sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()[:12]


class BeamSearchStrategy(BaseSearchStrategy):
    """
    Beam search with multi-objective optimization.

    Maintains top-k candidates at each iteration and explores their neighbors.
    """

    def __init__(
        self,
        config: BeamSearchConfig,
        evaluator: Optional[Any] = None,
    ) -> None:
        super().__init__(
            SearchConfig(strategy=SearchStrategyType.BEAM),
            evaluator,
        )
        self._beam_config = config
        self._rng = random.Random(config.seed)
        self._seen_hashes: Set[str] = set()

    def search(self, root: DJExecutableConfig) -> SearchReport:
        """Execute beam search."""
        errors = validate_executable_config(root)
        if errors:
            return SearchReport(
                ok=False,
                candidates=[],
                errors=["Invalid root config: " + "; ".join(errors)],
            )

        all_candidates: List[SearchResult] = []

        # Evaluate root
        root_cost, root_quality = self._evaluate(root)
        root_beam = BeamCandidate(
            config=deepcopy(root),
            cost=root_cost,
            quality=root_quality,
            origin="root",
            trace=[],
        )
        self._seen_hashes.add(root_beam.config_hash)

        beams: List[BeamCandidate] = [root_beam]
        all_candidates.append(self._beam_to_result(root_beam, 0))

        for iteration in range(self._beam_config.max_iterations):
            self._iteration_count = iteration + 1
            next_beams: List[BeamCandidate] = []

            for beam in beams:
                for dname in self._beam_config.expansion_directives:
                    if self._evaluated_count >= self._config.max_evaluations:
                        break

                    directive = DIRECTIVE_REGISTRY.get(dname)
                    if directive is None:
                        continue

                    step = directive.apply(beam.config)
                    if not step.ok or not step.applied or not step.config_after:
                        continue
                    if validate_executable_config(step.config_after):
                        continue

                    # Check for duplicates
                    new_beam = BeamCandidate(
                        config=step.config_after,
                        cost=CostBreakdown(),  # Will be filled by evaluator
                        quality=0.0,
                        origin=f"{beam.origin}+{dname}",
                        trace=beam.trace + [step],
                    )

                    if new_beam.config_hash in self._seen_hashes:
                        continue
                    self._seen_hashes.add(new_beam.config_hash)

                    # Evaluate
                    new_cost, new_quality = self._evaluate(new_beam.config)
                    new_beam.cost = new_cost
                    new_beam.quality = new_quality

                    next_beams.append(new_beam)
                    all_candidates.append(self._beam_to_result(new_beam, iteration + 1))

            if not next_beams:
                break

            # Select top-k by quality
            next_beams.sort(key=lambda b: b.quality, reverse=True)
            beams = next_beams[:self._beam_config.beam_width]

        # Compute Pareto front
        pareto = self._compute_pareto_front(all_candidates)

        return SearchReport(
            ok=True,
            candidates=all_candidates,
            pareto_front=pareto,
            total_iterations=self._iteration_count,
            total_evaluations=self._evaluated_count,
            best_by_quality=self._find_best_by_quality(all_candidates),
            best_by_cost=self._find_best_by_cost(all_candidates),
            best_balanced=self._find_best_balanced(all_candidates),
            metrics={
                "beam_width": self._beam_config.beam_width,
                "unique_configs": len(self._seen_hashes),
            },
        )

    def _beam_to_result(self, beam: BeamCandidate, generation: int) -> SearchResult:
        """Convert BeamCandidate to SearchResult."""
        return SearchResult(
            config=beam.config,
            cost=beam.cost,
            quality=beam.quality,
            origin=beam.origin,
            trace=beam.trace,
            generation=generation,
        )


__all__ = [
    "BeamSearchConfig",
    "BeamSearchStrategy",
    "BeamCandidate",
]