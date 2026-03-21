# -*- coding: utf-8 -*-
"""Orchestrate directive-only, search-only, or sequential optimization."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from data_juicer.planner.contracts.recipe import DJExecutableConfig
from data_juicer.planner.optimize.directive_engine import DirectiveEngine, DirectiveEngineConfig, DirectiveEngineRun
from data_juicer.planner.optimize.evaluator import PipelineEvaluator, StubPipelineEvaluator
from data_juicer.planner.optimize.search.beam import BeamSearchConfig, BeamSearchOptimizer, CandidateRecord


class OptimizationRunMode(str, Enum):
    """Which stages to run."""

    DIRECTIVE_ONLY = "directive_only"
    SEARCH_ONLY = "search_only"
    DIRECTIVE_THEN_SEARCH = "directive_then_search"


@dataclass
class OptimizationRunnerResult:
    mode: OptimizationRunMode
    directive: Optional[DirectiveEngineRun] = None
    candidates: Optional[List[CandidateRecord]] = None


class OptimizationRunner:
    """High-level API combining stage-1 directives and stage-2 beam search."""

    def __init__(
        self,
        *,
        mode: OptimizationRunMode,
        directive_config: Optional[Dict[str, Any]] = None,
        beam_config: Optional[Dict[str, Any]] = None,
        evaluator: Optional[PipelineEvaluator] = None,
    ) -> None:
        self.mode = mode
        self._directive_cfg = directive_config or {}
        self._beam_cfg = beam_config or {}
        self._evaluator = evaluator or StubPipelineEvaluator()

    def run(self, cfg: DJExecutableConfig) -> OptimizationRunnerResult:
        current = cfg
        dir_run: Optional[DirectiveEngineRun] = None

        if self.mode in (OptimizationRunMode.DIRECTIVE_ONLY, OptimizationRunMode.DIRECTIVE_THEN_SEARCH):
            engine = DirectiveEngine(DirectiveEngineConfig.model_validate(self._directive_cfg))
            dir_run = engine.run(current)
            current = dir_run.config

        if self.mode in (OptimizationRunMode.SEARCH_ONLY, OptimizationRunMode.DIRECTIVE_THEN_SEARCH):
            beam = BeamSearchOptimizer(
                BeamSearchConfig.model_validate(self._beam_cfg),
                evaluator=self._evaluator,
            )
            candidates = beam.search(current)
            return OptimizationRunnerResult(
                mode=self.mode,
                directive=dir_run,
                candidates=candidates,
            )

        return OptimizationRunnerResult(mode=self.mode, directive=dir_run, candidates=None)
