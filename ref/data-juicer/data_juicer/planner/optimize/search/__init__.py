# -*- coding: utf-8 -*-
"""
Search strategies for pipeline optimization.

This module provides pluggable search strategies:
- Greedy: Hill-climbing, always pick best immediate improvement
- Random: Random sampling of configurations
- Beam: Maintain top-k candidates at each iteration
"""

from data_juicer.planner.optimize.search.base import (
    BaseSearchStrategy,
    OptimizationObjective,
    SearchConfig,
    SearchReport,
    SearchResult,
    SearchStrategy,
    SearchStrategyType,
)
from data_juicer.planner.optimize.search.beam import (
    BeamCandidate,
    BeamSearchConfig,
    BeamSearchOptimizer,
    BeamSearchStrategy,
)
from data_juicer.planner.optimize.search.greedy import (
    GreedySearchConfig,
    GreedySearchStrategy,
)
from data_juicer.planner.optimize.search.random import (
    RandomSearchConfig,
    RandomSearchStrategy,
)

__all__ = [
    # Base types
    "BaseSearchStrategy",
    "OptimizationObjective",
    "SearchConfig",
    "SearchReport",
    "SearchResult",
    "SearchStrategy",
    "SearchStrategyType",
    # Greedy search
    "GreedySearchConfig",
    "GreedySearchStrategy",
    # Random search
    "RandomSearchConfig",
    "RandomSearchStrategy",
    # Beam search
    "BeamCandidate",
    "BeamSearchConfig",
    "BeamSearchOptimizer",
    "BeamSearchStrategy",
]


def create_search_strategy(
    strategy_type: str,
    config: dict,
    evaluator=None,
) -> BaseSearchStrategy:
    """
    Factory function to create search strategies.

    Args:
        strategy_type: Type of search ("greedy", "random", "beam")
        config: Configuration dict for the strategy
        evaluator: Evaluator for scoring configurations

    Returns:
        A search strategy instance
    """
    if strategy_type == "greedy":
        return GreedySearchStrategy(
            GreedySearchConfig.model_validate(config),
            evaluator,
        )
    elif strategy_type == "random":
        return RandomSearchStrategy(
            RandomSearchConfig.model_validate(config),
            evaluator,
        )
    elif strategy_type == "beam":
        return BeamSearchStrategy(
            BeamSearchConfig.model_validate(config),
            evaluator,
        )
    else:
        raise ValueError(f"Unknown search strategy: {strategy_type}")