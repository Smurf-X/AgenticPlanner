# -*- coding: utf-8 -*-
"""
Agentic Planner - Natural language pipeline design and optimization.

This package provides:
1. Generator: Convert natural language to Data-Juicer YAML configs
2. Optimizer: Optimize pipeline configurations using directives and search

Usage:
    from agentic_planner import PipelineGenerator, PipelineOptimizer
    
    # Generate from natural language
    generator = PipelineGenerator(llm_client=openai_client)
    config = generator.generate("Filter short texts and extract keywords")
    
    # Optimize the config
    optimizer = PipelineOptimizer(mode="directive_then_search")
    result = optimizer.run(config)
"""

from agentic_planner.contracts.recipe import (
    DJExecutableConfig,
    load_executable_config,
    save_executable_config,
    validate_executable_config,
)
from agentic_planner.contracts.cost import (
    CostBreakdown,
    compute_token_cost,
)
from agentic_planner.contracts.eval_protocol import (
    EvalConfig,
    EvaluationMode,
)

__all__ = [
    # Contracts
    "DJExecutableConfig",
    "load_executable_config",
    "save_executable_config",
    "validate_executable_config",
    "CostBreakdown",
    "compute_token_cost",
    "EvalConfig",
    "EvaluationMode",
    # Generator (lazy import to avoid circular deps)
    # "PipelineGenerator",
    # Optimizer (lazy import)
    # "PipelineOptimizer",
]

__version__ = "0.1.0"


def get_generator():
    """Lazy import for generator."""
    from agentic_planner.generator import PipelineGenerator
    return PipelineGenerator


def get_optimizer():
    """Lazy import for optimizer."""
    from agentic_planner.optimizer import PipelineOptimizer
    return PipelineOptimizer