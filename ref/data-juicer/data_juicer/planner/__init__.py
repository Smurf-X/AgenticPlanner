# -*- coding: utf-8 -*-
"""
Planner: natural-language recipe generation and optimization for Data-Juicer.

Public API (v1):
- contracts: DJ executable config helpers and evaluation types
- generate: NLRecipeGenerator, operator catalog
- optimize: DirectiveEngine, BeamSearchOptimizer, OptimizationRunner
"""

from data_juicer.planner.contracts import (
    CostBreakdown,
    DJExecutableConfig,
    EvalConfig,
    EvaluationMode,
    OperatorStep,
    PlanOperators,
    compute_token_cost,
    load_executable_config,
    plan_operators_to_process,
    process_to_plan_operators,
    save_executable_config,
)
from data_juicer.planner.generate import (
    NLRecipeGenerator,
    OpenAICompatibleJsonClient,
    assemble_executable_config,
    build_operator_catalog_text,
    build_operator_detail_text,
    build_schema_block,
    generate_recipe_from_llm_json_text,
    get_init_param_allowlist,
    sanitize_params,
)
from data_juicer.planner.optimize import (
    BeamSearchOptimizer,
    CandidateRecord,
    DirectiveEngine,
    OptimizationRunMode,
    OptimizationRunner,
)

__all__ = [
    "BeamSearchOptimizer",
    "CandidateRecord",
    "CostBreakdown",
    "DJExecutableConfig",
    "DirectiveEngine",
    "EvalConfig",
    "EvaluationMode",
    "NLRecipeGenerator",
    "OpenAICompatibleJsonClient",
    "OperatorStep",
    "OptimizationRunMode",
    "OptimizationRunner",
    "PlanOperators",
    "assemble_executable_config",
    "build_operator_catalog_text",
    "build_operator_detail_text",
    "build_schema_block",
    "get_init_param_allowlist",
    "sanitize_params",
    "compute_token_cost",
    "generate_recipe_from_llm_json_text",
    "load_executable_config",
    "plan_operators_to_process",
    "process_to_plan_operators",
    "save_executable_config",
]
