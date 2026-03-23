# -*- coding: utf-8 -*-
"""Pipeline optimization module for Data-Juicer.

This module provides:
- Directive-based optimization (Stage 1)
- Search-based optimization (Stage 2)
- Evaluation with LLM-as-a-judge
- Cost tracking and quality scoring
- Stable operator location via OpLocator
"""

from data_juicer.planner.optimize.directive_engine import (
    DirectiveEngine,
    DirectiveEngineConfig,
    DirectiveEngineMode,
    DirectiveEngineRun,
    apply_static_directives,
    optimize_with_inference,
)
from data_juicer.planner.optimize.directive_inference import (
    DirectiveInferenceEngine,
    DirectiveRecommendation,
    DirectiveRecommendations,
    InferredDirectiveEngineRun,
)
from data_juicer.planner.optimize.evaluator import (
    BaseEvaluator,
    EvaluationResult,
    LlmJudgeEvaluator,
    PipelineEvaluator,
    RealPipelineEvaluator,
    StubPipelineEvaluator,
    create_evaluator,
)
from data_juicer.planner.optimize.executor_adapter import (
    DJExecutorAdapter,
    ExecutorAdapter,
    SampleExecutionResult,
    StubExecutorAdapter,
    TokenUsageCollector,
    TokenUsageRecord,
    create_real_adapter,
    create_stub_adapter,
)
from data_juicer.planner.optimize.op_locator import (
    OpIdentity,
    OpLocator,
    ProcessIndex,
)
from data_juicer.planner.optimize.optimization_config import (
    CONFIG_TEMPLATE,
    DEFAULT_DIRECTIVE_ONLY_CONFIG,
    DEFAULT_FULL_CONFIG,
    DEFAULT_INFERENCE_CONFIG,
    DEFAULT_SEARCH_CONFIG,
    ExecutionConfig,
    LLMConfig,
    OptimizationConfig,
    PriceTable,
    create_sample_config_file,
    load_config,
)
from data_juicer.planner.optimize.runner import (
    OptimizationRunMode,
    OptimizationRunner,
    OptimizationRunnerResult,
)
from data_juicer.planner.optimize.search import (
    BaseSearchStrategy,
    BeamSearchConfig,
    BeamSearchOptimizer,
    BeamSearchStrategy,
    GreedySearchConfig,
    GreedySearchStrategy,
    OptimizationObjective,
    RandomSearchConfig,
    RandomSearchStrategy,
    SearchConfig,
    SearchReport,
    SearchResult,
    SearchStrategyType,
    create_search_strategy,
)

__all__ = [
    # Core types
    "OptimizationRunMode",
    "OptimizationRunner",
    "OptimizationRunnerResult",
    "PipelineEvaluator",
    # Operator location
    "OpIdentity",
    "OpLocator",
    "ProcessIndex",
    # Directive engine
    "DirectiveEngine",
    "DirectiveEngineConfig",
    "DirectiveEngineMode",
    "DirectiveEngineRun",
    "apply_static_directives",
    "optimize_with_inference",
    # Directive inference
    "DirectiveInferenceEngine",
    "DirectiveRecommendation",
    "DirectiveRecommendations",
    "InferredDirectiveEngineRun",
    # Evaluators
    "BaseEvaluator",
    "EvaluationResult",
    "LlmJudgeEvaluator",
    "RealPipelineEvaluator",
    "StubPipelineEvaluator",
    "create_evaluator",
    # Executor adapter
    "ExecutorAdapter",
    "DJExecutorAdapter",
    "StubExecutorAdapter",
    "SampleExecutionResult",
    "TokenUsageCollector",
    "TokenUsageRecord",
    "create_real_adapter",
    "create_stub_adapter",
    # Configuration
    "OptimizationConfig",
    "ExecutionConfig",
    "LLMConfig",
    "PriceTable",
    "load_config",
    "create_sample_config_file",
    "CONFIG_TEMPLATE",
    "DEFAULT_DIRECTIVE_ONLY_CONFIG",
    "DEFAULT_FULL_CONFIG",
    "DEFAULT_INFERENCE_CONFIG",
    "DEFAULT_SEARCH_CONFIG",
    # Search - base types
    "BaseSearchStrategy",
    "OptimizationObjective",
    "SearchConfig",
    "SearchReport",
    "SearchResult",
    "SearchStrategyType",
    "create_search_strategy",
    # Search - greedy
    "GreedySearchConfig",
    "GreedySearchStrategy",
    # Search - random
    "RandomSearchConfig",
    "RandomSearchStrategy",
    # Search - beam
    "BeamSearchConfig",
    "BeamSearchOptimizer",
    "BeamSearchStrategy",
]