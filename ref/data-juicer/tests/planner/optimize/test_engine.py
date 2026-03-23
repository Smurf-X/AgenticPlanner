# -*- coding: utf-8 -*-
"""Unit tests for directive engine and optimization runner."""

from __future__ import annotations

import pytest

from data_juicer.planner.contracts.recipe import DJExecutableConfig
from data_juicer.planner.optimize.directive_engine import (
    DirectiveEngine,
    DirectiveEngineConfig,
    DirectiveEngineMode,
    apply_static_directives,
)
from data_juicer.planner.optimize.evaluator import (
    EvaluationResult,
    LlmJudgeEvaluator,
    StubPipelineEvaluator,
)
from data_juicer.planner.optimize.optimization_config import (
    OptimizationConfig,
    PriceTable,
    load_config,
)
from data_juicer.planner.optimize.runner import (
    OptimizationRunner,
    OptimizationRunMode,
    optimize_pipeline,
)
from data_juicer.planner.optimize.search.beam import (
    BeamSearchConfig,
    BeamSearchOptimizer,
    CandidateRecord,
)


@pytest.fixture
def sample_config() -> DJExecutableConfig:
    """Sample pipeline configuration for testing."""
    return {
        "dataset_path": "/tmp/test.jsonl",
        "process": [
            {"text_length_filter": {"min_len": 10, "max_len": 1000}},
            {"language_id_score_filter": {"lang": "en"}},
        ],
    }


class TestDirectiveEngine:
    """Tests for DirectiveEngine."""

    def test_static_mode(self, sample_config):
        """Test running directives in static mode."""
        config = DirectiveEngineConfig(
            mode=DirectiveEngineMode.STATIC,
            directives=["reorder_filters_first", "remove_redundant_ops"],
        )
        engine = DirectiveEngine(config)
        result = engine.run(sample_config)

        assert result.ok
        assert len(result.trace) == 2

    def test_apply_static_directives(self, sample_config):
        """Test convenience function for static directives."""
        result = apply_static_directives(
            sample_config,
            ["reorder_filters_first"],
        )

        assert result.ok
        assert len(result.trace) == 1

    def test_unknown_directive(self, sample_config):
        """Test handling unknown directive."""
        config = DirectiveEngineConfig(
            mode=DirectiveEngineMode.STATIC,
            directives=["unknown_directive"],
        )
        engine = DirectiveEngine(config)
        result = engine.run(sample_config)

        assert not result.ok
        assert "unknown directive" in result.errors[0]

    def test_max_steps(self, sample_config):
        """Test max_steps limit."""
        config = DirectiveEngineConfig(
            mode=DirectiveEngineMode.STATIC,
            directives=["reorder_filters_first"] * 10,
            max_steps=2,
        )
        engine = DirectiveEngine(config)
        result = engine.run(sample_config)

        # Should stop after max_steps
        assert len(result.trace) <= 2


class TestStubPipelineEvaluator:
    """Tests for StubPipelineEvaluator."""

    def test_evaluate(self, sample_config):
        """Test stub evaluation."""
        evaluator = StubPipelineEvaluator()
        cost, quality = evaluator.evaluate(sample_config)

        assert cost.llm_token_cost == 0.0
        assert 0.0 <= quality <= 1.0

    def test_deterministic(self, sample_config):
        """Test that stub evaluation is deterministic."""
        evaluator = StubPipelineEvaluator()
        _, q1 = evaluator.evaluate(sample_config)
        _, q2 = evaluator.evaluate(sample_config)

        assert q1 == q2


class TestBeamSearchOptimizer:
    """Tests for BeamSearchOptimizer."""

    def test_search(self, sample_config):
        """Test beam search optimization."""
        config = BeamSearchConfig(
            beam_width=2,
            max_iterations=2,
            expansion_directives=["tighten_filters", "loosen_filters"],
        )
        evaluator = StubPipelineEvaluator()
        optimizer = BeamSearchOptimizer(config, evaluator=evaluator)
        candidates = optimizer.search(sample_config)

        assert len(candidates) > 0
        # Results should be sorted by quality
        for i in range(len(candidates) - 1):
            assert candidates[i].quality >= candidates[i + 1].quality

    def test_candidate_record(self):
        """Test CandidateRecord dataclass."""
        from data_juicer.planner.contracts.cost import CostBreakdown

        record = CandidateRecord(
            config={"process": []},
            cost=CostBreakdown(llm_token_cost=0.1),
            quality=0.8,
            origin="root+test",
        )

        assert record.quality == 0.8
        assert record.cost.llm_token_cost == 0.1


class TestOptimizationRunner:
    """Tests for OptimizationRunner."""

    def test_directive_only_mode(self, sample_config):
        """Test directive-only optimization."""
        runner = OptimizationRunner(
            mode=OptimizationRunMode.DIRECTIVE_ONLY,
            directive_config={
                "mode": DirectiveEngineMode.STATIC,
                "directives": ["reorder_filters_first"],
            },
        )
        result = runner.run(sample_config)

        assert result.ok
        assert result.directive is not None
        assert result.candidates is None

    def test_search_only_mode(self, sample_config):
        """Test search-only optimization."""
        evaluator = StubPipelineEvaluator()
        runner = OptimizationRunner(
            mode=OptimizationRunMode.SEARCH_ONLY,
            evaluator=evaluator,
            beam_config={
                "beam_width": 2,
                "max_iterations": 1,
                "expansion_directives": ["tighten_filters"],
            },
        )
        result = runner.run(sample_config)

        assert result.ok
        assert result.candidates is not None

    def test_optimize_pipeline_convenience(self, sample_config):
        """Test optimize_pipeline convenience function."""
        result = optimize_pipeline(
            sample_config,
            mode="directive_only",
            directives=["reorder_filters_first"],
        )

        assert result.ok


class TestOptimizationConfig:
    """Tests for OptimizationConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = OptimizationConfig()
        assert config.run_mode == "directive_only"
        assert config.evaluation.sample_size == 50

    def test_from_yaml(self):
        """Test parsing from YAML."""
        yaml_str = """
run_mode: directive_only
directive:
  mode: static
  directives:
    - reorder_filters_first
"""
        config = OptimizationConfig.from_yaml(yaml_str)
        assert config.run_mode == "directive_only"
        assert config.directive.directives == ["reorder_filters_first"]

    def test_to_yaml(self):
        """Test serializing to YAML."""
        config = OptimizationConfig()
        yaml_str = config.to_yaml()
        assert "run_mode:" in yaml_str

    def test_price_table(self):
        """Test price table."""
        prices = PriceTable()
        assert prices.get_price("gpt-4o-mini") == 0.15
        assert prices.get_price("unknown_model") == 0.0


class TestLlmJudgeEvaluator:
    """Tests for LlmJudgeEvaluator."""

    def test_judge_prompt_building(self):
        """Test that judge prompts are built correctly."""
        from data_juicer.planner.contracts.eval_protocol import EvalConfig

        eval_config = EvalConfig(
            task_description="Test task",
        )
        evaluator = LlmJudgeEvaluator(eval_config)

        # Check that prompts are defined
        assert evaluator.JUDGE_SYSTEM_PROMPT != ""
        assert evaluator.JUDGE_USER_PROMPT != ""

    def test_stub_evaluation(self):
        """Test stub evaluation without LLM client."""
        from data_juicer.planner.contracts.eval_protocol import EvalConfig

        eval_config = EvalConfig()
        evaluator = LlmJudgeEvaluator(eval_config)

        # Without LLM client, should raise or handle gracefully
        # In this case we just test the class can be instantiated
        assert evaluator._llm_client is None