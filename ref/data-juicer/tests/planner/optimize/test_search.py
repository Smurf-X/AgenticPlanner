# -*- coding: utf-8 -*-
"""Unit tests for search strategies."""

from __future__ import annotations

import pytest

from data_juicer.planner.contracts.recipe import DJExecutableConfig
from data_juicer.planner.optimize.evaluator import StubPipelineEvaluator
from data_juicer.planner.optimize.search import (
    BeamSearchConfig,
    BeamSearchStrategy,
    GreedySearchConfig,
    GreedySearchStrategy,
    RandomSearchConfig,
    RandomSearchStrategy,
    SearchStrategyType,
    create_search_strategy,
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


class TestGreedySearchStrategy:
    """Tests for GreedySearchStrategy."""

    def test_greedy_search(self, sample_config):
        """Test basic greedy search."""
        config = GreedySearchConfig(
            expansion_directives=["tighten_filters", "loosen_filters"],
            max_iterations=5,
        )
        evaluator = StubPipelineEvaluator()
        strategy = GreedySearchStrategy(config, evaluator)
        report = strategy.search(sample_config)

        assert report.ok
        assert len(report.candidates) > 0
        assert report.best_by_quality is not None

    def test_greedy_no_directives(self, sample_config):
        """Test greedy search with no directives."""
        config = GreedySearchConfig(
            expansion_directives=[],
            max_iterations=5,
        )
        evaluator = StubPipelineEvaluator()
        strategy = GreedySearchStrategy(config, evaluator)
        report = strategy.search(sample_config)

        assert report.ok
        # Should only have the root
        assert len(report.candidates) == 1
        assert report.candidates[0].origin == "root"

    def test_greedy_pareto_front(self, sample_config):
        """Test that Pareto front is computed."""
        config = GreedySearchConfig(
            expansion_directives=["tighten_filters", "loosen_filters"],
            max_iterations=3,
        )
        evaluator = StubPipelineEvaluator()
        strategy = GreedySearchStrategy(config, evaluator)
        report = strategy.search(sample_config)

        assert report.ok
        # Pareto front should be a subset of candidates
        for p in report.pareto_front:
            assert p in report.candidates


class TestRandomSearchStrategy:
    """Tests for RandomSearchStrategy."""

    def test_random_search(self, sample_config):
        """Test basic random search."""
        config = RandomSearchConfig(
            expansion_directives=["tighten_filters", "loosen_filters"],
            max_samples=10,
            max_depth=3,
        )
        evaluator = StubPipelineEvaluator()
        strategy = RandomSearchStrategy(config, evaluator)
        report = strategy.search(sample_config)

        assert report.ok
        # Should have root + samples
        assert len(report.candidates) >= 1
        assert report.total_evaluations == len(report.candidates)

    def test_random_search_reproducible(self, sample_config):
        """Test that random search is reproducible with same seed."""
        config1 = RandomSearchConfig(
            expansion_directives=["tighten_filters"],
            max_samples=5,
            seed=42,
        )
        config2 = RandomSearchConfig(
            expansion_directives=["tighten_filters"],
            max_samples=5,
            seed=42,
        )

        evaluator = StubPipelineEvaluator()
        strategy1 = RandomSearchStrategy(config1, evaluator, seed=42)
        strategy2 = RandomSearchStrategy(config2, evaluator, seed=42)

        report1 = strategy1.search(sample_config)
        report2 = strategy2.search(sample_config)

        # Should have same number of candidates
        assert len(report1.candidates) == len(report2.candidates)


class TestBeamSearchStrategy:
    """Tests for BeamSearchStrategy."""

    def test_beam_search(self, sample_config):
        """Test basic beam search."""
        config = BeamSearchConfig(
            beam_width=2,
            max_iterations=2,
            expansion_directives=["tighten_filters", "loosen_filters"],
        )
        evaluator = StubPipelineEvaluator()
        strategy = BeamSearchStrategy(config, evaluator)
        report = strategy.search(sample_config)

        assert report.ok
        assert len(report.candidates) > 0
        assert report.best_by_quality is not None

    def test_beam_search_pareto(self, sample_config):
        """Test beam search with Pareto tracking."""
        config = BeamSearchConfig(
            beam_width=3,
            max_iterations=2,
            expansion_directives=["tighten_filters", "loosen_filters"],
            track_pareto=True,
        )
        evaluator = StubPipelineEvaluator()
        strategy = BeamSearchStrategy(config, evaluator)
        report = strategy.search(sample_config)

        assert report.ok
        assert len(report.pareto_front) > 0

    def test_beam_search_cost_weight(self, sample_config):
        """Test beam search with cost weight."""
        config = BeamSearchConfig(
            beam_width=2,
            max_iterations=2,
            expansion_directives=["tighten_filters"],
            cost_weight=0.5,
        )
        evaluator = StubPipelineEvaluator()
        strategy = BeamSearchStrategy(config, evaluator)
        report = strategy.search(sample_config)

        assert report.ok
        assert report.best_by_cost is not None
        assert report.best_by_quality is not None


class TestSearchFactory:
    """Tests for create_search_strategy factory."""

    def test_create_greedy(self):
        """Test creating greedy strategy."""
        strategy = create_search_strategy(
            "greedy",
            {"max_iterations": 5, "expansion_directives": ["tighten_filters"]},
        )
        assert isinstance(strategy, GreedySearchStrategy)

    def test_create_random(self):
        """Test creating random strategy."""
        strategy = create_search_strategy(
            "random",
            {"max_samples": 10, "expansion_directives": ["tighten_filters"]},
        )
        assert isinstance(strategy, RandomSearchStrategy)

    def test_create_beam(self):
        """Test creating beam strategy."""
        strategy = create_search_strategy(
            "beam",
            {"beam_width": 4, "expansion_directives": ["tighten_filters"]},
        )
        assert isinstance(strategy, BeamSearchStrategy)

    def test_unknown_strategy(self):
        """Test creating unknown strategy raises error."""
        with pytest.raises(ValueError):
            create_search_strategy("unknown", {})


class TestSearchReport:
    """Tests for SearchReport."""

    def test_best_candidates(self, sample_config):
        """Test that best candidates are correctly identified."""
        config = BeamSearchConfig(
            beam_width=2,
            max_iterations=2,
            expansion_directives=["tighten_filters", "loosen_filters"],
        )
        evaluator = StubPipelineEvaluator()
        strategy = BeamSearchStrategy(config, evaluator)
        report = strategy.search(sample_config)

        # best_by_quality should have highest quality
        if report.best_by_quality:
            max_q = max(c.quality for c in report.candidates)
            assert report.best_by_quality.quality == max_q

        # best_by_cost should have lowest cost
        if report.best_by_cost:
            min_cost = min(
                c.cost.llm_token_cost + c.cost.wall_time_sec
                for c in report.candidates
            )
            best_cost = (
                report.best_by_cost.cost.llm_token_cost +
                report.best_by_cost.cost.wall_time_sec
            )
            assert best_cost == min_cost