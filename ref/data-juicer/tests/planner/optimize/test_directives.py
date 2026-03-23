# -*- coding: utf-8 -*-
"""Unit tests for optimization directives."""

from __future__ import annotations

import pytest

from data_juicer.planner.contracts.recipe import DJExecutableConfig
from data_juicer.planner.optimize.directives.adjust_params import BumpMinLenDirective
from data_juicer.planner.optimize.directives.adjust_threshold import (
    AdjustThresholdDirective,
    LoosenFiltersDirective,
    TightenFiltersDirective,
)
from data_juicer.planner.optimize.directives.base import DirectiveResult
from data_juicer.planner.optimize.directives.gleaning import (
    AddGleaningDirective,
    RemoveGleaningDirective,
)
from data_juicer.planner.optimize.directives.remove_redundant import (
    RemoveRedundantOpDirective,
)
from data_juicer.planner.optimize.directives.reorder import ReorderFiltersFirstDirective
from data_juicer.planner.optimize.directives.registry import (
    DIRECTIVE_REGISTRY,
    clear_dynamic_directives,
    list_directive_names,
    register_directive,
    register_threshold_directive,
)


@pytest.fixture
def sample_config() -> DJExecutableConfig:
    """Sample pipeline configuration for testing."""
    return {
        "dataset_path": "/tmp/test.jsonl",
        "process": [
            {"text_length_filter": {"min_len": 10, "max_len": 1000}},
            {"language_id_score_filter": {"lang": "en"}},
            {"text_length_filter": {"min_len": 10, "max_len": 1000}},  # Duplicate
            {"perplexity_filter": {"max_ppl": None}},  # No-op
        ],
    }


class TestBumpMinLenDirective:
    """Tests for BumpMinLenDirective."""

    def test_bump_min_len(self, sample_config):
        """Test bumping min_len on text_length_filter."""
        directive = BumpMinLenDirective(delta=10)
        result = directive.apply(sample_config)

        assert result.ok
        assert result.applied
        assert result.config_after is not None

        # Check that min_len was increased
        proc = result.config_after["process"]
        for step in proc:
            if "text_length_filter" in step:
                params = step["text_length_filter"]
                if "min_len" in params:
                    assert params["min_len"] == 20  # 10 + 10

    def test_no_text_length_filter(self):
        """Test behavior when no text_length_filter present."""
        config: DJExecutableConfig = {
            "process": [{"language_filter": {"lang": "en"}}]
        }
        directive = BumpMinLenDirective(delta=10)
        result = directive.apply(config)

        assert result.ok
        assert not result.applied
        assert "no text_length_filter" in result.message


class TestReorderFiltersFirstDirective:
    """Tests for ReorderFiltersFirstDirective."""

    def test_reorder_filters(self):
        """Test reordering filters before mappers."""
        config: DJExecutableConfig = {
            "process": [
                {"text_length_filter": {"min_len": 10}},  # Filter
                {"sentence_split_mapper": {}},  # Mapper
                {"language_id_score_filter": {"lang": "en"}},  # Filter
            ]
        }
        directive = ReorderFiltersFirstDirective()
        result = directive.apply(config)

        assert result.ok
        assert result.applied
        assert result.config_after is not None

        # Filters should come first
        proc = result.config_after["process"]
        assert "filter" in str(proc[0]).lower() or "language" in str(proc[0]).lower()

    def test_already_ordered(self):
        """Test when already ordered correctly."""
        config: DJExecutableConfig = {
            "process": [
                {"text_length_filter": {"min_len": 10}},
            ]
        }
        directive = ReorderFiltersFirstDirective()
        result = directive.apply(config)

        assert result.ok
        assert not result.applied


class TestRemoveRedundantOpDirective:
    """Tests for RemoveRedundantOpDirective."""

    def test_remove_duplicates(self, sample_config):
        """Test removing duplicate operators."""
        directive = RemoveRedundantOpDirective()
        result = directive.apply(sample_config)

        assert result.ok
        assert result.applied
        assert result.config_after is not None

        # Should have fewer operators (duplicate removed)
        original_count = len(sample_config["process"])
        new_count = len(result.config_after["process"])
        assert new_count < original_count

    def test_remove_noop(self):
        """Test removing no-op operators."""
        config: DJExecutableConfig = {
            "process": [
                {"text_length_filter": {"min_len": 0}},  # No-op (min_len=0)
                {"perplexity_filter": {"max_ppl": None}},  # No-op
                {"language_filter": {"lang": "en"}},  # Valid
            ]
        }
        directive = RemoveRedundantOpDirective(remove_duplicates=True, remove_noops=True)
        result = directive.apply(config)

        assert result.ok
        assert result.applied

    def test_no_redundant(self):
        """Test when no redundant operators."""
        config: DJExecutableConfig = {
            "process": [
                {"text_length_filter": {"min_len": 10}},
                {"language_filter": {"lang": "en"}},
            ]
        }
        directive = RemoveRedundantOpDirective()
        result = directive.apply(config)

        assert result.ok
        assert not result.applied


class TestAdjustThresholdDirective:
    """Tests for AdjustThresholdDirective."""

    def test_increase_min_len(self):
        """Test increasing min_len threshold."""
        config: DJExecutableConfig = {
            "process": [
                {"text_length_filter": {"min_len": 10}},
            ]
        }
        directive = AdjustThresholdDirective(
            "text_length_filter", "min_len", 5, "increase"
        )
        result = directive.apply(config)

        assert result.ok
        assert result.applied
        assert result.config_after is not None
        assert result.config_after["process"][0]["text_length_filter"]["min_len"] == 15

    def test_decrease_max_len(self):
        """Test decreasing max_len threshold."""
        config: DJExecutableConfig = {
            "process": [
                {"text_length_filter": {"max_len": 1000}},
            ]
        }
        directive = AdjustThresholdDirective(
            "text_length_filter", "max_len", 100, "decrease"
        )
        result = directive.apply(config)

        assert result.ok
        assert result.applied
        assert result.config_after["process"][0]["text_length_filter"]["max_len"] == 900


class TestTightenLoosenFiltersDirective:
    """Tests for TightenFiltersDirective and LoosenFiltersDirective."""

    def test_tighten_filters(self):
        """Test tightening all filter thresholds."""
        config: DJExecutableConfig = {
            "process": [
                {"text_length_filter": {"min_len": 10, "max_len": 1000}},
            ]
        }
        directive = TightenFiltersDirective(intensity=0.5)
        result = directive.apply(config)

        assert result.ok
        assert result.applied

        params = result.config_after["process"][0]["text_length_filter"]
        # min_len should increase, max_len should decrease
        assert params["min_len"] > 10
        assert params["max_len"] < 1000

    def test_loosen_filters(self):
        """Test loosening all filter thresholds."""
        config: DJExecutableConfig = {
            "process": [
                {"text_length_filter": {"min_len": 10, "max_len": 1000}},
            ]
        }
        directive = LoosenFiltersDirective(intensity=0.5)
        result = directive.apply(config)

        assert result.ok
        assert result.applied

        params = result.config_after["process"][0]["text_length_filter"]
        # min_len should decrease, max_len should increase
        assert params["min_len"] < 10
        assert params["max_len"] > 1000


class TestGleaningDirectives:
    """Tests for gleaning directives."""

    def test_add_gleaning(self):
        """Test adding gleaning to an LLM operator."""
        config: DJExecutableConfig = {
            "process": [
                {"llm_map": {"prompt": "Summarize the text"}},
            ]
        }
        directive = AddGleaningDirective(op_index=0, max_rounds=3)
        result = directive.apply(config)

        assert result.ok
        assert result.applied
        assert result.config_after["process"][0]["llm_map"]["max_rounds"] == 3

    def test_remove_gleaning(self):
        """Test removing gleaning from an operator."""
        config: DJExecutableConfig = {
            "process": [
                {"llm_map": {"prompt": "Summarize", "max_rounds": 3}},
            ]
        }
        directive = RemoveGleaningDirective(op_index=0)
        result = directive.apply(config)

        assert result.ok
        assert result.applied
        assert result.config_after["process"][0]["llm_map"]["max_rounds"] == 1


class TestDirectiveRegistry:
    """Tests for directive registry."""

    def setup_method(self):
        """Clear dynamic directives before each test."""
        clear_dynamic_directives()

    def test_list_directive_names(self):
        """Test listing registered directives."""
        names = list_directive_names()
        assert "reorder_filters_first" in names
        assert "remove_redundant_ops" in names
        assert "tighten_filters" in names

    def test_register_directive(self):
        """Test registering a custom directive."""
        custom = BumpMinLenDirective(50)
        register_directive(custom, "custom_bump")
        assert "custom_bump" in list_directive_names()

    def test_register_threshold_directive(self):
        """Test registering a threshold directive."""
        name = register_threshold_directive(
            "text_length_filter", "min_len", 20, "increase", "custom_threshold"
        )
        assert name == "custom_threshold"
        assert "custom_threshold" in DIRECTIVE_REGISTRY