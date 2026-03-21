# -*- coding: utf-8 -*-
import pytest

from data_juicer.planner.contracts.recipe import validate_executable_config
from data_juicer.planner.generate.generator import (
    NLRecipeGenerator,
    assemble_executable_config,
    generate_recipe_from_llm_json_text,
)


def test_assemble_executable_config():
    from data_juicer.planner.contracts.plan_bridge import OperatorStep

    cfg = assemble_executable_config(
        dataset_path="in.jsonl",
        export_path="out.jsonl",
        operators=[OperatorStep("text_length_filter", {"min_len": 1, "max_len": 99})],
        extra_config={"np": 2},
    )
    assert cfg["dataset_path"] == "in.jsonl"
    assert cfg["process"] == [{"text_length_filter": {"min_len": 1, "max_len": 99}}]
    assert cfg["np"] == 2
    assert validate_executable_config(cfg) == []


def test_generate_recipe_from_llm_json_text():
    text = '{"operators": [{"name": "text_length_filter", "params": {"min_len": 10, "max_len": 200}}]}'
    cfg = generate_recipe_from_llm_json_text(
        text=text,
        dataset_path="in.jsonl",
        export_path="out.jsonl",
    )
    assert validate_executable_config(cfg) == []


def test_generate_recipe_strips_unknown_keys():
    text = '{"operators": [{"name": "text_length_filter", "params": {"min_len": 10, "max_len": 200, "fake_key": 1}}]}'
    cfg = generate_recipe_from_llm_json_text(
        text=text,
        dataset_path="in.jsonl",
        export_path="out.jsonl",
        strict_params=True,
    )
    step = cfg["process"][0]["text_length_filter"]
    assert "fake_key" not in step


class _CountingLLM:
    """Return preprogrammed dicts in order (select, then one fill per op)."""

    def __init__(self, responses):
        self._responses = list(responses)
        self._i = 0

    def complete_json(self, system: str, user: str):
        if self._i >= len(self._responses):
            raise RuntimeError("unexpected extra LLM call")
        r = self._responses[self._i]
        self._i += 1
        return r


def test_nl_recipe_generator_per_operator_mock():
    gen = NLRecipeGenerator(
        _CountingLLM(
            [
                {"operator_names": ["text_length_filter"]},
                {"params": {"min_len": 10, "max_len": 100, "bogus": 1}},
            ]
        )
    )
    cfg = gen.generate(
        user_intent="filter by length",
        dataset_path="in.jsonl",
        export_path="out.jsonl",
        fill_mode="per_operator",
    )
    assert validate_executable_config(cfg) == []
    assert "bogus" not in cfg["process"][0]["text_length_filter"]
