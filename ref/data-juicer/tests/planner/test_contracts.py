# -*- coding: utf-8 -*-
import pytest

from data_juicer.planner.contracts.cost import CostBreakdown, compute_token_cost
from data_juicer.planner.contracts.plan_bridge import OperatorStep, plan_operators_to_process, process_to_plan_operators
from data_juicer.planner.contracts.recipe import validate_executable_config


def test_plan_operators_roundtrip():
    ops = [OperatorStep("text_length_filter", {"min_len": 1, "max_len": 100})]
    proc = plan_operators_to_process(ops)
    back = process_to_plan_operators(proc)
    assert len(back) == 1
    assert back[0].name == "text_length_filter"
    assert back[0].params["min_len"] == 1


def test_validate_executable_config_ok():
    cfg = {
        "dataset_path": "a.jsonl",
        "export_path": "b.jsonl",
        "process": [{"text_length_filter": {"min_len": 1, "max_len": 10}}],
    }
    assert validate_executable_config(cfg) == []


def test_validate_executable_config_bad_op():
    cfg = {
        "dataset_path": "a.jsonl",
        "export_path": "b.jsonl",
        "process": [{"not_a_real_op_xxx": {}}],
    }
    err = validate_executable_config(cfg)
    assert err


def test_compute_token_cost():
    usage = {"gpt-4o-mini": {"prompt": 1_000_000, "completion": 0}}
    price = {"gpt-4o-mini": 1.0}
    assert compute_token_cost(usage, price) == pytest.approx(1.0)


def test_cost_breakdown_repr():
    c = CostBreakdown(llm_token_cost=0.5, wall_time_sec=1.2, prompt_tokens=10, completion_tokens=2)
    assert "0.5" in repr(c) and "1.2" in repr(c)
