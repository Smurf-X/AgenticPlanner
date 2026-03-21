# -*- coding: utf-8 -*-
from data_juicer.planner.optimize.directive_engine import DirectiveEngine, DirectiveEngineConfig
from data_juicer.planner.optimize.runner import OptimizationRunner, OptimizationRunMode
from data_juicer.planner.optimize.search.beam import BeamSearchConfig, BeamSearchOptimizer


def _sample_cfg():
    return {
        "dataset_path": "in.jsonl",
        "export_path": "out.jsonl",
        "process": [{"language_id_score_filter": {"lang": "en"}}],
    }


def test_directive_engine_reorder():
    cfg = {
        "dataset_path": "in.jsonl",
        "export_path": "out.jsonl",
        "process": [
            {"whitespace_normalization_mapper": {}},
            {"text_length_filter": {"min_len": 1, "max_len": 100}},
        ],
    }
    engine = DirectiveEngine(DirectiveEngineConfig(directives=["reorder_filters_first"], max_steps=5))
    run = engine.run(cfg)
    assert run.ok
    assert run.trace[0].applied


def test_beam_search_optimizer():
    root = _sample_cfg()
    beam = BeamSearchOptimizer(
        BeamSearchConfig(
            beam_width=2,
            max_iterations=1,
            expansion_directives=["reorder_filters_first"],
        ),
    )
    out = beam.search(root)
    assert len(out) >= 1
    assert out[0].quality >= 0.0


def test_optimization_runner_directive_only():
    runner = OptimizationRunner(
        mode=OptimizationRunMode.DIRECTIVE_ONLY,
        directive_config={"directives": ["reorder_filters_first"], "max_steps": 2},
    )
    res = runner.run(_sample_cfg())
    assert res.mode == OptimizationRunMode.DIRECTIVE_ONLY
    assert res.directive is not None
    assert res.candidates is None
