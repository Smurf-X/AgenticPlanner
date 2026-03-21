# -*- coding: utf-8 -*-
from data_juicer.planner.generate.op_schema import (
    get_init_param_allowlist,
    sanitize_params,
    validate_params_bind,
)


def test_text_length_filter_allowlist():
    names = get_init_param_allowlist("text_length_filter")
    assert "min_len" in names
    assert "max_len" in names


def test_sanitize_drops_unknown():
    clean = sanitize_params(
        "text_length_filter",
        {"min_len": 1, "max_len": 10, "api_model": "x", "num_frames": 3},
    )
    assert clean == {"min_len": 1, "max_len": 10}
    assert validate_params_bind("text_length_filter", clean)[0]


def test_video_extract_frames_mapper_names():
    """LLM hallucinations like num_frames must not survive sanitize."""
    names = get_init_param_allowlist("video_extract_frames_mapper")
    assert "frame_num" in names
    assert "frame_sampling_method" in names
    assert "num_frames" not in names

    raw = {"num_frames": 8, "sample_method": "uniform", "frame_num": 8, "frame_sampling_method": "uniform"}
    clean = sanitize_params("video_extract_frames_mapper", raw)
    assert "num_frames" not in clean
    assert "sample_method" not in clean
    assert clean.get("frame_num") == 8
