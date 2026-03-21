# -*- coding: utf-8 -*-
from types import SimpleNamespace

from data_juicer.planner.generate.candidate_ranker import rank_candidates, tokenize_query


def test_tokenize_query_mixed():
    toks = tokenize_query("filter by LENGTH 视频")
    assert "length" in toks or "LENGTH".lower() in toks
    assert len(toks) >= 1


def test_rank_candidates_prefers_name_match():
    a = SimpleNamespace(
        name="zz_unrelated_mapper",
        tags=["cpu", "text"],
        desc="unrelated",
    )
    b = SimpleNamespace(
        name="text_length_filter",
        tags=["cpu", "text"],
        desc="filter by min max length",
    )
    names = rank_candidates("filter text by length constraints", [a, b], top_k=2)
    assert names[0] == "text_length_filter"


def test_rank_candidates_must_include_video_ops():
    pool = [
        SimpleNamespace(name="alphanumeric_filter", tags=["cpu", "text"], desc="ratio"),
        SimpleNamespace(
            name="video_duration_filter",
            tags=["cpu", "video"],
            desc="duration",
        ),
        SimpleNamespace(
            name="video_extract_frames_mapper",
            tags=["cpu", "video"],
            desc="frames",
        ),
    ]
    names = rank_candidates("按均匀采样抽8帧，并过滤时长", pool, top_k=5)
    assert "video_extract_frames_mapper" in names
    assert "video_duration_filter" in names
