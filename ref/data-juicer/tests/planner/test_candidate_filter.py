# -*- coding: utf-8 -*-
from types import SimpleNamespace

from data_juicer.planner.generate.candidate_filter import (
    detect_modalities,
    filter_ops_by_modality,
)


def test_detect_modalities_video_cn():
    m = detect_modalities("对视频做均匀抽帧", "")
    assert "video" in m


def test_detect_modalities_empty_when_no_keywords():
    # Avoid substrings like "word" inside "world" matching text keywords.
    m = detect_modalities("aaa bbb ccc ddd eee", "")
    assert m == set()


def test_filter_ops_by_modality_keeps_general_and_multimodal():
    v = SimpleNamespace(name="v1", tags=["video", "cpu"])
    t = SimpleNamespace(name="t1", tags=["text", "cpu"])
    g = SimpleNamespace(name="g1", tags=["cpu", "api"])
    mm = SimpleNamespace(name="m1", tags=["multimodal", "gpu"])
    out = filter_ops_by_modality([v, t, g, mm], {"video"})
    names = {x.name for x in out}
    assert "v1" in names
    assert "m1" in names
    assert "g1" in names
    assert "t1" not in names


def test_filter_ops_by_modality_no_modalities_returns_all():
    a = SimpleNamespace(name="a", tags=["text"])
    b = SimpleNamespace(name="b", tags=["video"])
    out = filter_ops_by_modality([a, b], set())
    assert len(out) == 2
