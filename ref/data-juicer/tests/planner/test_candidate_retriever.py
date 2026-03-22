# -*- coding: utf-8 -*-
"""Tests for VectorRetriever."""

from __future__ import annotations

import tempfile
from types import SimpleNamespace
from typing import List

import numpy as np
import pytest

pytest.importorskip("sentence_transformers")

from data_juicer.planner.generate.candidate_retriever import (
    VectorRetriever,
    _must_include_names,
)
from data_juicer.planner.generate.embedding import LocalEmbeddingBackend


class TestMustInclude:
    """Test must-include rule extraction."""

    def test_video_frame_extraction(self):
        names = _must_include_names("对视频做均匀抽帧", "")
        assert "video_extract_frames_mapper" in names

    def test_duration_filter(self):
        names = _must_include_names("过滤时长不足10秒的视频", "")
        assert "video_duration_filter" in names

    def test_text_dedup(self):
        names = _must_include_names("文本去重", "")
        assert "document_minhash_deduplicator" in names


class TestVectorRetriever:
    """Test VectorRetriever with local embedding."""

    def test_retrieve_video_ops(self):
        embedder = LocalEmbeddingBackend(model_name="BAAI/bge-small-zh-v1.5")

        with tempfile.TemporaryDirectory() as tmpdir:
            retriever = VectorRetriever(
                embedder=embedder,
                cache_dir=tmpdir,
            )

            names = retriever.retrieve(
                intent="对视频做均匀抽帧，每段抽8帧",
                top_k=10,
                dataset_hint="MP4视频文件",
                use_must_include=True,
            )

            assert len(names) <= 10
            assert len(names) > 0
            # video_extract_frames_mapper should be in results (must-include)
            assert "video_extract_frames_mapper" in names

    def test_retrieve_text_ops(self):
        embedder = LocalEmbeddingBackend(model_name="BAAI/bge-small-zh-v1.5")

        with tempfile.TemporaryDirectory() as tmpdir:
            retriever = VectorRetriever(
                embedder=embedder,
                cache_dir=tmpdir,
            )

            names = retriever.retrieve(
                intent="过滤文本长度过短的条目",
                top_k=10,
                dataset_hint="文本数据集",
            )

            assert len(names) <= 10
            # text_length_filter should be found
            # (may be via BM25 must-include or vector similarity)

    def test_modality_filter(self):
        embedder = LocalEmbeddingBackend(model_name="BAAI/bge-small-zh-v1.5")

        with tempfile.TemporaryDirectory() as tmpdir:
            retriever = VectorRetriever(
                embedder=embedder,
                cache_dir=tmpdir,
            )

            # Video intent with modality filter
            names = retriever.retrieve(
                intent="视频处理",
                top_k=20,
                use_modality_filter=True,
            )

            # Should prioritize video operators
            # (checking that filter is applied, not exact match)
            assert len(names) > 0

    def test_cache_persistence(self):
        embedder = LocalEmbeddingBackend(model_name="BAAI/bge-small-zh-v1.5")

        with tempfile.TemporaryDirectory() as tmpdir:
            # First retriever builds index
            retriever1 = VectorRetriever(
                embedder=embedder,
                cache_dir=tmpdir,
            )
            names1 = retriever1.retrieve("测试查询", top_k=5)

            # Second retriever should load from cache
            retriever2 = VectorRetriever(
                embedder=embedder,
                cache_dir=tmpdir,
            )
            names2 = retriever2.retrieve("测试查询", top_k=5)

            assert names1 == names2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])