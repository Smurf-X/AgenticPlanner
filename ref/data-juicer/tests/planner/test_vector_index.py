# -*- coding: utf-8 -*-
"""Tests for vector index and retrieval."""

from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import List

import numpy as np
import pytest

# Skip all tests if sentence-transformers not installed
pytest.importorskip("sentence_transformers")

from data_juicer.planner.generate.embedding import LocalEmbeddingBackend
from data_juicer.planner.generate.vector_index import (
    OperatorVectorIndex,
    _make_operator_document,
    _split_operator_name,
)


class MockOPRecord(SimpleNamespace):
    """Mock OPRecord for testing."""

    pass


def make_mock_records() -> List[MockOPRecord]:
    return [
        MockOPRecord(
            name="video_extract_frames_mapper",
            tags=["video", "cpu"],
            desc="Extract frames from video with uniform or keyframe sampling",
        ),
        MockOPRecord(
            name="text_length_filter",
            tags=["text", "cpu"],
            desc="Filter text by minimum and maximum length",
        ),
        MockOPRecord(
            name="document_minhash_deduplicator",
            tags=["text", "cpu"],
            desc="Deduplicate documents using MinHash algorithm",
        ),
        MockOPRecord(
            name="image_captioning_mapper",
            tags=["image", "gpu"],
            desc="Generate captions for images using vision-language model",
        ),
    ]


class TestOperatorDocument:
    def test_split_operator_name(self):
        parts = _split_operator_name("video_extract_frames_mapper")
        assert "video" in parts
        assert "extract" in parts
        assert "frames" in parts

    def test_make_document(self):
        rec = MockOPRecord(
            name="video_extract_frames_mapper",
            tags=["video", "cpu"],
            desc="Extract frames from video",
        )
        doc = _make_operator_document(rec)
        assert "video_extract_frames_mapper" in doc
        assert "video" in doc
        assert "Extract frames" in doc


class TestLocalEmbeddingBackend:
    def test_dimension(self):
        embedder = LocalEmbeddingBackend(model_name="BAAI/bge-small-zh-v1.5")
        assert embedder.dimension > 0
        assert embedder.model_name == "BAAI/bge-small-zh-v1.5"

    def test_embed_single_text(self):
        embedder = LocalEmbeddingBackend(model_name="BAAI/bge-small-zh-v1.5")
        vec = embedder.embed_query("这是一个测试句子")
        assert isinstance(vec, np.ndarray)
        assert vec.shape == (embedder.dimension,)
        # Check normalization
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 1e-5

    def test_embed_batch(self):
        embedder = LocalEmbeddingBackend(model_name="BAAI/bge-small-zh-v1.5")
        texts = ["视频抽帧", "文本去重", "图像标注"]
        vectors = embedder.embed_texts(texts)
        assert isinstance(vectors, np.ndarray)
        assert vectors.shape == (3, embedder.dimension)


class TestOperatorVectorIndex:
    def test_build_and_search(self):
        records = make_mock_records()
        embedder = LocalEmbeddingBackend(model_name="BAAI/bge-small-zh-v1.5")

        index = OperatorVectorIndex(embedder=embedder)
        index.build(records)

        assert index.is_built
        assert len(index._records) == 4

        results = index.search("对视频做均匀抽帧", top_k=2)
        assert len(results) == 2
        # video_extract_frames_mapper should be in top results
        names = [rec.name for rec, score in results]
        assert "video_extract_frames_mapper" in names

    def test_persistence(self):
        records = make_mock_records()
        embedder = LocalEmbeddingBackend(model_name="BAAI/bge-small-zh-v1.5")

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "test_index.npz"

            # Build and save
            index1 = OperatorVectorIndex(embedder=embedder)
            index1.build(records)
            index1.save(str(cache_path))
            assert cache_path.exists()

            # Load in new instance
            index2 = OperatorVectorIndex(embedder=embedder)
            loaded = index2.load(records, path=str(cache_path))
            assert loaded
            assert index2.is_built

            # Search should work
            results = index2.search("文本长度过滤", top_k=2)
            assert len(results) == 2

    def test_cache_invalidation_on_model_change(self):
        records = make_mock_records()
        embedder1 = LocalEmbeddingBackend(model_name="BAAI/bge-small-zh-v1.5")
        embedder2 = LocalEmbeddingBackend(model_name="BAAI/bge-base-zh-v1.5")

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "test_index.npz"

            index1 = OperatorVectorIndex(embedder=embedder1)
            index1.build(records)
            index1.save(str(cache_path))

            # Different model should not load cache
            index2 = OperatorVectorIndex(embedder=embedder2)
            loaded = index2.load(records, path=str(cache_path))
            assert not loaded  # Model mismatch


if __name__ == "__main__":
    pytest.main([__file__, "-v"])