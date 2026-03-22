# -*- coding: utf-8 -*-
"""Embedding backends for vector retrieval (local sentence-transformers or API)."""

from __future__ import annotations

from data_juicer.planner.generate.embedding.backend import EmbeddingBackend
from data_juicer.planner.generate.embedding.local import LocalEmbeddingBackend
from data_juicer.planner.generate.embedding.api import APIEmbeddingBackend

__all__ = [
    "EmbeddingBackend",
    "LocalEmbeddingBackend",
    "APIEmbeddingBackend",
]