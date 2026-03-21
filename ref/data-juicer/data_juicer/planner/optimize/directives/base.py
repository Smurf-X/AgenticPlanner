# -*- coding: utf-8 -*-
from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from data_juicer.planner.contracts.recipe import DJExecutableConfig


@dataclass
class DirectiveResult:
    """One transformation attempt."""

    ok: bool
    applied: bool
    directive_name: str
    message: str = ""
    config_before: Optional[Dict[str, Any]] = None
    config_after: Optional[Dict[str, Any]] = None
    details: Dict[str, Any] = field(default_factory=dict)


class Directive(ABC):
    """Single-step rewrite of a DJ executable config."""

    name: str = "base"

    @abstractmethod
    def apply(self, cfg: DJExecutableConfig) -> DirectiveResult:
        """Return a copy of ``cfg`` with the transformation applied, or ``applied=False`` if no-op."""

    def _clone(self, cfg: DJExecutableConfig) -> DJExecutableConfig:
        return deepcopy(cfg)
