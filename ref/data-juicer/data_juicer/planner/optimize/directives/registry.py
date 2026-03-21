# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Dict, List

from data_juicer.planner.optimize.directives.adjust_params import BumpMinLenDirective
from data_juicer.planner.optimize.directives.base import Directive
from data_juicer.planner.optimize.directives.reorder import ReorderFiltersFirstDirective
from data_juicer.planner.optimize.directives.swap_model import SwapApiModelDirective

DIRECTIVE_REGISTRY: Dict[str, Directive] = {
    ReorderFiltersFirstDirective().name: ReorderFiltersFirstDirective(),
    BumpMinLenDirective(10).name: BumpMinLenDirective(10),
}


def register_swap_model_directive(from_model: str, to_model: str) -> None:
    """Register a parameterized swap directive (multiple keys per pair)."""
    d = SwapApiModelDirective(from_model, to_model)
    name = f"{d.name}:{from_model}->{to_model}"
    # store under unique name for engine lookup
    DIRECTIVE_REGISTRY[name] = d


def list_directive_names() -> List[str]:
    return sorted(DIRECTIVE_REGISTRY.keys())
