# -*- coding: utf-8 -*-
"""Replace ``api_model`` in operator params (LLM mappers / filters)."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, MutableMapping

from data_juicer.planner.contracts.recipe import DJExecutableConfig
from data_juicer.planner.optimize.directives.base import Directive, DirectiveResult


def _replace_keys(obj: Any, old: str, new: str, keys: tuple[str, ...] = ("api_model", "model")) -> bool:
    changed = False
    if isinstance(obj, MutableMapping):
        for k in keys:
            if k in obj and obj[k] == old:
                obj[k] = new
                changed = True
        for v in obj.values():
            if _replace_keys(v, old, new, keys):
                changed = True
    elif isinstance(obj, list):
        for item in obj:
            if _replace_keys(item, old, new, keys):
                changed = True
    return changed


class SwapApiModelDirective(Directive):
    name = "swap_api_model"

    def __init__(self, from_model: str, to_model: str) -> None:
        self.from_model = str(from_model)
        self.to_model = str(to_model)

    def apply(self, cfg: DJExecutableConfig) -> DirectiveResult:
        before = deepcopy(cfg)
        after = deepcopy(cfg)
        proc = after.get("process")
        if not isinstance(proc, list):
            return DirectiveResult(
                ok=True,
                applied=False,
                directive_name=self.name,
                message="no process",
                config_before=before,
                config_after=after,
            )
        changed = _replace_keys(proc, self.from_model, self.to_model)
        return DirectiveResult(
            ok=True,
            applied=changed,
            directive_name=self.name,
            message=f"{self.from_model} -> {self.to_model}" if changed else "no matching model field",
            config_before=before,
            config_after=after,
        )
