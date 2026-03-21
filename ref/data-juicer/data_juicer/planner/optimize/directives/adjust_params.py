# -*- coding: utf-8 -*-
"""Bump numeric ``min_len`` on ``text_length_filter`` if present."""

from __future__ import annotations

from copy import deepcopy

from data_juicer.planner.contracts.recipe import DJExecutableConfig
from data_juicer.planner.optimize.directives.base import Directive, DirectiveResult


class BumpMinLenDirective(Directive):
    name = "bump_text_length_min_len"

    def __init__(self, delta: int = 10) -> None:
        self.delta = int(delta)

    def apply(self, cfg: DJExecutableConfig) -> DirectiveResult:
        before = deepcopy(cfg)
        proc = before.get("process")
        if not isinstance(proc, list):
            return DirectiveResult(
                ok=True,
                applied=False,
                directive_name=self.name,
                message="no process",
                config_before=before,
                config_after=before,
            )
        after = deepcopy(before)
        applied = False
        new_proc = []
        for step in after["process"]:
            if not isinstance(step, dict) or len(step) != 1:
                new_proc.append(step)
                continue
            name = next(iter(step.keys()))
            params = step[name]
            if name == "text_length_filter" and isinstance(params, dict):
                cur = params.get("min_len")
                if isinstance(cur, (int, float)):
                    params = dict(params)
                    params["min_len"] = int(cur) + self.delta
                    new_proc.append({name: params})
                    applied = True
                    continue
            new_proc.append(step)
        after["process"] = new_proc
        return DirectiveResult(
            ok=True,
            applied=applied,
            directive_name=self.name,
            message="bump min_len" if applied else "no text_length_filter with min_len",
            config_before=before,
            config_after=after,
        )
