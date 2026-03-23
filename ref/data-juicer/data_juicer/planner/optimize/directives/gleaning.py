# -*- coding: utf-8 -*-
"""Add gleaning (multi-turn verification) to LLM-based operators."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional

from data_juicer.planner.contracts.recipe import DJExecutableConfig
from data_juicer.planner.optimize.directives.base import Directive, DirectiveResult


# Operators that support gleaning
_GLEANING_SUPPORTED_OPS = {
    "map_wrapper",
    "llm_map",
    "filter_wrapper",
    "llm_filter",
}


class AddGleaningDirective(Directive):
    """
    Add gleaning parameters to an LLM-based operator.

    Gleaning enables multi-turn verification where the LLM can
    refine its output through additional passes.
    """

    name = "add_gleaning"

    def __init__(
        self,
        op_index: int,
        max_rounds: int = 2,
        gleaning_prompt: Optional[str] = None,
    ) -> None:
        """
        Args:
            op_index: Index of the operator in the process list
            max_rounds: Maximum number of gleaning rounds (default: 2)
            gleaning_prompt: Custom prompt for gleaning verification
        """
        self.op_index = op_index
        self.max_rounds = max(1, min(max_rounds, 5))
        self.gleaning_prompt = gleaning_prompt

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

        if self.op_index < 0 or self.op_index >= len(proc):
            return DirectiveResult(
                ok=False,
                applied=False,
                directive_name=self.name,
                message=f"invalid op_index {self.op_index}",
                config_before=before,
                config_after=before,
            )

        after = deepcopy(before)
        step = after["process"][self.op_index]
        if not isinstance(step, dict) or len(step) != 1:
            return DirectiveResult(
                ok=True,
                applied=False,
                directive_name=self.name,
                message="invalid step format",
                config_before=before,
                config_after=before,
            )

        op_name = next(iter(step.keys()))
        params = step.get(op_name, {})
        if not isinstance(params, dict):
            params = {}

        # Check if gleaning is supported for this operator
        if op_name not in _GLEANING_SUPPORTED_OPS:
            return DirectiveResult(
                ok=True,
                applied=False,
                directive_name=self.name,
                message=f"gleaning not supported for {op_name}",
                config_before=before,
                config_after=before,
            )

        # Check if gleaning already enabled
        if params.get("max_rounds", 1) > 1:
            return DirectiveResult(
                ok=True,
                applied=False,
                directive_name=self.name,
                message="gleaning already enabled",
                config_before=before,
                config_after=before,
            )

        new_params = dict(params)
        new_params["max_rounds"] = self.max_rounds

        if self.gleaning_prompt:
            new_params["gleaning_prompt"] = self.gleaning_prompt

        after["process"][self.op_index] = {op_name: new_params}

        return DirectiveResult(
            ok=True,
            applied=True,
            directive_name=self.name,
            message=f"added gleaning ({self.max_rounds} rounds) to {op_name}",
            config_before=before,
            config_after=after,
            details={
                "op_index": self.op_index,
                "op_name": op_name,
                "max_rounds": self.max_rounds,
            },
        )


class RemoveGleaningDirective(Directive):
    """Remove gleaning from an LLM-based operator (for cost optimization)."""

    name = "remove_gleaning"

    def __init__(self, op_index: int) -> None:
        self.op_index = op_index

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

        if self.op_index < 0 or self.op_index >= len(proc):
            return DirectiveResult(
                ok=False,
                applied=False,
                directive_name=self.name,
                message=f"invalid op_index {self.op_index}",
                config_before=before,
                config_after=before,
            )

        after = deepcopy(before)
        step = after["process"][self.op_index]
        if not isinstance(step, dict) or len(step) != 1:
            return DirectiveResult(
                ok=True,
                applied=False,
                directive_name=self.name,
                message="invalid step format",
                config_before=before,
                config_after=before,
            )

        op_name = next(iter(step.keys()))
        params = step.get(op_name, {})
        if not isinstance(params, dict):
            params = {}

        if params.get("max_rounds", 1) <= 1:
            return DirectiveResult(
                ok=True,
                applied=False,
                directive_name=self.name,
                message="gleaning not enabled",
                config_before=before,
                config_after=before,
            )

        new_params = dict(params)
        new_params["max_rounds"] = 1
        new_params.pop("gleaning_prompt", None)

        after["process"][self.op_index] = {op_name: new_params}

        return DirectiveResult(
            ok=True,
            applied=True,
            directive_name=self.name,
            message=f"removed gleaning from {op_name}",
            config_before=before,
            config_after=after,
            details={"op_index": self.op_index, "op_name": op_name},
        )