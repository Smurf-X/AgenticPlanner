# -*- coding: utf-8 -*-
"""Generic threshold adjustment directive for filter-type operators."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from data_juicer.planner.contracts.recipe import DJExecutableConfig
from data_juicer.planner.optimize.directives.base import Directive, DirectiveResult


# Known threshold parameters for common filter operators
_THRESHOLD_PARAMS: Dict[str, Dict[str, Tuple[str, float]]] = {
    "text_length_filter": {
        "min_len": ("increase", 10),
        "max_len": ("decrease", 100),
    },
    "words_num_filter": {
        "min_num": ("increase", 5),
        "max_num": ("decrease", 50),
    },
    "character_num_filter": {
        "min_num": ("increase", 50),
        "max_num": ("decrease", 500),
    },
    "perplexity_filter": {
        "max_ppl": ("decrease", 50),
    },
    "token_num_filter": {
        "min_num": ("increase", 10),
        "max_num": ("decrease", 100),
    },
    "special_characters_filter": {
        "min_ratio": ("increase", 0.05),
        "max_ratio": ("decrease", 0.05),
    },
    "alnum_ratio_filter": {
        "min_ratio": ("increase", 0.05),
        "max_ratio": ("decrease", 0.05),
    },
}


class AdjustThresholdDirective(Directive):
    """
    Adjust a numeric threshold parameter on a specific operator.

    This is a parameterized directive that can be registered multiple times
    with different configurations.
    """

    name = "adjust_threshold"

    def __init__(
        self,
        op_name: str,
        param_name: str,
        delta: float,
        direction: Optional[str] = None,
    ) -> None:
        """
        Initialize the threshold adjustment directive.

        Args:
            op_name: Name of the operator (e.g., "text_length_filter")
            param_name: Name of the parameter to adjust (e.g., "min_len")
            delta: Amount to adjust (positive value)
            direction: "increase" or "decrease"; if None, uses known defaults
        """
        self.op_name = op_name
        self.param_name = param_name
        self.delta = abs(delta)
        self._direction = direction

    def _get_direction(self) -> str:
        """Determine adjustment direction."""
        if self._direction:
            return self._direction
        # Look up known threshold info
        op_info = _THRESHOLD_PARAMS.get(self.op_name, {})
        if self.param_name in op_info:
            return op_info[self.param_name][0]
        # Default: increase for "min_*", decrease for "max_*"
        if self.param_name.startswith("min_"):
            return "increase"
        return "decrease"

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
        direction = self._get_direction()
        applied = False
        adjustments: List[Dict[str, Any]] = []

        new_proc = []
        for step in after.get("process", []):
            if not isinstance(step, dict) or len(step) != 1:
                new_proc.append(step)
                continue

            name = next(iter(step.keys()))
            params = step[name]

            if name == self.op_name and isinstance(params, dict):
                if self.param_name in params:
                    old_val = params[self.param_name]
                    if isinstance(old_val, (int, float)):
                        new_params = dict(params)
                        if direction == "increase":
                            new_params[self.param_name] = old_val + self.delta
                        else:
                            new_params[self.param_name] = max(0, old_val - self.delta)
                        new_proc.append({name: new_params})
                        applied = True
                        adjustments.append({
                            "op": name,
                            "param": self.param_name,
                            "old": old_val,
                            "new": new_params[self.param_name],
                        })
                        continue

            new_proc.append(step)

        after["process"] = new_proc

        return DirectiveResult(
            ok=True,
            applied=applied,
            directive_name=self.name,
            message=f"adjusted {self.op_name}.{self.param_name}" if applied else "no matching operator/param",
            config_before=before,
            config_after=after,
            details={"adjustments": adjustments, "direction": direction, "delta": self.delta},
        )


class TightenFiltersDirective(Directive):
    """
    Tighten all filter thresholds to be more selective (reduce output size).

    This applies known threshold adjustments to make filters more restrictive.
    """

    name = "tighten_filters"

    def __init__(self, intensity: float = 0.1) -> None:
        """
        Args:
            intensity: Multiplier for adjustment amounts (0.0-1.0)
        """
        self.intensity = max(0.0, min(1.0, intensity))

    def apply(self, cfg: DJExecutableConfig) -> DirectiveResult:
        before = deepcopy(cfg)
        proc = before.get("process")
        if not isinstance(proc, list) or not proc:
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
        adjustments: List[Dict[str, Any]] = []

        new_proc = []
        for step in after.get("process", []):
            if not isinstance(step, dict) or len(step) != 1:
                new_proc.append(step)
                continue

            op_name = next(iter(step.keys()))
            params = step.get(op_name, {})
            if not isinstance(params, dict):
                new_proc.append(step)
                continue

            op_info = _THRESHOLD_PARAMS.get(op_name)
            if not op_info:
                new_proc.append(step)
                continue

            new_params = dict(params)
            changed = False
            for param_name, (direction, base_delta) in op_info.items():
                if param_name in new_params:
                    old_val = new_params[param_name]
                    if isinstance(old_val, (int, float)):
                        delta = base_delta * self.intensity
                        if direction == "increase":
                            new_params[param_name] = old_val + delta
                        else:
                            new_params[param_name] = max(0, old_val - delta)
                        adjustments.append({
                            "op": op_name,
                            "param": param_name,
                            "old": old_val,
                            "new": new_params[param_name],
                        })
                        changed = True

            if changed:
                new_proc.append({op_name: new_params})
                applied = True
            else:
                new_proc.append(step)

        after["process"] = new_proc

        return DirectiveResult(
            ok=True,
            applied=applied,
            directive_name=self.name,
            message=f"tightened {len(adjustments)} threshold(s)" if applied else "no adjustable thresholds",
            config_before=before,
            config_after=after,
            details={"adjustments": adjustments, "intensity": self.intensity},
        )


class LoosenFiltersDirective(Directive):
    """
    Loosen all filter thresholds to be less selective (increase output size).

    Opposite of TightenFiltersDirective.
    """

    name = "loosen_filters"

    def __init__(self, intensity: float = 0.1) -> None:
        self.intensity = max(0.0, min(1.0, intensity))

    def apply(self, cfg: DJExecutableConfig) -> DirectiveResult:
        before = deepcopy(cfg)
        proc = before.get("process")
        if not isinstance(proc, list) or not proc:
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
        adjustments: List[Dict[str, Any]] = []

        new_proc = []
        for step in after.get("process", []):
            if not isinstance(step, dict) or len(step) != 1:
                new_proc.append(step)
                continue

            op_name = next(iter(step.keys()))
            params = step.get(op_name, {})
            if not isinstance(params, dict):
                new_proc.append(step)
                continue

            op_info = _THRESHOLD_PARAMS.get(op_name)
            if not op_info:
                new_proc.append(step)
                continue

            new_params = dict(params)
            changed = False
            for param_name, (direction, base_delta) in op_info.items():
                if param_name in new_params:
                    old_val = new_params[param_name]
                    if isinstance(old_val, (int, float)):
                        delta = base_delta * self.intensity
                        # Reverse direction to loosen
                        if direction == "increase":
                            new_params[param_name] = max(0, old_val - delta)
                        else:
                            new_params[param_name] = old_val + delta
                        adjustments.append({
                            "op": op_name,
                            "param": param_name,
                            "old": old_val,
                            "new": new_params[param_name],
                        })
                        changed = True

            if changed:
                new_proc.append({op_name: new_params})
                applied = True
            else:
                new_proc.append(step)

        after["process"] = new_proc

        return DirectiveResult(
            ok=True,
            applied=applied,
            directive_name=self.name,
            message=f"loosened {len(adjustments)} threshold(s)" if applied else "no adjustable thresholds",
            config_before=before,
            config_after=after,
            details={"adjustments": adjustments, "intensity": self.intensity},
        )