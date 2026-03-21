# -*- coding: utf-8 -*-
"""Stage-1: apply a list of directives in order with an audit trail."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from data_juicer.planner.contracts.recipe import DJExecutableConfig, validate_executable_config
from data_juicer.planner.optimize.directives.base import DirectiveResult
from data_juicer.planner.optimize.directives.registry import DIRECTIVE_REGISTRY


class DirectiveEngineConfig(BaseModel):
    """YAML-friendly config for :class:`DirectiveEngine`."""

    directives: List[str] = Field(
        default_factory=list,
        description="Directive keys to apply in order (see DIRECTIVE_REGISTRY).",
    )
    max_steps: int = Field(default=50, ge=1, le=500)
    """Max applied directives (same name may repeat if listed)."""

    model_config = {"extra": "allow"}


@dataclass
class DirectiveEngineRun:
    """Result of a directive-only pass."""

    ok: bool
    config: DJExecutableConfig
    trace: List[DirectiveResult] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class DirectiveEngine:
    """Apply configured directives sequentially."""

    def __init__(self, config: DirectiveEngineConfig) -> None:
        self._config = config

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DirectiveEngine":
        return cls(DirectiveEngineConfig.model_validate(data))

    def run(self, cfg: DJExecutableConfig) -> DirectiveEngineRun:
        trace: List[DirectiveResult] = []
        errors: List[str] = []
        current = cfg
        steps = 0
        for name in self._config.directives:
            if steps >= self._config.max_steps:
                break
            directive = DIRECTIVE_REGISTRY.get(name)
            if directive is None:
                errors.append(f"unknown directive: {name}")
                continue
            res = directive.apply(current)
            trace.append(res)
            if not res.ok:
                errors.append(f"{name}: directive failed")
                break
            if res.config_after is not None:
                current = res.config_after
            steps += 1

        val_err = validate_executable_config(current)
        if val_err:
            errors.extend(val_err)
        return DirectiveEngineRun(
            ok=not errors,
            config=current,
            trace=trace,
            errors=errors,
        )
