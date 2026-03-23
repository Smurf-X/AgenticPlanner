# -*- coding: utf-8 -*-
"""Registry for optimization directives."""

from __future__ import annotations

from typing import Dict, List, Optional, Type

from data_juicer.planner.optimize.directives.adjust_params import BumpMinLenDirective
from data_juicer.planner.optimize.directives.adjust_threshold import (
    AdjustThresholdDirective,
    LoosenFiltersDirective,
    TightenFiltersDirective,
)
from data_juicer.planner.optimize.directives.base import Directive
from data_juicer.planner.optimize.directives.gleaning import (
    AddGleaningDirective,
    RemoveGleaningDirective,
)
from data_juicer.planner.optimize.directives.remove_redundant import (
    RemoveRedundantOpDirective,
)
from data_juicer.planner.optimize.directives.reorder import ReorderFiltersFirstDirective
from data_juicer.planner.optimize.directives.rewrite_prompt import (
    AddFewShotExamplesDirective,
    RewritePromptDirective,
)
from data_juicer.planner.optimize.directives.swap_model import SwapApiModelDirective

# Core registry with singleton instances
DIRECTIVE_REGISTRY: Dict[str, Directive] = {}


def _register_instance(d: Directive) -> None:
    """Register a directive instance under its name."""
    DIRECTIVE_REGISTRY[d.name] = d


# Register default directives
_register_instance(ReorderFiltersFirstDirective())
_register_instance(RemoveRedundantOpDirective())
_register_instance(TightenFiltersDirective())
_register_instance(LoosenFiltersDirective())
_register_instance(BumpMinLenDirective(10))


def register_directive(directive: Directive, name: Optional[str] = None) -> str:
    """
    Register a custom directive instance.

    Args:
        directive: The directive instance to register
        name: Optional custom name (default: use directive.name)

    Returns:
        The name under which the directive was registered
    """
    key = name or directive.name
    DIRECTIVE_REGISTRY[key] = directive
    return key


def register_swap_model_directive(from_model: str, to_model: str) -> str:
    """
    Register a parameterized model swap directive.

    Args:
        from_model: Source model name
        to_model: Target model name

    Returns:
        The unique directive name
    """
    d = SwapApiModelDirective(from_model, to_model)
    name = f"swap_model:{from_model}->{to_model}"
    DIRECTIVE_REGISTRY[name] = d
    return name


def register_threshold_directive(
    op_name: str,
    param_name: str,
    delta: float,
    direction: Optional[str] = None,
    name: Optional[str] = None,
) -> str:
    """
    Register a parameterized threshold adjustment directive.

    Args:
        op_name: Operator name
        param_name: Parameter to adjust
        delta: Adjustment amount
        direction: "increase" or "decrease"
        name: Custom registry name

    Returns:
        The directive name
    """
    d = AdjustThresholdDirective(op_name, param_name, delta, direction)
    key = name or f"adjust_{op_name}_{param_name}"
    DIRECTIVE_REGISTRY[key] = d
    return key


def register_prompt_rewrite_directive(
    op_index: int,
    new_prompt: Optional[str] = None,
    prompt_suffix: Optional[str] = None,
    clarify_instruction: Optional[str] = None,
    name: Optional[str] = None,
) -> str:
    """
    Register a prompt rewrite directive for a specific operator.

    Args:
        op_index: Index of the operator in process list
        new_prompt: Replace prompt entirely
        prompt_suffix: Append to prompt
        clarify_instruction: Prepend clarification
        name: Custom registry name

    Returns:
        The directive name
    """
    d = RewritePromptDirective(op_index, new_prompt, prompt_suffix, clarify_instruction)
    key = name or f"rewrite_prompt_op{op_index}"
    DIRECTIVE_REGISTRY[key] = d
    return key


def register_gleaning_directive(
    op_index: int,
    max_rounds: int = 2,
    gleaning_prompt: Optional[str] = None,
    name: Optional[str] = None,
) -> str:
    """
    Register a gleaning directive for a specific operator.

    Args:
        op_index: Index of the operator
        max_rounds: Maximum gleaning rounds
        gleaning_prompt: Custom gleaning prompt
        name: Custom registry name

    Returns:
        The directive name
    """
    d = AddGleaningDirective(op_index, max_rounds, gleaning_prompt)
    key = name or f"add_gleaning_op{op_index}"
    DIRECTIVE_REGISTRY[key] = d
    return key


def register_few_shot_directive(
    op_index: int,
    examples: List[Dict[str, str]],
    name: Optional[str] = None,
) -> str:
    """
    Register a few-shot examples directive.

    Args:
        op_index: Index of the operator
        examples: List of {input, output} example dicts
        name: Custom registry name

    Returns:
        The directive name
    """
    d = AddFewShotExamplesDirective(op_index, examples)
    key = name or f"add_fewshot_op{op_index}"
    DIRECTIVE_REGISTRY[key] = d
    return key


def get_directive(name: str) -> Optional[Directive]:
    """Get a directive by name."""
    return DIRECTIVE_REGISTRY.get(name)


def list_directive_names() -> List[str]:
    """List all registered directive names."""
    return sorted(DIRECTIVE_REGISTRY.keys())


def clear_dynamic_directives() -> None:
    """
    Clear dynamically registered directives (keep core ones).

    Useful for testing or resetting state.
    """
    # Keep only core directives
    core_names = {
        "reorder_filters_first",
        "remove_redundant_ops",
        "tighten_filters",
        "loosen_filters",
        "bump_text_length_min_len",
    }
    to_remove = [k for k in DIRECTIVE_REGISTRY if k not in core_names]
    for k in to_remove:
        del DIRECTIVE_REGISTRY[k]