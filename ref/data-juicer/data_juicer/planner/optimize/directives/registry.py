# -*- coding: utf-8 -*-
"""Registry for optimization directives."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from data_juicer.planner.optimize.directives.adjust_params import BumpMinLenDirective
from data_juicer.planner.optimize.directives.adjust_threshold import (
    AdjustThresholdDirective,
    LoosenFiltersDirective,
    TightenFiltersDirective,
)
from data_juicer.planner.optimize.directives.base import Directive
from data_juicer.planner.optimize.directives.change_model import (
    LLMChangeModelDirective,
    SwapApiModelDirective,
    SwapModelByTypeDirective,
    SwapSingleOpModelDirective,
)
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
from data_juicer.planner.optimize.op_locator import OpLocator

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


# ============================================================================
# Model Change Directive Registration Functions
# ============================================================================

def register_single_op_model_directive(
    locator: OpLocator,
    from_model: str,
    to_model: str,
    name: Optional[str] = None,
) -> str:
    """
    Register a directive to change model for a specific operator.

    Args:
        locator: Locator for the target operator
        from_model: Source model name (only replace if matches)
        to_model: Target model name
        name: Custom registry name

    Returns:
        The directive name
    """
    d = SwapSingleOpModelDirective(locator, from_model, to_model)
    key = name or f"swap_model:{from_model}->{to_model}"
    DIRECTIVE_REGISTRY[key] = d
    return key


def register_model_by_type_directive(
    op_type: str,
    from_model: str,
    to_model: str,
    name: Optional[str] = None,
) -> str:
    """
    Register a directive to change model for all operators of a type.

    Args:
        op_type: Operator type (e.g., "filter", "mapper")
        from_model: Source model name
        to_model: Target model name
        name: Custom registry name

    Returns:
        The directive name
    """
    d = SwapModelByTypeDirective(op_type, from_model, to_model)
    key = name or f"swap_model_by_type:{op_type}:{from_model}->{to_model}"
    DIRECTIVE_REGISTRY[key] = d
    return key


def register_swap_model_directive(from_model: str, to_model: str) -> str:
    """
    Register a global model swap directive (replaces ALL matching models).

    WARNING: This affects all operators. Consider using
    register_single_op_model_directive for finer control.

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


def register_llm_change_model_directive(
    locator: OpLocator,
    allowed_models: List[str],
    optimize_goal: str = "balanced",
    llm_client: Optional[Any] = None,
    name: Optional[str] = None,
) -> str:
    """
    Register an LLM-based model change directive.

    The LLM will analyze the operator and recommend the best model.

    Args:
        locator: Locator for the target operator
        allowed_models: List of allowed model choices
        optimize_goal: "cost", "quality", or "balanced"
        llm_client: LLM client for making recommendations
        name: Custom registry name

    Returns:
        The directive name
    """
    d = LLMChangeModelDirective(locator, allowed_models, optimize_goal, llm_client)
    key = name or "llm_change_model"
    DIRECTIVE_REGISTRY[key] = d
    return key


# ============================================================================
# Prompt Directive Registration Functions
# ============================================================================

def register_prompt_rewrite_directive(
    locator: OpLocator,
    new_prompt: Optional[str] = None,
    prompt_suffix: Optional[str] = None,
    clarify_instruction: Optional[str] = None,
    name: Optional[str] = None,
) -> str:
    """
    Register a prompt rewrite directive for a specific operator.

    Args:
        locator: Locator for the target operator
        new_prompt: Replace prompt entirely
        prompt_suffix: Append to prompt
        clarify_instruction: Prepend clarification
        name: Custom registry name

    Returns:
        The directive name
    """
    d = RewritePromptDirective(locator, new_prompt, prompt_suffix, clarify_instruction)
    key = name or "rewrite_prompt"
    DIRECTIVE_REGISTRY[key] = d
    return key


def register_few_shot_directive(
    locator: OpLocator,
    examples: List[Dict[str, str]],
    name: Optional[str] = None,
) -> str:
    """
    Register a few-shot examples directive.

    Args:
        locator: Locator for the target operator
        examples: List of {input, output} example dicts
        name: Custom registry name

    Returns:
        The directive name
    """
    d = AddFewShotExamplesDirective(locator, examples)
    key = name or "add_few_shot_examples"
    DIRECTIVE_REGISTRY[key] = d
    return key


# ============================================================================
# Gleaning Directive Registration Functions
# ============================================================================

def register_gleaning_directive(
    locator: OpLocator,
    max_rounds: int = 2,
    gleaning_prompt: Optional[str] = None,
    name: Optional[str] = None,
) -> str:
    """
    Register a gleaning directive for a specific operator.

    Args:
        locator: Locator for the target operator
        max_rounds: Maximum gleaning rounds
        gleaning_prompt: Custom gleaning prompt
        name: Custom registry name

    Returns:
        The directive name
    """
    d = AddGleaningDirective(locator, max_rounds, gleaning_prompt)
    key = name or "add_gleaning"
    DIRECTIVE_REGISTRY[key] = d
    return key


# ============================================================================
# Threshold Directive Registration Functions
# ============================================================================

def register_threshold_directive(
    op_type: str,
    param_name: str,
    delta: float,
    direction: Optional[str] = None,
    name: Optional[str] = None,
) -> str:
    """
    Register a parameterized threshold adjustment directive.

    Args:
        op_type: Operator type
        param_name: Parameter to adjust
        delta: Adjustment amount
        direction: "increase" or "decrease"
        name: Custom registry name

    Returns:
        The directive name
    """
    d = AdjustThresholdDirective(op_type, param_name, delta, direction)
    key = name or f"adjust_{op_type}_{param_name}"
    DIRECTIVE_REGISTRY[key] = d
    return key


# ============================================================================
# Utility Functions
# ============================================================================

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