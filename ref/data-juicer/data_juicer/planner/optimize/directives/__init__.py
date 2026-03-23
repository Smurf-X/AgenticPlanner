# -*- coding: utf-8 -*-
"""Directives for pipeline optimization."""

from data_juicer.planner.optimize.directives.adjust_params import BumpMinLenDirective
from data_juicer.planner.optimize.directives.adjust_threshold import (
    AdjustThresholdDirective,
    LoosenFiltersDirective,
    TightenFiltersDirective,
)
from data_juicer.planner.optimize.directives.base import Directive, DirectiveResult
from data_juicer.planner.optimize.directives.change_model import (
    LLMChangeModelDirective,
    MODEL_INFO,
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
from data_juicer.planner.optimize.directives.registry import (
    DIRECTIVE_REGISTRY,
    clear_dynamic_directives,
    get_directive,
    list_directive_names,
    register_directive,
    register_few_shot_directive,
    register_gleaning_directive,
    register_llm_change_model_directive,
    register_model_by_type_directive,
    register_prompt_rewrite_directive,
    register_single_op_model_directive,
    register_swap_model_directive,
    register_threshold_directive,
)
from data_juicer.planner.optimize.directives.reorder import ReorderFiltersFirstDirective
from data_juicer.planner.optimize.directives.rewrite_prompt import (
    AddFewShotExamplesDirective,
    RewritePromptDirective,
)

__all__ = [
    # Base classes
    "Directive",
    "DirectiveResult",
    # Non-LLM operator directives
    "AdjustThresholdDirective",
    "BumpMinLenDirective",
    "LoosenFiltersDirective",
    "RemoveRedundantOpDirective",
    "ReorderFiltersFirstDirective",
    "TightenFiltersDirective",
    # LLM operator directives - prompt
    "AddFewShotExamplesDirective",
    "RewritePromptDirective",
    # LLM operator directives - gleaning
    "AddGleaningDirective",
    "RemoveGleaningDirective",
    # LLM operator directives - model
    "SwapApiModelDirective",
    "SwapSingleOpModelDirective",
    "SwapModelByTypeDirective",
    "LLMChangeModelDirective",
    "MODEL_INFO",
    # Registry
    "DIRECTIVE_REGISTRY",
    "clear_dynamic_directives",
    "get_directive",
    "list_directive_names",
    "register_directive",
    # Registration functions
    "register_single_op_model_directive",
    "register_model_by_type_directive",
    "register_swap_model_directive",
    "register_llm_change_model_directive",
    "register_few_shot_directive",
    "register_gleaning_directive",
    "register_prompt_rewrite_directive",
    "register_threshold_directive",
]