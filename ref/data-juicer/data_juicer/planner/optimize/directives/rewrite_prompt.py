# -*- coding: utf-8 -*-
"""LLM-driven prompt rewriting directive for LLM-based operators."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional

from data_juicer.planner.contracts.recipe import DJExecutableConfig
from data_juicer.planner.optimize.directives.base import Directive, DirectiveResult


# Operators that have a 'prompt' parameter
_LLM_PROMPT_OPS = {
    "map_wrapper": "prompt",
    "filter_wrapper": "prompt",
    "llm_map": "prompt",
    "llm_filter": "prompt",
}


class RewritePromptDirective(Directive):
    """
    Rewrite the prompt of an LLM-based operator.

    This directive can either:
    1. Apply a static prompt template transformation
    2. Use an LLM to improve the prompt (future: requires LLM client)

    For v1, we support template-based transformations.
    """

    name = "rewrite_prompt"

    def __init__(
        self,
        op_index: int,
        new_prompt: Optional[str] = None,
        prompt_suffix: Optional[str] = None,
        clarify_instruction: Optional[str] = None,
    ) -> None:
        """
        Args:
            op_index: Index of the operator in the process list
            new_prompt: Replace the entire prompt with this
            prompt_suffix: Append this to the existing prompt
            clarify_instruction: Prepend clarification to the prompt
        """
        self.op_index = op_index
        self.new_prompt = new_prompt
        self.prompt_suffix = prompt_suffix
        self.clarify_instruction = clarify_instruction

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

        # Check if this operator has a prompt parameter
        prompt_key = _LLM_PROMPT_OPS.get(op_name, "prompt")
        if prompt_key not in params:
            return DirectiveResult(
                ok=True,
                applied=False,
                directive_name=self.name,
                message=f"operator {op_name} has no prompt parameter",
                config_before=before,
                config_after=before,
            )

        old_prompt = params.get(prompt_key, "")
        new_params = dict(params)

        if self.new_prompt:
            new_params[prompt_key] = self.new_prompt
        elif self.prompt_suffix:
            new_params[prompt_key] = old_prompt + "\n\n" + self.prompt_suffix
        elif self.clarify_instruction:
            new_params[prompt_key] = f"[Instruction: {self.clarify_instruction}]\n\n{old_prompt}"
        else:
            return DirectiveResult(
                ok=True,
                applied=False,
                directive_name=self.name,
                message="no transformation specified",
                config_before=before,
                config_after=before,
            )

        after["process"][self.op_index] = {op_name: new_params}

        return DirectiveResult(
            ok=True,
            applied=True,
            directive_name=self.name,
            message=f"rewrote prompt for {op_name}",
            config_before=before,
            config_after=after,
            details={
                "op_index": self.op_index,
                "op_name": op_name,
                "old_prompt_preview": old_prompt[:100] + "..." if len(old_prompt) > 100 else old_prompt,
            },
        )


class AddFewShotExamplesDirective(Directive):
    """Add few-shot examples to an LLM operator's prompt."""

    name = "add_few_shot_examples"

    def __init__(self, op_index: int, examples: List[Dict[str, str]]) -> None:
        """
        Args:
            op_index: Index of the operator in the process list
            examples: List of example dicts with 'input' and 'output' keys
        """
        self.op_index = op_index
        self.examples = examples

    def apply(self, cfg: DJExecutableConfig) -> DirectiveResult:
        before = deepcopy(cfg)
        proc = before.get("process")
        if not isinstance(proc, list) or not self.examples:
            return DirectiveResult(
                ok=True,
                applied=False,
                directive_name=self.name,
                message="no process or no examples",
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

        prompt_key = _LLM_PROMPT_OPS.get(op_name, "prompt")
        old_prompt = params.get(prompt_key, "")
        new_params = dict(params)

        # Build few-shot section
        examples_text = "\n\n## Examples:\n"
        for i, ex in enumerate(self.examples, 1):
            inp = ex.get("input", "")
            out = ex.get("output", "")
            examples_text += f"\n### Example {i}:\n"
            examples_text += f"Input: {inp}\n"
            examples_text += f"Output: {out}\n"

        new_params[prompt_key] = old_prompt + examples_text
        after["process"][self.op_index] = {op_name: new_params}

        return DirectiveResult(
            ok=True,
            applied=True,
            directive_name=self.name,
            message=f"added {len(self.examples)} example(s) to {op_name}",
            config_before=before,
            config_after=after,
            details={"op_index": self.op_index, "examples_count": len(self.examples)},
        )