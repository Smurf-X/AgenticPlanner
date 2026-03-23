# -*- coding: utf-8 -*-
"""
Model change directives for LLM-based operators.

Provides fine-grained control over model replacement:
1. SwapSingleOpModelDirective - Replace model for a specific operator
2. SwapModelByTypeDirective - Replace models by operator type
3. SwapApiModelDirective - Global replacement (legacy, kept for compatibility)
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, MutableMapping, Optional, Set

from data_juicer.planner.contracts.recipe import DJExecutableConfig
from data_juicer.planner.optimize.directives.base import Directive, DirectiveResult


# Model information for LLM-based recommendation
MODEL_INFO = """
Model comparison (pricing per 1M tokens):
| Model          | Input  | Output | Context  | Best For                    |
|----------------|--------|--------|----------|----------------------------|
| gpt-4o-mini    | $0.15  | $0.60  | 128K     | Simple tasks, cost-saving   |
| gpt-4o         | $2.50  | $10.00 | 128K     | Balanced quality/cost       |
| gpt-4-turbo    | $10.00 | $30.00 | 128K     | Complex reasoning           |
| claude-3-haiku | $0.25  | $1.25  | 200K     | Fast, simple tasks          |
| claude-3-sonnet| $3.00  | $15.00 | 200K     | Balanced tasks              |
| claude-3-opus  | $15.00 | $75.00 | 200K     | Complex analysis            |
"""


def _replace_model_in_dict(
    obj: Any,
    old_model: str,
    new_model: str,
    keys: tuple[str, ...] = ("api_model", "model"),
) -> bool:
    """Recursively replace model in a dict. Returns True if changed."""
    changed = False
    if isinstance(obj, MutableMapping):
        for k in keys:
            if k in obj and obj[k] == old_model:
                obj[k] = new_model
                changed = True
        for v in obj.values():
            if _replace_model_in_dict(v, old_model, new_model, keys):
                changed = True
    elif isinstance(obj, list):
        for item in obj:
            if _replace_model_in_dict(item, old_model, new_model, keys):
                changed = True
    return changed


def _get_op_name(step: Dict[str, Any]) -> Optional[str]:
    """Get operator name from a process step."""
    if not isinstance(step, dict) or len(step) != 1:
        return None
    return next(iter(step.keys()), None)


def _get_op_params(step: Dict[str, Any]) -> Dict[str, Any]:
    """Get operator params from a process step."""
    if not isinstance(step, dict) or len(step) != 1:
        return {}
    params = next(iter(step.values()), {})
    return params if isinstance(params, dict) else {}


class SwapSingleOpModelDirective(Directive):
    """
    Replace model for a specific operator by name.

    This is the recommended directive for fine-grained model control.
    """

    name = "swap_single_op_model"

    def __init__(
        self,
        op_name: str,
        from_model: str,
        to_model: str,
        model_keys: tuple[str, ...] = ("api_model", "model"),
    ) -> None:
        """
        Args:
            op_name: Name of the operator to modify (e.g., "llm_filter")
            from_model: Source model name (only replace if matches)
            to_model: Target model name
            model_keys: Parameter keys that contain model name
        """
        self.op_name = op_name
        self.from_model = from_model
        self.to_model = to_model
        self.model_keys = model_keys

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
        changed = False
        changes: List[Dict[str, Any]] = []

        for i, step in enumerate(after.get("process", [])):
            op_name = _get_op_name(step)
            if op_name != self.op_name:
                continue

            params = _get_op_params(step)
            for key in self.model_keys:
                if key in params and params[key] == self.from_model:
                    params[key] = self.to_model
                    changed = True
                    changes.append({
                        "index": i,
                        "op_name": op_name,
                        "key": key,
                        "old": self.from_model,
                        "new": self.to_model,
                    })

        return DirectiveResult(
            ok=True,
            applied=changed,
            directive_name=self.name,
            message=f"changed {len(changes)} model(s) in {self.op_name}" if changed else f"no matching operator or model",
            config_before=before,
            config_after=after,
            details={"changes": changes},
        )


class SwapModelByTypeDirective(Directive):
    """
    Replace model for all operators of a specific type.

    Example: Replace models for all "filter" type operators.
    """

    name = "swap_model_by_type"

    # Operator types that typically use LLM
    LLM_OP_TYPES = {
        "mapper",
        "filter",
        "aggregator",
    }

    def __init__(
        self,
        op_type: str,
        from_model: str,
        to_model: str,
        model_keys: tuple[str, ...] = ("api_model", "model"),
    ) -> None:
        """
        Args:
            op_type: Operator type to target (e.g., "filter", "mapper")
            from_model: Source model name
            to_model: Target model name
            model_keys: Parameter keys that contain model name
        """
        self.op_type = op_type.lower()
        self.from_model = from_model
        self.to_model = to_model
        self.model_keys = model_keys

    def _get_op_type(self, op_name: str) -> str:
        """Get operator type from name or registry."""
        # Try to get from registry
        from data_juicer.tools.op_search import OPSearcher
        try:
            searcher = OPSearcher(specified_op_list=[op_name])
            if searcher.op_records:
                return searcher.op_records[0].type.lower()
        except Exception:
            pass
        # Fallback: infer from name prefix
        for t in ["filter", "mapper", "deduplicator", "selector", "aggregator"]:
            if t in op_name.lower():
                return t
        return "mapper"  # Default

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
        changed = False
        changes: List[Dict[str, Any]] = []

        for i, step in enumerate(after.get("process", [])):
            op_name = _get_op_name(step)
            if not op_name:
                continue

            # Check if this operator matches the target type
            op_type = self._get_op_type(op_name)
            if op_type != self.op_type:
                continue

            params = _get_op_params(step)
            for key in self.model_keys:
                if key in params and params[key] == self.from_model:
                    params[key] = self.to_model
                    changed = True
                    changes.append({
                        "index": i,
                        "op_name": op_name,
                        "op_type": op_type,
                        "old": self.from_model,
                        "new": self.to_model,
                    })

        return DirectiveResult(
            ok=True,
            applied=changed,
            directive_name=self.name,
            message=f"changed {len(changes)} model(s) in {self.op_type} operators" if changed else f"no matching operators of type {self.op_type}",
            config_before=before,
            config_after=after,
            details={"changes": changes},
        )


class SwapApiModelDirective(Directive):
    """
    Global model replacement for ALL operators matching the source model.

    WARNING: This replaces models across all operators. Use with caution.
    Consider using SwapSingleOpModelDirective or SwapModelByTypeDirective
    for finer control.
    """

    name = "swap_api_model"

    def __init__(
        self,
        from_model: str,
        to_model: str,
        model_keys: tuple[str, ...] = ("api_model", "model"),
    ) -> None:
        """
        Args:
            from_model: Source model name
            to_model: Target model name
            model_keys: Parameter keys that contain model name
        """
        self.from_model = from_model
        self.to_model = to_model
        self.model_keys = model_keys

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
        changed = _replace_model_in_dict(
            after.get("process", []),
            self.from_model,
            self.to_model,
            self.model_keys,
        )

        return DirectiveResult(
            ok=True,
            applied=changed,
            directive_name=self.name,
            message=f"{self.from_model} -> {self.to_model}" if changed else "no matching model field",
            config_before=before,
            config_after=after,
        )


class LLMChangeModelDirective(Directive):
    """
    Use LLM to recommend and apply model changes for a specific operator.

    The LLM analyzes the operator's task complexity and recommends
    an appropriate model based on cost/quality trade-offs.
    """

    name = "llm_change_model"

    def __init__(
        self,
        op_name: str,
        allowed_models: List[str],
        optimize_goal: str = "balanced",  # "cost", "quality", "balanced"
        llm_client: Optional[Any] = None,
    ) -> None:
        """
        Args:
            op_name: Name of the operator to modify
            allowed_models: List of allowed model choices
            optimize_goal: Optimization objective
            llm_client: LLM client for making recommendations
        """
        self.op_name = op_name
        self.allowed_models = allowed_models
        self.optimize_goal = optimize_goal
        self._llm_client = llm_client

    def set_llm_client(self, client: Any) -> None:
        """Set the LLM client."""
        self._llm_client = client

    def _get_recommendation(
        self,
        op_config: Dict[str, Any],
    ) -> Optional[str]:
        """Use LLM to recommend a model."""
        if not self._llm_client:
            return None

        prompt = f"""You are an expert at choosing the most suitable LLM model for a given task.

Original Operation:
{op_config}

Allowed Models: {self.allowed_models}

Optimization Goal: {self.optimize_goal}
- "cost": Prioritize cheaper models
- "quality": Prioritize better performance
- "balanced": Balance cost and quality

{MODEL_INFO}

Based on the task complexity (prompt, output schema) and optimization goal,
recommend the best model from the allowed list.

Return ONLY a JSON object with one field:
{{"recommended_model": "model_name"}}
"""
        try:
            import json
            response = self._llm_client.generate(
                system_prompt="You are a helpful AI assistant for model selection.",
                user_prompt=prompt,
                temperature=0.1,
            )
            # Parse response
            text = response.strip()
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            result = json.loads(text)
            model = result.get("recommended_model")
            if model in self.allowed_models:
                return model
        except Exception:
            pass
        return None

    def apply(self, cfg: DJExecutableConfig) -> DirectiveResult:
        before = deepcopy(cfg)
        proc = before.get("process")
        if not isinstance(proc, list):
            return DirectiveResult(
                ok=False,
                applied=False,
                directive_name=self.name,
                message="no process",
                config_before=before,
                config_after=before,
            )

        # Find the target operator
        target_step = None
        target_index = -1
        for i, step in enumerate(proc):
            op_name = _get_op_name(step)
            if op_name == self.op_name:
                target_step = step
                target_index = i
                break

        if target_step is None:
            return DirectiveResult(
                ok=False,
                applied=False,
                directive_name=self.name,
                message=f"operator {self.op_name} not found",
                config_before=before,
                config_after=before,
            )

        # Get current model
        params = _get_op_params(target_step)
        current_model = params.get("model") or params.get("api_model")

        # Get LLM recommendation
        recommended_model = self._get_recommendation(target_step)

        if not recommended_model:
            return DirectiveResult(
                ok=False,
                applied=False,
                directive_name=self.name,
                message="failed to get LLM recommendation",
                config_before=before,
                config_after=before,
            )

        if recommended_model == current_model:
            return DirectiveResult(
                ok=True,
                applied=False,
                directive_name=self.name,
                message="LLM recommends keeping current model",
                config_before=before,
                config_after=before,
            )

        # Apply the change
        after = deepcopy(before)
        after["process"][target_index][self.op_name]["model"] = recommended_model

        return DirectiveResult(
            ok=True,
            applied=True,
            directive_name=self.name,
            message=f"LLM recommended: {current_model} -> {recommended_model}",
            config_before=before,
            config_after=after,
            details={
                "recommended_model": recommended_model,
                "optimize_goal": self.optimize_goal,
            },
        )


__all__ = [
    "SwapSingleOpModelDirective",
    "SwapModelByTypeDirective",
    "SwapApiModelDirective",
    "LLMChangeModelDirective",
    "MODEL_INFO",
]