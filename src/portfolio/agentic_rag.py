"""Agentic RAG Portfolio Allocator.

ReAct framework (Reason → Act → Observe) using Anthropic Claude tool use.
The agent reasons about portfolio allocation using structured tools that
expose model predictions, regime state, and macro context.
Every reasoning trace is logged to results/agent_traces/.
"""
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_TRACE_DIR = Path("results/agent_traces")
_MODEL_ID = "claude-sonnet-4-6"
_MAX_TOKENS = 4096
_MAX_WEIGHT = 0.25


def _get_anthropic_client():
    """Return an Anthropic client using ANTHROPIC_API_KEY env var."""
    try:
        import anthropic
    except ImportError as e:
        raise ImportError("anthropic SDK required: pip install anthropic") from e
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY must be set as an environment variable.")
    return anthropic.Anthropic(api_key=api_key)


def _project_weights(w: np.ndarray, max_w: float = _MAX_WEIGHT) -> np.ndarray:
    w = np.clip(w, 0.0, max_w)
    total = w.sum()
    return w / total if total > 0 else np.full(len(w), 1.0 / len(w))


# ---------------------------------------------------------------------------
# Tool implementations (callable by the agent)
# ---------------------------------------------------------------------------

def tool_get_predictions(
    model_name: str,
    asset: str,
    predictions_store: dict[str, pd.DataFrame],
) -> dict:
    """Return predicted return and confidence for an asset from a named model."""
    key = model_name.lower()
    if key not in predictions_store:
        return {"error": f"Model '{model_name}' not found. Available: {list(predictions_store)}"}
    pred_df = predictions_store[key]
    if asset not in pred_df.columns:
        return {"error": f"Asset '{asset}' not in predictions for {model_name}."}
    latest = pred_df[asset].dropna().iloc[-1] if not pred_df[asset].dropna().empty else np.nan
    return {
        "model": model_name,
        "asset": asset,
        "predicted_return": float(latest) if not np.isnan(latest) else None,
        "confidence": "point_estimate",
    }


def tool_get_regime_state(
    date: str,
    regime_probs: pd.DataFrame,
) -> dict:
    """Return regime probabilities for a given date."""
    try:
        ts = pd.Timestamp(date)
    except Exception:
        return {"error": f"Invalid date: {date}"}
    if ts not in regime_probs.index:
        ts = regime_probs.index[regime_probs.index.searchsorted(ts) - 1]
    row = regime_probs.loc[ts]
    state_cols = [c for c in row.index if c.startswith("state_")]
    return {
        "date": str(ts.date()),
        "probabilities": {c: float(row[c]) for c in state_cols},
        "predicted_state": str(row.get("predicted_state", "unknown")),
        "predicted_label": str(row.get("predicted_label", "unknown")),
    }


def tool_get_macro_context(
    date: str,
    macro_df: pd.DataFrame,
    top_n: int = 10,
) -> dict:
    """Return a snapshot of the most recent macro feature values."""
    try:
        ts = pd.Timestamp(date)
    except Exception:
        return {"error": f"Invalid date: {date}"}
    idx = macro_df.index.searchsorted(ts, side="right") - 1
    if idx < 0:
        return {"error": "Date before macro data starts."}
    row = macro_df.iloc[idx]
    top = row.abs().nlargest(top_n)
    return {
        "date": str(macro_df.index[idx].date()),
        "top_features": {k: float(v) for k, v in row[top.index].items()},
    }


def tool_propose_allocation(
    weights_dict: dict[str, float],
    asset_universe: list[str],
    max_weight: float = _MAX_WEIGHT,
) -> dict:
    """Validate and normalise a proposed weight vector.

    Parameters
    ----------
    weights_dict:
        Mapping asset -> proposed weight (fractions, not percentages).
    asset_universe:
        Full list of ETFs the allocator can trade.
    max_weight:
        Per-asset cap.

    Returns
    -------
    dict
        Validated and normalised weights, plus any constraint violations.
    """
    unknown = [k for k in weights_dict if k not in asset_universe]
    if unknown:
        return {"error": f"Unknown assets: {unknown}. Universe: {asset_universe}"}

    w = np.array([weights_dict.get(a, 0.0) for a in asset_universe], dtype=float)
    violations = []
    if w.min() < 0:
        violations.append(f"Negative weights not allowed: {w[w < 0].min():.4f}")
    caps = [(a, float(w[i])) for i, a in enumerate(asset_universe) if w[i] > max_weight]
    if caps:
        violations.append(f"Weights exceed {max_weight:.0%} cap: {caps}")

    w_proj = _project_weights(w, max_weight)
    return {
        "status": "accepted",
        "violations": violations,
        "weights": {a: float(w_proj[i]) for i, a in enumerate(asset_universe)},
        "sum": float(w_proj.sum()),
    }


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class AgenticRAGAllocator:
    """Claude-powered ReAct portfolio allocator.

    Parameters
    ----------
    asset_universe:
        List of ETF tickers in the investment universe.
    predictions_store:
        Dict mapping model_name -> pd.DataFrame with columns per ticker.
    regime_probs:
        DataFrame with state probability columns indexed by date.
    macro_df:
        Macro feature DataFrame indexed by date.
    model_id:
        Anthropic model to use.
    max_tokens:
        Max tokens for the model response.
    regime_dim:
        If > 0, regime context is included in the system prompt.
    """

    def __init__(
        self,
        asset_universe: list[str],
        predictions_store: dict[str, pd.DataFrame],
        regime_probs: pd.DataFrame,
        macro_df: pd.DataFrame,
        model_id: str = _MODEL_ID,
        max_tokens: int = _MAX_TOKENS,
        regime_dim: int = 0,
    ):
        self.asset_universe = asset_universe
        self.predictions_store = predictions_store
        self.regime_probs = regime_probs
        self.macro_df = macro_df
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.regime_dim = regime_dim
        self._client = _get_anthropic_client()
        _TRACE_DIR.mkdir(parents=True, exist_ok=True)

    def _build_tools(self) -> list[dict]:
        return [
            {
                "name": "get_predictions",
                "description": (
                    "Returns the predicted next-period return and confidence for a specific "
                    "asset from a named prediction model."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "model_name": {"type": "string", "description": "Name of the prediction model"},
                        "asset": {"type": "string", "description": "ETF ticker symbol"},
                    },
                    "required": ["model_name", "asset"],
                },
            },
            {
                "name": "get_regime_state",
                "description": "Returns current market regime probabilities and the most likely regime label.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "date": {"type": "string", "description": "ISO date string (YYYY-MM-DD)"},
                    },
                    "required": ["date"],
                },
            },
            {
                "name": "get_macro_context",
                "description": "Returns a snapshot of the most significant macro feature values for a given date.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "date": {"type": "string", "description": "ISO date string (YYYY-MM-DD)"},
                    },
                    "required": ["date"],
                },
            },
            {
                "name": "propose_allocation",
                "description": (
                    "Validates and commits a proposed portfolio weight vector. "
                    "Weights must be non-negative, sum to 1, and respect the 25% per-asset cap."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "weights": {
                            "type": "object",
                            "description": "Mapping from ticker to weight fraction (e.g. {\"SPY\": 0.10})",
                            "additionalProperties": {"type": "number"},
                        },
                    },
                    "required": ["weights"],
                },
            },
        ]

    def _dispatch_tool(self, name: str, inputs: dict) -> Any:
        if name == "get_predictions":
            return tool_get_predictions(inputs["model_name"], inputs["asset"], self.predictions_store)
        elif name == "get_regime_state":
            return tool_get_regime_state(inputs["date"], self.regime_probs)
        elif name == "get_macro_context":
            return tool_get_macro_context(inputs["date"], self.macro_df)
        elif name == "propose_allocation":
            return tool_propose_allocation(inputs["weights"], self.asset_universe)
        else:
            return {"error": f"Unknown tool: {name}"}

    def allocate(
        self,
        decision_date: pd.Timestamp,
        current_weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Run the ReAct agent loop to produce portfolio weights.

        Parameters
        ----------
        decision_date:
            The date for which to compute the allocation.
        current_weights:
            Previous period weights (for turnover context).

        Returns
        -------
        np.ndarray
            Portfolio weights of shape (n_assets,).
        """
        date_str = decision_date.strftime("%Y-%m-%d")
        system_prompt = (
            "You are a quantitative portfolio manager allocating capital across ETFs. "
            "Your goal is to construct a diversified, risk-aware portfolio using "
            "statistical model predictions and macroeconomic context. "
            "Constraints: long-only, weights sum to 1, max 25% per asset. "
            "Use the provided tools to gather evidence before proposing an allocation. "
            "Always call propose_allocation as your final action."
        )
        if self.regime_dim > 0:
            system_prompt += " Pay special attention to the current market regime state."

        user_prompt = (
            f"Please construct a portfolio allocation for {date_str}. "
            f"Asset universe: {self.asset_universe}. "
            "Use get_predictions, get_regime_state, and get_macro_context to inform your reasoning, "
            "then call propose_allocation with your final weights."
        )

        messages = [{"role": "user", "content": user_prompt}]
        tools = self._build_tools()
        trace = {"date": date_str, "turns": [], "final_weights": None}

        final_weights = None
        max_turns = 10

        for turn in range(max_turns):
            response = self._client.messages.create(
                model=self.model_id,
                max_tokens=self.max_tokens,
                system=system_prompt,
                tools=tools,
                messages=messages,
                temperature=0.0,
            )

            trace["turns"].append({
                "turn": turn,
                "stop_reason": response.stop_reason,
                "content": [c.model_dump() if hasattr(c, "model_dump") else str(c) for c in response.content],
            })

            if response.stop_reason == "end_turn":
                break

            if response.stop_reason != "tool_use":
                break

            # Execute tool calls
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                result = self._dispatch_tool(block.name, block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result),
                })
                trace["turns"].append({"tool": block.name, "input": block.input, "output": result})

                # Capture final allocation if propose_allocation was called
                if block.name == "propose_allocation" and "weights" in result:
                    final_weights = result["weights"]

            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

        # Save trace
        trace_file = _TRACE_DIR / f"agent_{date_str}_{datetime.now().strftime('%H%M%S')}.json"
        with open(trace_file, "w") as f:
            json.dump(trace, f, indent=2, default=str)
        logger.info("Agent trace saved: %s", trace_file)

        if final_weights is None:
            logger.warning("Agent did not call propose_allocation for %s. Using equal weights.", date_str)
            return np.full(len(self.asset_universe), 1.0 / len(self.asset_universe))

        w = np.array([final_weights.get(a, 0.0) for a in self.asset_universe])
        trace["final_weights"] = {a: float(w[i]) for i, a in enumerate(self.asset_universe)}
        return _project_weights(w)
