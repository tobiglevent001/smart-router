"""SmartRouter — Hermes Agent plugin.

Injects routing recommendations into the conversation via the
``pre_llm_call`` hook. The main model sees the recommendation as
context and can act on it (e.g. delegate to a capable sub-agent)
or ignore it for simple responses.

The plugin does NOT modify the model selection directly — it provides
actionable context that the agent can use autonomously.

Installation
------------
Copy to ~/.hermes/plugins/smart_router/::

    mkdir -p ~/.hermes/plugins/smart_router
    cp -r plugins/smart_router/* ~/.hermes/plugins/smart_router/

Restart Hermes. The plugin activates automatically.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

from smart_router import Classifier, ClassificationResult
from smart_router.tiers import BuiltinTiers, TierConfig, TierLevel

logger = logging.getLogger(__name__)

# ── Singleton classifier instance (lazy-init) ─────────────────────
_classifier: Optional[Classifier] = None
_stats_collector = None


def _get_classifier() -> Classifier:
    """Get or create the global classifier instance."""
    global _classifier
    if _classifier is None:
        # Try loading user config, fall back to defaults
        config_path = os.environ.get(
            "SMART_ROUTER_CONFIG",
            os.path.expanduser("~/.hermes/smart-router-config.yaml"),
        )
        if os.path.exists(config_path):
            _classifier = Classifier.from_yaml(config_path)
            logger.info("SmartRouter: loaded config from %s", config_path)
        else:
            _classifier = Classifier()
            logger.info("SmartRouter: using default tier config")
    return _classifier


def _get_stats():
    """Get or create stats collector."""
    global _stats_collector
    if _stats_collector is None:
        from smart_router.stats import StatsCollector
        _stats_collector = StatsCollector()
    return _stats_collector


# ── Plugin registration entry point ──────────────────────────────

def register(ctx) -> None:
    """Register the SmartRouter plugin with Hermes."""
    logger.info("SmartRouter plugin registering...")

    # Register the pre_llm_call hook
    ctx.register_hook("pre_llm_call", on_pre_llm_call)

    # Register a slash command for manual routing
    ctx.register_command(
        name="route",
        handler=_cmd_route,
        help="Classify and route a task to the optimal model",
    )

    # Register stats command
    ctx.register_command(
        name="route-stats",
        handler=_cmd_route_stats,
        help="Show SmartRouter routing statistics and cost savings report",
    )

    logger.info("SmartRouter plugin registered successfully")


# ── Hook handler ──────────────────────────────────────────────────

def on_pre_llm_call(
    session_id: str = "",
    user_message: str = "",
    conversation_history: Any = None,
    is_first_turn: bool = False,
    model: str = "",
    platform: str = "",
    sender_id: str = "",
    **kwargs,
) -> Optional[Dict[str, Any]]:
    """``pre_llm_call`` hook — classify message and inject routing context.

    Returns a dict with a ``context`` key that is appended to the
    current turn's user message. The agent can use this information
    to decide whether to delegate the task to a more capable model.
    """
    if not user_message or user_message.startswith("/"):
        # Don't process slash commands
        return None

    classifier = _get_classifier()
    result = classifier.classify(user_message)

    # Build routing context
    tier = result.tier.name
    model_rec = result.tier_config.model
    reason = result.reason

    context = (
        f"[SmartRouter Analysis]\n"
        f"Task complexity: {tier} (score: {result.score:.0%})\n"
        f"Recommended model: {model_rec}\n"
        f"Reason: {reason}\n"
        f"If this task requires advanced capabilities, consider delegating "
        f"it using /goal with the recommended model."
    )

    # Log the routing decision
    logger.info(
        "SmartRouter: [%s] → %s (%s) — %s",
        tier, model_rec, result.tier_config.provider or "default", reason,
    )

    # Record stats
    try:
        stats = _get_stats()
        stats.record(result, features_summary=reason)
    except Exception as exc:
        logger.debug("SmartRouter stats recording failed: %s", exc)

    return {"context": context}


# ── Slash command handlers ────────────────────────────────────────

def _cmd_route(args: str = "") -> str:
    """Manual route command — classify arbitrary text."""
    if not args.strip():
        return (
            "Usage: /route <your task description>\n"
            "Example: /route 帮我写一个分布式爬虫"
        )

    classifier = _get_classifier()
    result = classifier.classify(args)

    lines = [
        f"📊 SmartRouter Analysis",
        f"",
        f"  Tier:      {result.tier.name}",
        f"  Model:     {result.tier_config.model}",
        f"  Provider:  {result.tier_config.provider or '(default)'}",
    ]
    if result.tier_config.reasoning_effort:
        lines.append(f"  Reasoning: {result.tier_config.reasoning_effort}")
    lines.extend([
        f"  Score:     {result.score:.0%}",
        f"  Reason:    {result.reason}",
        f"",
        f"Message: \"{args[:80]}{'...' if len(args) > 80 else ''}\"",
    ])
    return "\n".join(lines)


def _cmd_route_stats(args: str = "") -> str:
    """Show routing statistics and cost savings."""
    try:
        stats = _get_stats()
        report = stats.get_report()
    except Exception as exc:
        return f"Error getting stats: {exc}"

    if report.get("status") == "no_data":
        return "📊 No routing data collected yet. Send some messages first."

    lines = [
        f"📊 SmartRouter Statistics",
        f"",
        f"  Total routes:      {report['total_routes']}",
        f"  Tier distribution:",
    ]
    for tier, count in report["tier_distribution"].items():
        pct = round(count / report["total_routes"] * 100, 1) if report["total_routes"] else 0
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        lines.append(f"    {tier:4s}  {bar}  {count:4d} ({pct:.1f}%)")

    lines.extend([
        f"",
        f"  Estimated cost:     ${report['estimated_total_cost']:.4f}",
        f"  Cost if all T3:     ${report['estimated_cost_if_t3_all']:.4f}",
        f"  Savings:            ${report['savings_usd']:.4f} ({report['savings_percentage']:.1f}%)",
        f"",
        f"  Recent routes (last 10):",
    ])

    for r in report.get("recent_routes", []):
        lines.append(f"    [{r['time']}] {r['tier']:4s} → {r['model']:30s} {r['message']}")

    return "\n".join(lines)
