"""Core classifier — determines task complexity and recommends model tier."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .features import TaskFeatures, extract_features
from .tiers import BuiltinTiers, TierConfig, TierLevel


@dataclass
class ClassificationResult:
    """Result of a single classification."""

    tier: TierLevel
    tier_config: TierConfig
    features: TaskFeatures = field(default_factory=lambda: TaskFeatures())
    reason: str = ""
    score: float = 0.0  # confidence 0.0–1.0


class Classifier:
    """Rule-based task complexity classifier.

    Analyzes user messages to determine the optimal model tier.
    Uses human-interpretable rules — no ML, no ONNX, no external deps.

    Usage::

        router = Classifier()
        result = router.classify("帮我写一个分布式爬虫")
        print(result.tier, result.tier_config.model)
        # => TierLevel.T3 claude-sonnet-4-20250514
    """

    def __init__(
        self,
        tiers: Optional[Dict[TierLevel, TierConfig]] = None,
        default_tier: TierLevel = TierLevel.T1,
    ):
        self.tiers = tiers or BuiltinTiers.qijing_default()
        self.default_tier = default_tier

    # ── Public API ─────────────────────────────────────────────────

    def classify(self, message: str) -> ClassificationResult:
        """Classify a user message and return the recommended tier."""
        features = extract_features(message)
        tier, score, reason = self._predict(features)
        return ClassificationResult(
            tier=tier,
            tier_config=self.tiers.get(tier, self.tiers[self.default_tier]),
            features=features,
            reason=reason,
            score=score,
        )

    def classify_bulk(self, messages: List[str]) -> List[ClassificationResult]:
        """Classify multiple messages in one call."""
        return [self.classify(m) for m in messages]

    # ── Internal prediction logic ──────────────────────────────────

    def _predict(self, features: TaskFeatures) -> Tuple[TierLevel, float, str]:
        """Run the rule-based prediction pipeline.

        Returns (tier, confidence_score, reason_string).
        """
        # Priority 1: Simple greetings → T0
        if features.is_simple_greeting:
            return (TierLevel.T0, 0.95, "Detected simple greeting/acknowledgment")

        # Priority 2: High-complexity keywords → T3
        if features.high_complexity_count >= 3:
            return (
                TierLevel.T3,
                0.85,
                f"Found {features.high_complexity_count} high-complexity keywords "
                f"(security/architecture/refactor)",
            )
        if features.high_complexity_count >= 1:
            # Elevated but not definitive
            pass

        # Priority 3: Large code blocks → check depth
        if features.has_code_block and features.code_block_count >= 2:
            # Multiple code blocks often means multi-file or complex logic
            return (
                TierLevel.T2,
                0.80,
                f"Multiple code blocks ({features.code_block_count}) detected",
            )

        if features.has_code_block and features.code_block_count == 1:
            # Single code block — could be simple or complex
            if features.high_complexity_count >= 1:
                return (
                    TierLevel.T2,
                    0.75,
                    f"Code block + high-complexity keywords suggest non-trivial task",
                )
            return (
                TierLevel.T1,
                0.65,
                "Single code block — moderate complexity",
            )

        # Priority 4: Code ratio without explicit blocks (inline code)
        if features.code_ratio > 0.3 and features.length > 500:
            return (
                TierLevel.T2,
                0.70,
                f"High code ratio ({features.code_ratio:.0%}) in long message",
            )

        # Priority 5: File paths + keywords
        if features.has_file_paths and features.keyword_count >= 2:
            if features.high_complexity_count >= 1:
                return (
                    TierLevel.T3,
                    0.75,
                    "File paths + complex keywords indicate system-level work",
                )
            return (
                TierLevel.T2,
                0.65,
                "File paths with task keywords suggest development work",
            )

        # Priority 6: Long message with analysis keywords
        if features.length > 2000:
            if features.keyword_count >= 3:
                return (
                    TierLevel.T3,
                    0.70,
                    f"Long message ({features.length} chars) with {features.keyword_count} "
                    f"task keywords — likely complex analysis",
                )
            return (
                TierLevel.T2,
                0.60,
                f"Long message ({features.length} chars) — allocate capable model",
            )

        # Priority 7: Medium-length with questions and keywords
        if 200 < features.length <= 2000 and features.keyword_count >= 2:
            return (
                TierLevel.T1,
                0.70,
                f"Moderate length with {features.keyword_count} keywords — "
                f"info retrieval or analysis",
            )

        # Priority 8: Very short messages with no signals
        if features.length <= 200:
            return (
                TierLevel.T0,
                0.80,
                f"Short message ({features.length} chars, minimal signals) — "
                f"use cheapest tier",
            )

        # Fallback: default tier
        return (
            self.default_tier,
            0.50,
            f"No strong signals — using default tier ({self.default_tier.name})",
        )

    # ── Estimation helpers ─────────────────────────────────────────

    def estimate_cost(self, message: str) -> dict:
        """Estimate token cost for each tier for a given message."""
        result = self.classify(message)
        # Rough token estimation
        input_tokens = len(message) * 0.35  # ~0.35 tokens per char for Chinese text
        output_tokens = 500  # conservative average
        total_tokens = input_tokens + output_tokens

        # Approximate cost per 1K tokens (USD)
        cost_per_1k = {
            "deepseek-v4-flash": 0.00015,
            "qwen3.7-max": 0.0015,
            "claude-sonnet-4-20250514": 0.003,
        }

        model_name = result.tier_config.model
        base_cost = cost_per_1k.get(model_name, 0.001)
        cost = (total_tokens / 1000) * base_cost

        # If using alternative tiers for comparison
        alternatives = {}
        for tier_level in TierLevel:
            if tier_level == result.tier:
                continue
            cfg = self.tiers.get(tier_level)
            if cfg is None:
                continue
            alt_cost = cost_per_1k.get(cfg.model, 0.001)
            alternatives[tier_level.name] = round(
                (total_tokens / 1000) * alt_cost, 6
            )

        return {
            "recommended_tier": result.tier.name,
            "recommended_model": model_name,
            "reason": result.reason,
            "estimated_cost_usd": round(cost, 6),
            "input_tokens_est": int(input_tokens),
            "output_tokens_est": output_tokens,
            "alternatives": alternatives,
            "savings_vs_t3": (
                round(
                    (1 - cost / alternatives.get("T3", cost)) * 100, 1
                )
                if "T3" in alternatives
                else 0
            ),
        }

    # ── Config management ─────────────────────────────────────────

    def to_yaml(self, path: str) -> None:
        """Export current tier config to YAML."""
        BuiltinTiers.to_yaml(self.tiers, path)

    @classmethod
    def from_yaml(cls, path: str) -> "Classifier":
        """Load classifier from YAML config."""
        tiers = BuiltinTiers.from_yaml(path)
        return cls(tiers=tiers)
