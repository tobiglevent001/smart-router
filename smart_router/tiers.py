"""Model tier definitions for SmartRouter.

Each tier maps a complexity level to a model/provider pairing.
Tiers are user-configurable — ship with sensible defaults for
common setups.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, Optional


class TierLevel(IntEnum):
    """Task complexity tiers — higher = more capable (and expensive)."""

    T0 = 0  # Simple chat, greetings, confirmations
    T1 = 1  # Information retrieval, analysis
    T2 = 2  # Code generation, debugging, reasoning
    T3 = 3  # Complex architecture, security, refactoring


@dataclass
class TierConfig:
    """Configuration for a single model tier."""

    level: TierLevel
    model: str
    provider: str = ""
    base_url: str = ""
    reasoning_effort: str = ""
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "level": self.level.value,
            "model": self.model,
            "provider": self.provider,
            "base_url": self.base_url,
            "reasoning_effort": self.reasoning_effort,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TierConfig":
        return cls(
            level=TierLevel(d.get("level", 0)),
            model=d.get("model", ""),
            provider=d.get("provider", ""),
            base_url=d.get("base_url", ""),
            reasoning_effort=d.get("reasoning_effort", ""),
            description=d.get("description", ""),
        )


# ── Built-in tier presets ─────────────────────────────────────────

class BuiltinTiers:
    """Factory for common tier presets."""

    @staticmethod
    def qijing_default() -> Dict[TierLevel, TierConfig]:
        """淇经数科默认路由配置 — 日常对话走 DeepSeek，开发走 Qwen3.7-Max。"""
        return {
            TierLevel.T0: TierConfig(
                level=TierLevel.T0,
                model="deepseek-v4-flash",
                provider="deepseek",
                reasoning_effort="",
                description="Simple chat, greetings, confirmations — cheap & fast",
            ),
            TierLevel.T1: TierConfig(
                level=TierLevel.T1,
                model="qwen3.7-max",
                provider="ali-token-plan",
                reasoning_effort="",
                description="Information retrieval, analysis — balanced",
            ),
            TierLevel.T2: TierConfig(
                level=TierLevel.T2,
                model="qwen3.7-max",
                provider="ali-token-plan",
                reasoning_effort="high",
                description="Code generation, debugging — capable with reasoning",
            ),
            TierLevel.T3: TierConfig(
                level=TierLevel.T3,
                model="claude-sonnet-4-20250514",
                provider="ali-token-plan",
                reasoning_effort="high",
                description="Complex architecture, security — top capability",
            ),
        }

    @staticmethod
    def deepseek_only() -> Dict[TierLevel, TierConfig]:
        """All tasks use DeepSeek (single-model setup)."""
        cfg = TierConfig(
            level=TierLevel.T0,
            model="deepseek-v4-flash",
            provider="deepseek",
            description="Single model — all tasks use DeepSeek-V4-Flash",
        )
        return {t: cfg for t in TierLevel}

    @staticmethod
    def from_yaml(path: str) -> Dict[TierLevel, TierConfig]:
        """Load tier config from a YAML file."""
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)

        tiers: Dict[TierLevel, TierConfig] = {}
        for entry in data.get("tiers", []):
            level = TierLevel(entry["level"])
            tiers[level] = TierConfig.from_dict(entry)
        return tiers

    @staticmethod
    def to_yaml(tiers: Dict[TierLevel, TierConfig], path: str) -> None:
        """Dump tier config to a YAML file."""
        import yaml

        data = {
            "tiers": [
                {
                    "level": level.value,
                    "model": cfg.model,
                    "provider": cfg.provider,
                    "base_url": cfg.base_url,
                    "reasoning_effort": cfg.reasoning_effort,
                    "description": cfg.description,
                }
                for level, cfg in sorted(tiers.items())
            ]
        }
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
