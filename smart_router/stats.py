"""Cost tracking and performance statistics for SmartRouter.

Tracks routing decisions and calculates cost savings vs.
always-using-top-tier baselines.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .tiers import TierLevel


@dataclass
class RoutingRecord:
    """A single routing decision record."""

    timestamp: float
    message_preview: str
    tier: int
    model: str
    provider: str
    estimated_cost: float
    features_summary: str
    score: float


@dataclass
class RouterStats:
    """Aggregated routing statistics."""

    total_routes: int = 0
    tier_counts: Dict[str, int] = field(default_factory=lambda: {t.name: 0 for t in TierLevel})
    total_estimated_cost: float = 0.0
    records: List[RoutingRecord] = field(default_factory=list)


class StatsCollector:
    """Collects and reports routing statistics.

    Stores data in memory and optionally persists to JSON for
    dashboard/report generation.
    """

    def __init__(self, persist_path: Optional[str] = None):
        self.stats = RouterStats()
        self.persist_path = persist_path or os.environ.get(
            "SMART_ROUTER_STATS_PATH",
            os.path.expanduser("~/.hermes/smart-router-stats.json"),
        )

    def record(self, result, features_summary: str = "") -> None:
        """Record a routing decision."""
        self.stats.total_routes += 1
        tier_name = result.tier.name
        self.stats.tier_counts[tier_name] = (
            self.stats.tier_counts.get(tier_name, 0) + 1
        )

        # Rough cost estimate
        cost_map = {
            "deepseek-v4-flash": 0.00015,
            "qwen3.7-max": 0.0015,
            "claude-sonnet-4-20250514": 0.003,
        }
        cost = cost_map.get(result.tier_config.model, 0.001)
        self.stats.total_estimated_cost += cost

        record = RoutingRecord(
            timestamp=time.time(),
            message_preview=result.features.text[:80],
            tier=result.tier.value,
            model=result.tier_config.model,
            provider=result.tier_config.provider,
            estimated_cost=cost,
            features_summary=features_summary or result.reason,
            score=result.score,
        )
        self.stats.records.append(record)
        self._persist()

    def get_report(self) -> dict:
        """Generate a human-readable stats report."""
        s = self.stats
        if s.total_routes == 0:
            return {"status": "no_data", "message": "No routing data collected yet."}

        # Calculate what cost would have been if all went to T3
        total_if_t3 = s.total_routes * 0.003  # rough T3 cost per call
        savings = total_if_t3 - s.total_estimated_cost
        savings_pct = (savings / total_if_t3 * 100) if total_if_t3 > 0 else 0

        return {
            "status": "active",
            "total_routes": s.total_routes,
            "tier_distribution": dict(sorted(s.tier_counts.items())),
            "estimated_total_cost": round(s.total_estimated_cost, 4),
            "estimated_cost_if_t3_all": round(total_if_t3, 4),
            "savings_usd": round(savings, 4),
            "savings_percentage": round(savings_pct, 1),
            "recent_routes": [
                {
                    "time": time.strftime(
                        "%Y-%m-%d %H:%M:%S", time.localtime(r.timestamp)
                    ),
                    "tier": TierLevel(r.tier).name,
                    "model": r.model,
                    "message": r.message_preview,
                }
                for r in s.records[-10:]
            ][::-1],
        }

    def _persist(self) -> None:
        """Persist stats to disk if path is set."""
        if not self.persist_path:
            return
        try:
            os.makedirs(os.path.dirname(self.persist_path) or ".", exist_ok=True)
            with open(self.persist_path, "w") as f:
                json.dump(
                    {
                        "total_routes": self.stats.total_routes,
                        "tier_counts": self.stats.tier_counts,
                        "total_estimated_cost": self.stats.total_estimated_cost,
                        "recent_records": [
                            {
                                "timestamp": r.timestamp,
                                "tier": r.tier,
                                "model": r.model,
                                "message": r.message_preview,
                            }
                            for r in self.stats.records[-100:]
                        ],
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
        except OSError:
            pass  # non-critical, don't crash
