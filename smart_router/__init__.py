"""smart-router: Token-efficient AI task routing.

A lightweight classifier that analyzes task complexity and recommends
the optimal model/provider pairing — rules-based, zero ML dependency,
ready to use as a standalone CLI, a Python library, or a Hermes plugin.
"""

from __future__ import annotations

from .classifier import Classifier, ClassificationResult
from .tiers import TierConfig, TierLevel, BuiltinTiers
from .features import extract_features, TaskFeatures

__version__ = "0.1.0"
__all__ = [
    "Classifier",
    "ClassificationResult",
    "TierConfig",
    "TierLevel",
    "BuiltinTiers",
    "extract_features",
    "TaskFeatures",
]
