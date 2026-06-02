"""Tests for SmartRouter classifier and features."""

from __future__ import annotations

from smart_router import Classifier
from smart_router.tiers import TierLevel
from smart_router.features import extract_features


def test_simple_greeting():
    """Short greetings should route to T0."""
    c = Classifier()
    result = c.classify("你好")
    assert result.tier == TierLevel.T0, f"Expected T0, got {result.tier}"
    assert result.score >= 0.9


def test_acknowledgment():
    """Common acknowledgment words should be T0."""
    c = Classifier()
    for msg in ["好的", "继续", "好的，收到", "ok", "谢谢"]:
        result = c.classify(msg)
        assert result.tier == TierLevel.T0, f"Expected T0 for '{msg}', got {result.tier}"


def test_code_request():
    """Code request should route to T1+ (not T0)."""
    c = Classifier()
    result = c.classify("帮我写一个Python脚本来爬取网页数据\n\n```python\nimport requests\n\n```")
    assert result.tier != TierLevel.T0, f"Code request shouldn't be T0, got {result.tier}"
    assert result.tier in (TierLevel.T1, TierLevel.T2), f"Expected T1+, got {result.tier}"


def test_complex_architecture():
    """Architecture/security tasks should route to T3."""
    c = Classifier()
    result = c.classify(
        "设计一个分布式微服务架构，需要处理并发竞态、安全认证和数据库迁移"
    )
    assert result.tier == TierLevel.T3, f"Expected T3, got {result.tier}"


def test_simple_question():
    """Simple questions should route to T0/T1."""
    c = Classifier()
    result = c.classify("今天天气怎么样？")
    assert result.tier in (TierLevel.T0, TierLevel.T1), f"Expected T0/T1, got {result.tier}"


def test_file_paths_moderate():
    """File paths with security keywords should route to T2/T3."""
    c = Classifier()
    result = c.classify("在 src/utils/auth.py 中添加 JWT 验证逻辑")
    assert result.tier in (TierLevel.T2, TierLevel.T3), f"Expected T2/T3, got {result.tier}"


def test_long_message():
    """Very long messages should route to higher tiers."""
    c = Classifier()
    long_msg = "分析以下系统的性能瓶颈 " + "数据 " * 500  # ~1000 chars
    result = c.classify(long_msg)
    assert result.tier != TierLevel.T0, f"Long message shouldn't be T0, got {result.tier}"


def test_bulk_classification():
    """Bulk classification should work correctly."""
    c = Classifier()
    messages = [
        "你好",
        "帮我写一个分布式爬虫\n\n```python\nimport requests\n\n```",
        "今天天气怎么样",
        "设计安全架构",
    ]
    results = c.classify_bulk(messages)
    assert len(results) == 4
    assert results[0].tier == TierLevel.T0  # greeting
    assert results[2].tier in (TierLevel.T0, TierLevel.T1)  # simple question


def test_features_code_block():
    """Code block detection should work."""
    features = extract_features("Some text\n```python\nprint('hello')\n```\nmore text")
    assert features.has_code_block
    assert features.code_block_count == 1
    assert "python" in features.code_languages


def test_features_simple_greeting():
    """Greeting detection should work."""
    features = extract_features("你好")
    assert features.is_simple_greeting
    features2 = extract_features("帮我写代码")
    assert not features2.is_simple_greeting


def test_features_multi_line():
    """Multi-line detection should work."""
    features = extract_features("line1\nline2\nline3")
    assert features.is_multi_line
    assert features.line_count == 3


def test_features_high_complexity():
    """High-complexity keyword detection."""
    features = extract_features("security audit and architecture review for kubernetes migration")
    assert features.high_complexity_count >= 3


def test_cost_estimate():
    """Cost estimation should return sensible values."""
    c = Classifier()
    est = c.estimate_cost("简单的问候")
    assert est["recommended_tier"] == "T0"
    assert est["estimated_cost_usd"] > 0
    assert "alternatives" in est
    assert "T3" in est["alternatives"]
    assert est["savings_vs_t3"] > 0


def test_default_config():
    """Default tier config should cover all 4 levels."""
    c = Classifier()
    for level in TierLevel:
        assert level in c.tiers, f"Missing tier config for {level}"
        assert c.tiers[level].model, f"Empty model for {level}"


def test_config_round_trip(tmp_path):
    """YAML config export/import should round-trip."""
    from smart_router.tiers import BuiltinTiers

    path = tmp_path / "test-config.yaml"
    c1 = Classifier()
    c1.to_yaml(str(path))
    assert path.exists()

    c2 = Classifier.from_yaml(str(path))
    for level in TierLevel:
        assert c1.tiers[level].model == c2.tiers[level].model
        assert c1.tiers[level].provider == c2.tiers[level].provider
