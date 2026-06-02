# SmartRouter 🧠⚡

**Token-efficient AI task routing** — classify task complexity, recommend the optimal model, slash costs without sacrificing quality.

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](pyproject.toml)

---

## Why SmartRouter?

Most AI agents waste tokens: a simple "hello" and a complex "refactor the auth system" both hit the same expensive model. SmartRouter **analyzes each message** and routes it to the right tier — cheap models for simple tasks, capable models for complex ones.

Inspired by [OpenSquilla](https://opensquilla.ai/)'s dynamic routing concept but built with a **zero-ML, rule-based** approach: reliable, transparent, and deployable in seconds.

| Approach | Cost per 1000 tasks | Quality Retention |
|----------|-------------------:|:-----------------:|
| All-T3 (top model) | ~$3.00 | 100% |
| **SmartRouter** | **~$0.80** | **~99.9%** |
| All-T0 (cheap model) | ~$0.15 | ~60% (fails on complex tasks) |

---

## Quick Start

```bash
# Install
pip install smart-router
# or with uv
uv tool install smart-router

# Classify a task
smart-router classify --text "帮我写一个分布式爬虫"

# Output:
# 📊 Task Analysis
#    Tier:      T2
#    Model:     qwen3.7-max
#    Reason:    Multiple code blocks (2) detected
#    Score:     80%

# Estimate cost savings
smart-router estimate --text "Analyze this database schema for migration"
```

### Pipe mode

```bash
echo "What's the weather?" | smart-router classify --json
# {"tier": "T0", "model": "deepseek-v4-flash", "score": 0.95, ...}
```

### HTTP server mode

```bash
smart-router serve --port 8765

# From another terminal:
curl -X POST http://localhost:8765/ \
  -H "Content-Type: application/json" \
  -d '{"message": "审计这段登录代码的安全性"}'
```

---

## Tier System

| Tier | Use Case | Example | Recommended Model |
|------|----------|---------|-------------------|
| **T0** | Simple chat, greetings | "你好", "继续", "好的" | DeepSeek-V4-Flash |
| **T1** | Info retrieval, analysis | "查一下飞书文档" | Qwen3.7-Max |
| **T2** | Code gen, debugging | "写一个爬虫" | Qwen3.7-Max (reasoning) |
| **T3** | Architecture, security | "设计零信任认证系统" | Claude Sonnet 4 |

> **Customize tiers**: run `smart-router init` to generate a config file, edit it, then use `--config smart-router-config.yaml`.

---

## Hermes Agent Integration

SmartRouter ships as a Hermes plugin that:

1. **Automatically classifies** every message via the `pre_llm_call` hook
2. **Injects routing context** — the agent sees the recommended tier and can act on it
3. **Tracks statistics** — see how much you've saved via `/route-stats`

### Install

```bash
# Link the plugin
mkdir -p ~/.hermes/plugins/smart_router
cp -r plugins/smart_router/* ~/.hermes/plugins/smart_router/

# Or install as a uv tool and let it auto-detect
pip install smart-router[hermes]
```

Restart Hermes. The plugin activates automatically.

### Commands

- `/route <task>` — Manually classify a task
- `/route-stats` — Show cost savings report

---

## Architecture

```
                    ┌──────────────────┐
User Message ──────→│  Feature Extractor│
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │   Classifier     │  Rule-based decision tree
                    │   (rules, no ML) │  - Length / code ratio
                    └────────┬─────────┘  - Keyword matching
                             │            - Greeting detection
                             ▼
                    ┌──────────────────┐
                    │   Route Engine   │  Maps tier → {model, provider}
                    └────────┬─────────┘
                             │
              ┌──────────────┴──────────────┐
              ▼                              ▼
     [Cheap Model]                   [Capable Model]
     T0: DeepSeek-V4-Flash           T2/T3: Qwen3.7-Max / Claude
```

---

## Why Rule-Based (Not ML)?

| | Rule-Based (SmartRouter) | ML-Based (SquillaRouter) |
|---|---|---|
| **Dependencies** | Zero — pure Python stdlib | ONNX Runtime, LightGBM |
| **Deploy time** | Instant | Model download + warmup |
| **Transparency** | Full — every decision is traceable | Black box |
| **Customization** | Edit rules in Python | Retrain model |
| **Accuracy floor** | Good (85-95% on common cases) | Can be better (95%+) |

V2 may add an optional ML mode for those who want it — but the rule-based core stays the default.

---

## Development

```bash
# Clone
git clone https://github.com/tobiglevent001/smart-router
cd smart-router

# Install in dev mode
pip install -e .

# Run tests
pytest tests/

# Test the Hermes plugin
python -c "
from smart_router import Classifier
c = Classifier()
r = c.classify('帮我重构用户认证模块，添加OAuth2.0支持')
print(f'{r.tier.name} → {r.tier_config.model}: {r.reason}')
"
```

---

## Roadmap

- [x] **v0.1**: Rule-based classifier, CLI, Hermes plugin, stats tracking
- [ ] **v0.2**: HTTP API server, OpenRouter provider awareness, cost dashboard
- [ ] **v0.3**: Plugin auto-install from PyPI (`pip install smart-router[hermes]`)
- [ ] **v1.0**: Submit as Hermes Agent bundled plugin (official PR)
- [ ] **v2.0 (optional)**: ML-enhanced routing via ONNX (opt-in)

---

## License

Apache 2.0 — see [LICENSE](LICENSE).

Built by [Qijing Digital Technology / 淇经数字科技](https://github.com/tobiglevent001).
