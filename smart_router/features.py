"""Feature extraction from user messages.

Extracts structured features used by the classifier to determine
task complexity. All rule-based — no ML dependencies required.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional


# ── Language / code detection ─────────────────────────────────────

_CODE_BLOCK_OPEN = re.compile(r"(?:^|\n)```(\w*)", re.MULTILINE)
_SHELL_PATTERN = re.compile(
    r"\b(?:bash|sh|zsh|powershell|cmd|terminal|console)\b", re.IGNORECASE
)
_FILE_PATH_PATTERN = re.compile(
    r"(?:~/|/|[a-zA-Z]:[/\\]|\.\.?/)[\w.\-/\\]+\.[a-z]{1,4}(?::\d+)?", re.IGNORECASE
)
_KEYWORD_PATTERN = re.compile(
    r"(?:"
    r"重构|优化|审计|安全|迁移|部署|设计|架构|"
    r"refactor|optimize|audit|security|migrate|deploy|design|architecture|"
    r"并发|竞态|deadlock|race|thread|"
    r"concurrent|deadlock|race\s*condition|thread|"
    r"加密|auth|oauth|jwt|token|certificate|"
    r"encrypt|auth|oauth|jwt|token|cert|"
    r"数据库|sql|query|schema|migration|"
    r"database|sql|query|schema|migration"
    r")",
    re.IGNORECASE,
)

_HIGH_COMPLEXITY_KEYWORDS = re.compile(
    r"(?:"
    r"security|audit|审计|安全|penetration|渗透|"
    r"architecture|架构|distributed|分布式|"
    r"refactor|重构|migration|迁移|rollback|回滚|"
    r"crypto|cryptography|加密|encrypt|decrypt|"
    r"thread(?:\s*safe)?|并发|竞态|deadlock|"
    r"zero\s*trust|零信任|auth[0-9]|oauth|jwt|"
    r"kubernetes|k8s|docker|container|容器|"
    r"performance|性能|optimization|优化|瓶颈|bottleneck"
    r")",
    re.IGNORECASE,
)

_SIMPLE_GREETINGS = re.compile(
    r"^(?:"
    r"你好|嗨|hello|hi|hey|早安|午安|晚安|"
    r"谢谢|thank|thanks|好的|ok|好的|继续|"
    r"嗯|是|对|好|行|可以|再?见|bye"
    r")\s*[.。!！?？]?\s*$",
    re.IGNORECASE,
)


@dataclass
class TaskFeatures:
    """Structured features extracted from a user message."""

    text: str = ""
    length: int = 0
    has_code_block: bool = False
    code_block_count: int = 0
    code_languages: List[str] = field(default_factory=list)
    code_ratio: float = 0.0  # fraction of message that is code
    has_shell_commands: bool = False
    has_file_paths: bool = False
    keyword_count: int = 0
    high_complexity_count: int = 0
    is_simple_greeting: bool = False
    has_questions: bool = False
    question_count: int = 0
    is_multi_line: bool = False
    line_count: int = 0
    contains_url: bool = False
    contains_json: bool = False

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


def extract_features(message: str) -> TaskFeatures:
    """Extract feature vector from a user message."""
    features = TaskFeatures(text=message[:200], length=len(message))

    # ── Code blocks ──
    # Count opening ``` lines (with optional language). Exclude bare ``` closings.
    blocks = _CODE_BLOCK_OPEN.findall(message)
    opening_blocks = [b for b in blocks if b]  # only count blocks with language tags
    features.has_code_block = len(blocks) > 0
    features.code_block_count = len(opening_blocks) if opening_blocks else (1 if blocks else 0)
    features.code_languages = [b for b in blocks if b]

    # Estimate code ratio: lines inside backtick blocks
    code_lines = 0
    total_lines = message.count("\n") + 1
    in_code = False
    for line in message.split("\n"):
        if line.strip().startswith("```"):
            in_code = not in_code
        elif in_code:
            code_lines += 1
    features.code_ratio = code_lines / max(total_lines, 1)

    # ── Shell commands ──
    features.has_shell_commands = bool(_SHELL_PATTERN.search(message))

    # ── File paths ──
    features.has_file_paths = bool(_FILE_PATH_PATTERN.search(message))

    # ── Keywords ──
    features.keyword_count = len(_KEYWORD_PATTERN.findall(message))
    features.high_complexity_count = len(_HIGH_COMPLEXITY_KEYWORDS.findall(message))

    # ── Simple greetings ──
    features.is_simple_greeting = bool(_SIMPLE_GREETINGS.match(message.strip()))

    # ── Questions ──
    features.question_count = message.count("?") + message.count("？")
    features.has_questions = features.question_count > 0

    # ── Multi-line ──
    features.is_multi_line = "\n" in message.strip()
    features.line_count = total_lines

    # ── URLs ──
    features.contains_url = bool(
        re.search(r"https?://[^\s,，。；;'\"<>()（）]+", message)
    )

    # ── JSON ──
    features.contains_json = bool(
        re.search(r'\{[\s\S]*"[^"]+"\s*:', message)
    )

    return features
