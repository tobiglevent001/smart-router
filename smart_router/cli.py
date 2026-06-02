"""CLI interface for SmartRouter — classify tasks from the command line."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Dict, Optional

from .classifier import Classifier, ClassificationResult
from .tiers import BuiltinTiers, TierLevel


def _format_result(result: ClassificationResult, verbose: bool) -> str:
    """Format a classification result for display."""
    lines = [
        f"📊 Task Analysis",
        f"   Tier:      {result.tier.name}",
        f"   Model:     {result.tier_config.model}",
        f"   Provider:  {result.tier_config.provider or '(default)'}",
    ]
    if result.tier_config.reasoning_effort:
        lines.append(
            f"   Reasoning: {result.tier_config.reasoning_effort}"
        )
    lines.append(f"   Reason:    {result.reason}")
    lines.append(f"   Score:     {result.score:.0%}")

    if verbose:
        lines.append(f"\n── Features ──")
        for k, v in result.features.to_dict().items():
            if k == "text":
                continue
            if isinstance(v, bool) and v:
                lines.append(f"   ✓ {k}")
            elif isinstance(v, (int, float)) and v > 0:
                lines.append(f"   {k}: {v}")

    return "\n".join(lines)


def cmd_classify(args: argparse.Namespace) -> int:
    """Classify a single message."""
    classifier = _load_classifier(getattr(args, "config", None))
    if args.text:
        message = args.text
    elif not sys.stdin.isatty():
        message = sys.stdin.read().strip()
    else:
        print("Error: provide --text or pipe input", file=sys.stderr)
        return 1

    result = classifier.classify(message)

    if args.json:
        print(json.dumps({
            "tier": result.tier.name,
            "model": result.tier_config.model,
            "provider": result.tier_config.provider,
            "reasoning_effort": result.tier_config.reasoning_effort,
            "reason": result.reason,
            "score": round(result.score, 3),
            "features": result.features.to_dict(),
        }, ensure_ascii=False, indent=2))
    else:
        print(_format_result(result, args.verbose))

    return 0


def cmd_estimate(args: argparse.Namespace) -> int:
    """Estimate cost for a message across all tiers."""
    classifier = _load_classifier(getattr(args, "config", None))
    if args.text:
        message = args.text
    elif not sys.stdin.isatty():
        message = sys.stdin.read().strip()
    else:
        print("Error: provide --text or pipe input", file=sys.stderr)
        return 1

    estimate = classifier.estimate_cost(message)
    print(json.dumps(estimate, ensure_ascii=False, indent=2))
    return 0


def cmd_init_config(args: argparse.Namespace) -> int:
    """Generate a default tier config YAML."""
    path = args.path or "smart-router-config.yaml"
    BuiltinTiers.to_yaml(BuiltinTiers.qijing_default(), path)
    print(f"✅ Default config written to: {path}")
    return 0


def cmd_bulk(args: argparse.Namespace) -> int:
    """Classify messages from a file (one per line) or stdin."""
    classifier = _load_classifier(getattr(args, "config", None))
    if args.file:
        with open(args.file) as f:
            messages = [line.strip() for line in f if line.strip()]
    elif not sys.stdin.isatty():
        messages = [line.strip() for line in sys.stdin if line.strip()]
    else:
        print("Error: provide --file or pipe input", file=sys.stderr)
        return 1

    results = classifier.classify_bulk(messages)
    for msg, res in zip(messages, results):
        print(f"{msg[:60]:60s} → {res.tier.name:4s} ({res.tier_config.model})")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    """Run as a simple HTTP server for classification."""
    classifier = _load_classifier(getattr(args, "config", None))
    host = args.host
    port = args.port

    try:
        from http.server import BaseHTTPRequestHandler, HTTPServer
    except ImportError:
        print("Error: http.server not available", file=sys.stderr)
        return 1

    class RouterHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length).decode("utf-8")
            try:
                data = json.loads(body)
                message = data.get("message", "")
            except (json.JSONDecodeError, KeyError):
                message = body

            result = classifier.classify(message)
            response = json.dumps({
                "tier": result.tier.name,
                "model": result.tier_config.model,
                "provider": result.tier_config.provider,
                "reasoning_effort": result.tier_config.reasoning_effort,
                "reason": result.reason,
                "score": round(result.score, 3),
            }, ensure_ascii=False)
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(response.encode("utf-8"))

        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"SmartRouter server running. POST / with {\"message\": \"...\"}")

        def log_message(self, fmt, *args):
            pass  # quiet

    server = HTTPServer((host, port), RouterHandler)
    print(f"🚀 SmartRouter server listening on http://{host}:{port}")
    print(f"   POST / with {{\"message\": \"your task\"}}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()
    return 0


def _load_classifier(config_path: Optional[str]) -> Classifier:
    """Load classifier, optionally from a YAML config file."""
    if config_path:
        return Classifier.from_yaml(config_path)
    return Classifier()


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="smart-router",
        description="Token-efficient AI task routing — classify complexity and recommend optimal model",
    )

    sub = parser.add_subparsers(dest="command", help="Subcommand")

    p_classify = sub.add_parser("classify", help="Classify a single message")
    p_classify.add_argument("--text", "-t", help="Message text to classify")
    p_classify.add_argument("--json", "-j", action="store_true", help="Output as JSON")
    p_classify.add_argument("--verbose", "-v", action="store_true", help="Show feature details")
    p_classify.add_argument("--config", "-c", help="Path to tier config YAML")
    p_classify.set_defaults(func=cmd_classify)

    p_estimate = sub.add_parser("estimate", help="Estimate cost across tiers")
    p_estimate.add_argument("--text", "-t", help="Message text to estimate")
    p_estimate.add_argument("--config", "-c", help="Path to tier config YAML")
    p_estimate.set_defaults(func=cmd_estimate)

    p_init = sub.add_parser("init", help="Generate default tier config YAML")
    p_init.add_argument("--path", "-p", default="smart-router-config.yaml",
                        help="Output path for config")
    p_init.set_defaults(func=cmd_init_config)

    p_bulk = sub.add_parser("bulk", help="Classify messages from file")
    p_bulk.add_argument("--file", "-f", help="File with messages (one per line)")
    p_bulk.add_argument("--config", "-c", help="Path to tier config YAML")
    p_bulk.set_defaults(func=cmd_bulk)

    p_serve = sub.add_parser("serve", help="Start HTTP classification server")
    p_serve.add_argument("--host", default="127.0.0.1", help="Bind address")
    p_serve.add_argument("--port", type=int, default=8765, help="Bind port")
    p_serve.add_argument("--config", "-c", help="Path to tier config YAML")
    p_serve.set_defaults(func=cmd_serve)

    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
