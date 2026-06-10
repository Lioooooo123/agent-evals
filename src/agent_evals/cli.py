"""Command line interface for agent-evals."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agent-evals")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run eval cases.")
    run_parser.add_argument(
        "--dataset",
        help="Path to an EvalCase JSONL dataset. The runner is not implemented in M1.",
    )
    run_parser.set_defaults(func=_run_not_implemented)
    return parser


def _run_not_implemented(args: argparse.Namespace) -> int:
    raise SystemExit("agent-evals run is not implemented in M1 yet.")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 0
    result = args.func(args)
    return int(result or 0)


if __name__ == "__main__":
    raise SystemExit(main())
