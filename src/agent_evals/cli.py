"""Command line interface for agent-evals."""

from __future__ import annotations

import argparse
from pathlib import Path

from agent_evals.runners import RunOptions, run_eval


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agent-evals")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run eval cases.")
    run_parser.add_argument(
        "--cases",
        "--dataset",
        dest="cases",
        required=True,
        help="Path to an EvalCase JSONL dataset.",
    )
    run_parser.add_argument(
        "--out",
        required=True,
        help="Directory where the run directory should be written.",
    )
    run_parser.add_argument(
        "--adapter",
        default="mock",
        choices=["mock"],
        help="Adapter to use. Only mock is implemented in this phase.",
    )
    run_parser.add_argument(
        "--weights",
        default="configs/weights.yaml",
        help="Path to aggregate scorer weights YAML.",
    )
    run_parser.add_argument(
        "--rubric",
        default="configs/judge_rubric.yaml",
        help="Path to judge rubric YAML. Use 'none' to disable judge scoring.",
    )
    run_parser.add_argument(
        "--run-id",
        help="Optional run id. Defaults to a UTC timestamp.",
    )
    run_parser.set_defaults(func=_run)
    return parser


def _run(args: argparse.Namespace) -> int:
    summary = run_eval(
        RunOptions(
            cases_path=Path(args.cases),
            out_dir=Path(args.out),
            adapter_name=args.adapter,
            weights_path=Path(args.weights),
            rubric_path=None if args.rubric == "none" else Path(args.rubric),
            run_id=args.run_id,
        )
    )
    passed = sum(1 for record in summary.records if record.passed)
    total = len(summary.records)
    print(f"Run written to {summary.out_dir}")
    print(f"Pass rate: {passed}/{total}")
    return 0


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
