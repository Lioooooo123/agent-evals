"""Command line interface for agent-evals."""

from __future__ import annotations

import argparse
from pathlib import Path

from agent_evals.comparisons import CompareOptions, compare_runs
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
        choices=["mock", "pi"],
        help="Adapter to use.",
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

    compare_parser = subparsers.add_parser("compare", help="Compare two eval runs.")
    compare_parser.add_argument(
        "--baseline",
        required=True,
        help="Path to the baseline eval_results.jsonl file.",
    )
    compare_parser.add_argument(
        "--candidate",
        required=True,
        help="Path to the candidate eval_results.jsonl file.",
    )
    compare_parser.add_argument(
        "--out",
        help="Directory where compare_report.md and compare_results.json should be written.",
    )
    compare_parser.add_argument(
        "--gates",
        default="configs/gates.yaml",
        help="Path to gate config YAML. Use 'none' to disable config gates.",
    )
    compare_parser.add_argument(
        "--fail-if-drop",
        action="append",
        default=[],
        metavar="METRIC=THRESHOLD",
        help="Fail when candidate drops from baseline by more than the threshold.",
    )
    compare_parser.set_defaults(func=_compare)
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


def _compare(args: argparse.Namespace) -> int:
    summary = compare_runs(
        CompareOptions(
            baseline_path=Path(args.baseline),
            candidate_path=Path(args.candidate),
            out_dir=Path(args.out) if args.out else None,
            gates_path=None if args.gates == "none" else Path(args.gates),
            fail_if_drop=_parse_fail_if_drop(args.fail_if_drop),
        )
    )
    out_dir = Path(args.out) if args.out else Path(args.candidate).parent
    print(f"Compare report written to {out_dir / 'compare_report.md'}")
    print(f"Compare JSON written to {out_dir / 'compare_results.json'}")
    if summary.gate_triggered:
        print("Gate failed:")
        for failure in summary.gate_failures:
            print(f"- {failure.reason}")
        return 1
    print("Gate passed.")
    return 0


def _parse_fail_if_drop(values: list[str]) -> dict[str, float]:
    parsed: dict[str, float] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--fail-if-drop must use METRIC=THRESHOLD, got {value!r}")
        metric, threshold = value.split("=", 1)
        if not metric:
            raise ValueError("--fail-if-drop metric cannot be empty")
        parsed[metric] = float(threshold)
    return parsed


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
