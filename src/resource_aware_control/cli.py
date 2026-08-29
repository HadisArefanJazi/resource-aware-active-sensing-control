import argparse
import json
from pathlib import Path

from resource_aware_control.baselines import RLController
from resource_aware_control.config import TrainingConfig, evaluation_scenarios
from resource_aware_control.evaluation import run_episode
from resource_aware_control.experiment import format_results, run_experiment, write_results
from resource_aware_control.networks import PolicyNetwork


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="resource-control",
        description="Train and evaluate resource-aware active sensing policies.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    experiment = commands.add_parser("experiment", help="train policies and compare baselines")
    experiment.add_argument("--episodes", type=int, default=1_500)
    experiment.add_argument("--eval-episodes", type=int, default=100)
    experiment.add_argument("--seed", type=int, default=7)
    experiment.add_argument("--output", type=Path, default=Path("results/metrics.json"))
    experiment.add_argument("--artifacts", type=Path, default=Path("artifacts"))

    demo = commands.add_parser("demo", help="evaluate one saved constrained policy episode")
    demo.add_argument("--policy", type=Path, default=Path("artifacts/constrained_policy.pt"))
    demo.add_argument("--scenario", default="nominal")
    demo.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "experiment":
        if args.episodes <= 0 or args.eval_episodes <= 0:
            raise SystemExit("Episode counts must be positive.")
        results = run_experiment(
            TrainingConfig(episodes=args.episodes, seed=args.seed),
            evaluation_episodes=args.eval_episodes,
            artifact_directory=args.artifacts,
        )
        write_results(results, args.output)
        print(format_results(results))
        print(f"\nSaved metrics to {args.output} and policies to {args.artifacts}")
        return 0

    scenarios = {}
    for scenario in evaluation_scenarios():
        scenarios[scenario.name] = scenario

    if args.scenario not in scenarios:
        choices = ", ".join(sorted(scenarios))
        raise SystemExit(f"Unknown scenario. Choose one of: {choices}")

    policy = PolicyNetwork.load(args.policy)
    metrics = run_episode(RLController(policy), scenarios[args.scenario], args.seed)
    print(json.dumps(metrics.__dict__, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
