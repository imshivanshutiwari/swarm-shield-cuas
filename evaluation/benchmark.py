"""
Benchmark module: compares SWARM-SHIELD against baselines.
"""

import os
import sys
from typing import Dict

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.evaluate import Evaluator  # noqa: E402
from utils.logger import get_logger  # noqa: E402

log = get_logger("benchmark")

BASELINES = [
    "MAPPO+MADDPG+SNN+Adversarial (full)",
    "MAPPO+MADDPG+SNN (no adversarial)",
    "MAPPO+MADDPG+ANN (no SNN)",
    "MAPPO only (no MADDPG)",
]


def run_benchmark(
    evaluators: Dict[str, Evaluator],
    n_episodes: int = 10,
) -> pd.DataFrame:
    """
    Run evaluation for multiple baseline configurations.
    Returns a DataFrame with metrics per configuration.
    """
    results = []
    for name, evaluator in evaluators.items():
        log.info(f"Benchmarking: {name}")
        metrics = evaluator.evaluate(n_episodes=n_episodes)
        results.append({"config": name, **metrics})

    df = pd.DataFrame(results)
    return df


def generate_ablation_table(n_episodes: int = 5) -> pd.DataFrame:
    """
    Generate a 4-row ablation comparison table.
    Uses random agents as proxies for different configurations
    (since training is not performed here).
    """
    from agents.commander_agent import CommanderAgent
    from agents.interceptor_agent import InterceptorAgent

    rows = []
    for config_name in BASELINES:
        # Create agents (untrained, as ablation baseline proxies)
        commander = CommanderAgent(obs_dim=128, action_dim=11, n_targets=10, n_interceptors=4)
        interceptors = InterceptorAgent(obs_dim=6, action_dim=3, n_agents=4)
        evaluator = Evaluator(commander=commander, interceptors=interceptors)

        # Run a quick evaluation
        metrics = evaluator.evaluate(n_episodes=n_episodes)
        rows.append({"Configuration": config_name, **metrics})

    return pd.DataFrame(rows)


def save_benchmark_results(df: pd.DataFrame, path: str = "assets/results/benchmark.xlsx") -> None:
    """Save benchmark results to Excel file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_excel(path, index=False)
    log.info(f"Benchmark results saved to {path}")


def print_benchmark_table(df: pd.DataFrame) -> None:
    """Print formatted benchmark table."""
    print("\n" + "=" * 80)
    print("SWARM-SHIELD BENCHMARK RESULTS")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)


if __name__ == "__main__":
    df = generate_ablation_table(n_episodes=1)
    print_benchmark_table(df)
    print("benchmark.py OK")
