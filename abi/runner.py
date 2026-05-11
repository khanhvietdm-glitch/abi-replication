"""
Cross-mechanism / cross-replication runner.

Implements the paired comparison from Appendix C.3: every mechanism sees
the same population draws within a given replication seed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List
import numpy as np
import pandas as pd

from .config import SimulationParams
from .mechanisms import build_mechanism
from .simulation import Simulator, ReplicationResult


@dataclass
class MultiResult:
    """Aggregated results across replications, per mechanism."""
    summary: pd.DataFrame                 # mean +/- ci per (mechanism, metric)
    per_replication: pd.DataFrame         # one row per (mechanism, seed)
    per_epoch_long: pd.DataFrame          # all per-epoch rows; useful for figures


# ---------------------------------------------------------------------------
def _ci95(values: np.ndarray) -> float:
    """Half-width of a 95% normal-approximation confidence interval."""
    n = values.size
    if n <= 1:
        return 0.0
    return float(1.96 * values.std(ddof=1) / np.sqrt(n))


def run_compare(
    mechanisms: Iterable[str],
    params: SimulationParams,
    seeds: Iterable[int],
    total_reward_budget: float = 100.0,
    verbose: bool = True,
) -> MultiResult:
    """Run each mechanism over each master seed and aggregate."""

    per_rep_rows: List[dict] = []
    per_epoch_rows: List[pd.DataFrame] = []

    metric_names = [
        "false_trust_rate",
        "social_welfare",
        "high_type_participation",
        "reward_gini",
        "collusion_success_rate",
        "capital_exclusion_rate",
    ]

    seeds = list(seeds)

    for mech_name in mechanisms:
        if verbose:
            print(f"[runner] mechanism = {mech_name}  ({len(seeds)} replications)")
        for k, seed in enumerate(seeds):
            mech = build_mechanism(mech_name, params)
            sim = Simulator(params, mech, seed, total_reward_budget=total_reward_budget)
            result: ReplicationResult = sim.run()
            # Drop burn-in
            df = result.per_epoch[result.per_epoch["epoch"] >= params.burn_in]
            row = {"mechanism": mech_name, "seed": seed, "final_S_star": result.final_S_star}
            for m in metric_names:
                row[m] = float(df[m].mean())
            per_rep_rows.append(row)
            df = df.assign(mechanism=mech_name, seed=seed)
            per_epoch_rows.append(df)
            if verbose and (k + 1) % max(1, len(seeds) // 5) == 0:
                print(f"  seed {seed}: welfare = {row['social_welfare']:.2f}")

    per_rep = pd.DataFrame(per_rep_rows)
    per_epoch = pd.concat(per_epoch_rows, ignore_index=True)

    summary_rows = []
    for mech_name, g in per_rep.groupby("mechanism"):
        for m in metric_names:
            vals = g[m].to_numpy()
            summary_rows.append({
                "mechanism": mech_name,
                "metric": m,
                "mean": float(vals.mean()),
                "ci95": _ci95(vals),
                "n_replications": vals.size,
            })
    summary = pd.DataFrame(summary_rows)

    return MultiResult(summary=summary, per_replication=per_rep, per_epoch_long=per_epoch)
