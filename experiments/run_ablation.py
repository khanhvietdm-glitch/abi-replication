"""
Appendix C.4 -- Ablation studies.

For each variant in {ABI-NoAudit, ABI-NoCredit, ABI-NoDynamic, ABI-NoReliability}
the marginal effect on (i) false-trust rate, (ii) social welfare, (iii)
capital-exclusion rate is computed against the full ABI mechanism.

Outputs:
    results/ablation_summary.csv     -- per (variant, metric)
    figures/ablation_marginal.png    -- bar chart of marginal effects
"""
from __future__ import annotations

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from abi.config import SimulationParams
from abi.runner import run_compare


HERE = os.path.dirname(__file__)
RESULTS = os.path.abspath(os.path.join(HERE, "..", "results"))
FIG = os.path.abspath(os.path.join(HERE, "..", "figures"))
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

VARIANTS = ("abi", "abi_no_audit", "abi_no_credit", "abi_no_dynamic", "abi_no_reliability")

PRETTY = {
    "abi": "ABI (full)",
    "abi_no_audit": "ABI-NoAudit",
    "abi_no_credit": "ABI-NoCredit",
    "abi_no_dynamic": "ABI-NoDynamic",
    "abi_no_reliability": "ABI-NoReliability",
}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=15)
    ap.add_argument("--T", type=int, default=120)
    ap.add_argument("--quick", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.K = 4
        args.T = 60

    params = SimulationParams(T=args.T, K=args.K, N_p=120, N_v=30)
    seeds = list(range(1, args.K + 1))

    print(f"[run_ablation] {len(VARIANTS)} variants x {args.K} seeds x {args.T} epochs")
    multi = run_compare(VARIANTS, params, seeds, verbose=True)
    summary = multi.summary
    summary["variant"] = summary["mechanism"]
    summary.to_csv(os.path.join(RESULTS, "ablation_summary.csv"), index=False)

    # Marginal effect = variant - full ABI
    full = summary[summary["mechanism"] == "abi"].set_index("metric")["mean"]
    marg_rows = []
    for v in VARIANTS:
        if v == "abi":
            continue
        sub = summary[summary["mechanism"] == v].set_index("metric")["mean"]
        for metric in ["false_trust_rate", "social_welfare", "capital_exclusion_rate"]:
            marg_rows.append({
                "variant": v,
                "metric": metric,
                "marginal_effect_vs_full_ABI": float(sub[metric] - full[metric]),
                "variant_value": float(sub[metric]),
                "full_abi_value": float(full[metric]),
            })
    marg = pd.DataFrame(marg_rows)
    marg.to_csv(os.path.join(RESULTS, "ablation_marginal_effects.csv"), index=False)

    print("\n[run_ablation] mean per variant:")
    pivot = summary.pivot(index="metric", columns="mechanism", values="mean").round(4)
    print(pivot.to_string())

    # --- bar chart of marginal effects ----------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, metric in zip(axes, ["false_trust_rate", "social_welfare", "capital_exclusion_rate"]):
        sub = marg[marg["metric"] == metric]
        x = np.arange(len(sub))
        ax.bar(x, sub["marginal_effect_vs_full_ABI"], color="#c0504d", edgecolor="black", linewidth=0.5)
        ax.axhline(0, color="black", linewidth=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([PRETTY[v] for v in sub["variant"]], rotation=18, fontsize=8)
        ax.set_title(metric)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Appendix C.4 -- Marginal effect of removing each ABI component")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "ablation_marginal.png"), dpi=150)
    plt.close(fig)
    print(f"[run_ablation] wrote results to {RESULTS} and figure to {FIG}")


if __name__ == "__main__":
    main()
