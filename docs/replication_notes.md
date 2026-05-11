# Replication notes

## What this package is

A self-contained replication artefact for the simulations of the ABI
paper. It is *not* a production trust-protocol implementation — there
is no on-chain logic, no token bridge, no smart contracts. The four
mechanisms are stripped to the minimum needed to reproduce the
qualitative claims in Section 7.5 and Appendix C.

## What this package is **not**

* Not bit-level identical to Table 7. The paper uses `K = 50`
  replications, `T = 200` epochs, `N_p = 200`, `N_v = 40`, `N_u = 5000`.
  Default scripts use smaller settings for fast turnaround; pass `--K
  50 --T 200` (and `--full` for sensitivity sweeps) to reproduce the
  paper-scale grid.
* Not a Bittensor emulator. The "Yuma-like" baseline implements the
  stylized stake-weighted ranking with top-quantile clipping that the
  paper describes — not the full Yuma Consensus.
* Not a token-price model. The speculative-flow process is calibrated
  to the variance-decomposition assumptions of Theorem 5, not to any
  specific market microstructure.

## Calibration choices that influence the absolute numbers

| Choice | Where | Effect |
| --- | --- | --- |
| `welfare_quality_value = 1.0`, `welfare_harm_cost = 3.0` | `abi/config.SimulationParams` | sets the *scale* of social welfare; the paper normalizes its budget so the relative gap is identifiable rather than the absolute level |
| `audit_capacity_frac = 0.10` | `abi/config.SimulationParams` | per-epoch audit budget; sensitivity sweep in `run_sensitivity.py` confirms ABI ≥ Yuma at all values |
| `target_audit_gain_share = 0.30` | `abi/config.SimulationParams` | how aggressively the ABI risk-targeted audit (Theorem 3) prioritizes high-`G_i` producers |
| `slash_rate_purestake = 0.30` | `abi/config.SimulationParams` | inside Theorem 2's feasible interval `[0.078, 0.313]`; shared across Pure-Stake, Yuma, and ABI base rate |
| `psi_credit, f_max_credit` | `abi/config.SimulationParams` | governs how quickly Proposition 2's quality credit boosts effective stake |

All parameters are exposed as dataclass fields; no value is hard-coded
inside a mechanism.

## Known qualitative differences with Table 7

* **High-type participation.** The paper's pure-reputation baseline
  shows 0.71. Our implementation has no threshold for pure-reputation,
  so participation is 1.0. To match the paper, set
  `apply_threshold` to use a reputation cutoff. We keep the simpler
  "no threshold" semantics because it makes the role of stake-based
  exclusion more legible.
* **Welfare scale.** Welfare uses ad-hoc weighting (`v_quality` vs
  `cost_harm`); the absolute numbers are not comparable to Table 7's
  211.4 but the *ranking* and *widening* with `φ` are stable.

## When to trust the simulation

* Direction of dominance across (mechanism × metric) → stable across
  random seeds and parameter perturbations.
* Magnitude of marginal ablation effects (`run_ablation.py`) → stable
  for `K ≥ 12` replications.
* Heatmap monotonicity in `φ` and `π_c` (`run_sensitivity.py`) →
  stable but noisy at the corners; use `--full` for paper-grade
  resolution.

## When to be cautious

* Absolute welfare numbers depend on the choice of
  `welfare_quality_value` and `welfare_harm_cost` — these are not
  identified from the paper text and were chosen to keep welfare in a
  human-readable range.
* `collusion_success_rate` is sensitive to the threshold definition
  ("realized share > capital share"). It is monotone in coalition
  fraction across all four mechanisms in our simulation, but absolute
  rates differ from Table 7.

## Extending the package

To add a new mechanism:

```python
from abi.mechanisms import Mechanism

class MyMechanism(Mechanism):
    name = "my_mechanism"
    def apply_threshold(self, prod): ...
    def aggregate_reports(self, y, val): ...
    def select_audit(self, prod, Q_hat, eligible, rng): ...
    def allocate_rewards(self, prod, Q_hat, eligible, E_t): ...
    def compute_slashing(self, prod, audited, audit_fail, Q_hat): ...
```

Then register it in `abi.mechanisms.build_mechanism` and add the name
to the `MECHANISMS` tuple of any experiment script. The
`Simulator.run` epoch loop is mechanism-agnostic.

To add a new metric:

```python
# in abi/metrics.py, extend EpochMetrics and compute_metrics
```

The runner aggregates new fields automatically through
`per_replication` and `summary` (it iterates over the dataclass).

## Software stack

Python 3.11+, NumPy ≥ 1.24, Pandas ≥ 2.0, SciPy ≥ 1.10, Matplotlib ≥ 3.7.
No Numba or Cython — the inner loop is pure NumPy, around 8 s per
mechanism for `T = 200, N_p = 200, N_v = 40` on a 2024-era laptop.
