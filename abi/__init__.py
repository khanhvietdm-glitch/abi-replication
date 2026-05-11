"""
ABI: Asset-Backed Intelligence replication package.

Reference: "Asset-Backed Intelligence: A Welfare-Guided Mechanism Design
Framework for Trust in Decentralized AI Markets."

Implements:
    * Comparative statics from Section 7.2 (feasible slashing interval,
      minimum audit probability, speculative attenuation, governance stability).
    * Agent-based simulation from Section 7.5 with four mechanisms:
      Pure-Stake, Pure-Reputation, Yuma-like, ABI.
    * Ablation studies (Appendix C.4) and sensitivity sweeps (Appendix C.5).
"""

from .config import Params, ABI_BASELINE, SIM_BASELINE
from .mechanisms import (
    PureStakeMechanism,
    PureReputationMechanism,
    YumaLikeMechanism,
    ABIMechanism,
)
from .simulation import Simulator
from .metrics import EpochMetrics

__version__ = "1.0.0"
__all__ = [
    "Params",
    "ABI_BASELINE",
    "SIM_BASELINE",
    "PureStakeMechanism",
    "PureReputationMechanism",
    "YumaLikeMechanism",
    "ABIMechanism",
    "Simulator",
    "EpochMetrics",
]
