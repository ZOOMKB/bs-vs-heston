from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OptionGreeks:
    """Option sensitivities with vega per 1.00 volatility unit and theta per calendar day."""

    delta: float
    gamma: float
    vega: float
    theta: float

    def to_dict(self) -> dict[str, float]:
        return {
            "delta": self.delta,
            "gamma": self.gamma,
            "vega": self.vega,
            "theta": self.theta,
        }


@dataclass(frozen=True)
class HestonParams:
    """Parameters of the Heston stochastic volatility model."""

    v0: float
    kappa: float
    theta: float
    sigma_v: float
    rho: float

    def __post_init__(self) -> None:
        if self.v0 <= 0:
            raise ValueError("v0 must be positive.")
        if self.kappa <= 0:
            raise ValueError("kappa must be positive.")
        if self.theta <= 0:
            raise ValueError("theta must be positive.")
        if self.sigma_v <= 0:
            raise ValueError("sigma_v must be positive.")
        if not (-1 < self.rho < 1):
            raise ValueError("rho must be in (-1, 1).")

    @property
    def feller_ratio(self) -> float:
        return float(2.0 * self.kappa * self.theta / self.sigma_v**2)

    @property
    def feller_satisfied(self) -> bool:
        return self.feller_ratio > 1.0

    @property
    def vol_0(self) -> float:
        return float(np.sqrt(self.v0))

    @property
    def vol_long(self) -> float:
        return float(np.sqrt(self.theta))

    def to_array(self) -> np.ndarray:
        return np.array([self.v0, self.kappa, self.theta, self.sigma_v, self.rho], dtype=float)

    @classmethod
    def from_array(cls, values: np.ndarray | list[float] | tuple[float, ...]) -> "HestonParams":
        if len(values) != 5:
            raise ValueError("HestonParams requires exactly five values.")
        return cls(
            v0=float(values[0]),
            kappa=float(values[1]),
            theta=float(values[2]),
            sigma_v=float(values[3]),
            rho=float(values[4]),
        )
