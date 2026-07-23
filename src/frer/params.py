"""Analysis parameter definitions (code versions, risk-target parameters)."""

from dataclasses import dataclass

# Site classes pulled by default when a site does not specify one, and the
# vs30 (m/s) value used for the hazard-curve call for each class.
DEFAULT_SITE_CLASSES = {"BC": 760, "C": 530, "D": 265, "E": 150}

# Code version -> National Seismic Hazard Model release year
VERSION_NSHM = {
    "ASCE 7-22": 2018,
    "ASCE 7-28": 2023,
}


@dataclass(frozen=True)
class RiskTargetParams:
    """Parameters defining a risk-targeted ground motion calculation."""

    beta: float  # fragility lognormal standard deviation
    poe: float  # probability of exceedance given target shaking (fragility anchor)
    risk_target: float  # target risk of exceedance over the timespan
    t: float  # timespan of risk in years
    maxdir: bool  # convert from RotD50 (geomean-ish) to RotD100 (max direction)


# Functional Recovery: 10% risk of exceedance in 50 years, fragility with 25%
# PoE at the target motion, beta = 0.6, no max-direction conversion (RotD50).
FR_PARAMS = RiskTargetParams(beta=0.6, poe=0.25, risk_target=0.1, t=50, maxdir=False)

# ASCE 7-28 Life Safety / DE ground motion parameters: 1.5% risk in 50 years,
# 10% PoE, beta = 0.6, converted to max direction (RotD100).
LS_PARAMS = RiskTargetParams(beta=0.6, poe=0.1, risk_target=0.015, t=50, maxdir=True)
