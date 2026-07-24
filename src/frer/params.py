"""Analysis parameter definitions (code versions, risk-target parameters)."""

from dataclasses import dataclass

# Site classes pulled by default when a site does not specify one, and the
# vs30 (m/s) value used for the hazard-curve call for each class.
DEFAULT_SITE_CLASSES = {"BC": 760, "C": 530, "D": 265, "E": 150}

# Every ASCE 7-22 site class and the vs30 (m/s) the USGS nshmp service maps it
# to, as published by that service's usage endpoint. Used to seed the default
# vs30 for an arbitrary single site; the batch pull keeps using
# DEFAULT_SITE_CLASSES above, whose D is 265 rather than 260 (a value inherited
# from the original MATLAB tool that the committed benchmark fixtures depend
# on). Both are correct for their own purpose -- do not reconcile them.
SITE_CLASS_VS30 = {
    "A": 1500,
    "B": 1080,
    "BC": 760,
    "C": 530,
    "CD": 365,
    "D": 260,
    "DE": 185,
    "E": 150,
}

# Range of vs30 (m/s) the nshmp hazard service accepts.
VS30_RANGE = (150, 1500)

# Geographic extent of the conterminous-U.S. hazard models, as
# (lon_min, lat_min, lon_max, lat_max). Coordinates outside this box still
# return an HTTP 200 with status "success" but with unusable curves, so callers
# must check the bounds themselves before making a request.
NSHM_BOUNDS = (-125.0, 24.4, -65.0, 50.0)

# Risk categories accepted by the design-value web service.
RISK_CATEGORIES = ("I", "II", "III", "IV")

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
