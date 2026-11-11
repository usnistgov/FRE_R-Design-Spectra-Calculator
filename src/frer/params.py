"""Analysis parameter definitions (code versions, risk-target parameters)."""

from dataclasses import dataclass

# Site classes pulled by default when a site does not specify one, and the
# vs30 (m/s) sent to the hazard service for each.
#
# Only the hazard service takes a vs30 from us: it is a path segment of the
# request URL (see frer.usgs_api.HAZARD_URL), and the service does not round
# or snap it. The design-maps service, by contrast, takes the site class as a
# string and resolves the vs30 itself -- so these numbers never reach it, and
# changing them cannot affect an ASCE 7-22 Sds or Sd1.
#
# D is 260 to agree with what the design-maps service uses for class D, which
# it reports back as response.metadata.vs30. It was 265 until 2026-11-11, a
# value inherited from the original MATLAB tool; the consequence was that an
# ASCE 7-22 class D result mixed two soil profiles, its DE values computed at
# 260 and its FREr values at 265. Refreshing the committed class D curves at
# 260 moved design values by 0.26% to 2.08% across 160 sites (median under
# 1%), and left the MATLAB return-period comparison unchanged: no row moved
# outside its band, so KNOWN_RP_EXCEPTIONS did not grow.
DEFAULT_SITE_CLASSES = {"BC": 760, "C": 530, "D": 260, "E": 150}

# Every ASCE 7-22 site class and the vs30 (m/s) the USGS design-maps service
# maps it to, as published by that service's usage endpoint. Nothing is sent to
# that service from here -- it is given the class as a string -- so this table
# exists to seed the vs30 box for an arbitrary single site in the web app, and
# to say on the record what the service means by each class.
#
# It still differs from DEFAULT_SITE_CLASSES above, which covers only the four
# classes the batch pull fetches and carries no entry for A, B, CD, DE.
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

# Ground-motion orientation each spectrum is reported in. FREr keeps the RotD50
# orientation of the underlying hazard curves (FR_PARAMS.maxdir is False), while
# the DE/MCEr ordinates are max direction: for ASCE 7-28 via the MAXDIR factors
# (LS_PARAMS.maxdir), and for ASCE 7-22 because the USGS design-value service
# reports max-direction values. FREr and DE ordinates are therefore NOT directly
# comparable -- one has to be converted to the other's orientation first (the
# MAXDIR factors in frer.spectrum do that conversion). These names annotate the
# CSV headers and UI labels so an exported value carries its own orientation.
#
# Return periods are deliberately not annotated: every one is read off the
# RotD50 hazard curve at the unscaled RTGM, so they share a single orientation
# and are directly comparable with each other.
FR_ORIENTATION = "RotD50"
DE_ORIENTATION = "RotD100"
