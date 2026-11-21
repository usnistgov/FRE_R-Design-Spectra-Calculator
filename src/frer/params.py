"""Analysis parameter definitions (code version, risk-target parameters)."""

from dataclasses import dataclass

# Site classes the batch workflow calculates, and the vs30 (m/s) used for
# each. The vs30 is a path segment of the hazard request URL (see
# frer.usgs_api.HAZARD_URL); the service takes it literally and does not round
# or snap it.
#
# These are the NEHRP classes at the Vs30 values the USGS publishes for them.
# The set was BC/C/D/E while every curve cost an API call and CD and DE were
# not worth 320 more; the frozen cache is now built from a data release that
# carries all of them, so the two intermediate classes are free and included.
#
# D is 260, which is what the USGS uses for class D. It was 265 until
# 2026-11-11, a value inherited from the original MATLAB tool. Refreshing the
# class D curves at 260 moved design values by 0.26% to 2.08% across 160 sites
# (median under 1%).
DEFAULT_SITE_CLASSES = {
    "BC": 760,
    "C": 530,
    "CD": 365,
    "D": 260,
    "DE": 185,
    "E": 150,
}

# Every NEHRP site class and the vs30 (m/s) the USGS maps it to, as published
# by the hazard service's usage endpoint. Wider than DEFAULT_SITE_CLASSES
# above, which omits A and B: the batch workflow has no use for rock that hard,
# but the web app lets a single site be placed on any of them, and this seeds
# its vs30 box.
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

# Code version -> National Seismic Hazard Model release year.
#
# ASCE 7-22 was supported until 2026-11-20 and is gone. It was a different kind
# of calculation: its DE values came from the USGS design-value web service,
# which mixes risk-targeted and deterministic caps, while only its FREr values
# were developed here from NSHM 2018 hazard curves. ASCE 7-28 is fully
# probabilistic, so every number this tool reports now comes from one place --
# the hazard curves -- and the two halves of a result can no longer be on
# different bases.
VERSION_NSHM = {
    "ASCE 7-28": 2023,
}

# The one code version this tool develops values for.
VERSION_ID = "ASCE 7-28"

# NSHM release year -> the model revision to request from the hazard service.
# This string becomes a path segment of the request URL, which
# frer.usgs_api.HAZARD_URL builds as "conus-{edition}".
#
# Pinned to an explicit revision rather than the bare year, for two reasons.
#
# The bare year is an alias that follows whatever revision USGS currently
# serves: "conus-2023" answered as 2023.R1 and now answers as 2023.R2, with no
# change on our side. A pull is supposed to be reproducible, and under an alias
# it is not -- re-running it after a USGS revision silently returns different
# curves. This is not hypothetical: tests/usgs_cache was pulled through the
# alias and holds R2 curves, which is why it disagreed with the published R1
# data release across the central and eastern U.S. at the softer site classes
# while matching it exactly on the west coast.
#
# R1 is the revision these design values are developed from, and that is a code
# decision rather than a scientific one. R1 is the model the committee adopted,
# so it is the basis of an official design value. R2 is a later science update:
# more current, and the better estimate of hazard, but not the adopted basis.
#
# Changing a string here changes every value this tool produces, so it is a
# deliberate act, not a configuration knob.
NSHM_EDITION = {
    2023: "2023.R1",
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
# the DE/MCEr ordinates are converted to max direction by the MAXDIR factors in
# frer.spectrum (LS_PARAMS.maxdir). FREr and DE ordinates are therefore NOT directly
# comparable -- one has to be converted to the other's orientation first (the
# MAXDIR factors in frer.spectrum do that conversion). These names annotate the
# CSV headers and UI labels so an exported value carries its own orientation.
#
# Return periods are deliberately not annotated: every one is read off the
# RotD50 hazard curve at the unscaled RTGM, so they share a single orientation
# and are directly comparable with each other.
FR_ORIENTATION = "RotD50"
DE_ORIENTATION = "RotD100"
