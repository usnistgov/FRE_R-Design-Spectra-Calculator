"""Risk-targeted spectra and 2-point design values.

Ports of MATLAB ``+usgs/fn_develop_RT_spectrum.m`` and
``+usgs/fn_develop_2point_design_values.m``.
"""

from dataclasses import dataclass, field

import numpy as np

from .params import RiskTargetParams
from .rtgm import calc_rtgm

# Fixed period vector for the multi-period spectrum (seconds)
T_VEC = np.array(
    [0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3,
     0.4, 0.5, 0.75, 1, 1.5, 2, 3, 4, 5, 7.5, 10]
)  # fmt: skip

# RotD50 -> RotD100 (max direction) conversion factors for each period in
# T_VEC, based on the USGS conversion (from Nico)
MAXDIR = np.array(
    [1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.203, 1.206,
     1.213, 1.219, 1.234, 1.250, 1.253, 1.256, 1.261, 1.267, 1.272, 1.286, 1.3]
)  # fmt: skip


@dataclass
class RTSpectrum:
    """A risk-targeted multi-period spectrum."""

    period: np.ndarray = field(default_factory=lambda: T_VEC.copy())
    sa: np.ndarray = field(default_factory=lambda: np.zeros(len(T_VEC)))
    return_period: np.ndarray = field(default_factory=lambda: np.zeros(len(T_VEC)))
    return_period_67: np.ndarray = field(default_factory=lambda: np.zeros(len(T_VEC)))


def _find_hazard_curve(hazard_curves: list, t: float):
    """Locate the (gms, afes) hazard curve for period ``t`` in a USGS response.

    ``hazard_curves`` is the ``hazardCurves`` list from the USGS nshmp hazard
    web service: each item has ``imt: {value, display}`` and
    ``data[0].values: {xs, ys}``. Period 0 maps to PGA; other periods match
    the numeric portion of the display label (e.g., "0.2 s").
    """
    for curve in hazard_curves:
        label = curve["imt"]["value"]
        display = curve["imt"]["display"]
        try:
            spectral_value = float(display[:-2])
        except ValueError:
            spectral_value = np.nan

        if (label == "PGA" and t == 0) or ("SA" in label and spectral_value == t):
            gms = np.asarray(curve["data"][0]["values"]["xs"], dtype=float)
            afes = np.asarray(curve["data"][0]["values"]["ys"], dtype=float)
            keep = afes > 0
            return gms[keep], afes[keep]
    raise ValueError(f"couldn't find hazard values for period {t}")


def _interp_return_period(gms: np.ndarray, afes: np.ndarray, sa: float) -> float:
    """Return period (years) of ``sa`` on the hazard curve, log-linear in AFE.

    Matches MATLAB ``1/exp(interp1(gms, log(afes), sa))``: NaN outside range.
    """
    if sa < gms[0] or sa > gms[-1]:
        return float("nan")
    return float(1 / np.exp(np.interp(sa, gms, np.log(afes))))


def develop_rt_spectrum(hazard_curves: list, rt_params: RiskTargetParams) -> RTSpectrum:
    """Calculate a risk-targeted spectrum from USGS hazard curves."""
    spec = RTSpectrum()
    for i, t in enumerate(T_VEC):
        gms, afes = _find_hazard_curve(hazard_curves, t)

        try:
            rtgm = calc_rtgm(
                gms,
                afes,
                rt_params.beta,
                rt_params.poe,
                rt_params.risk_target,
                rt_params.t,
            )
        except Exception as e:
            raise RuntimeError("failed to converge on risk target") from e

        spec.sa[i] = MAXDIR[i] * rtgm if rt_params.maxdir else rtgm
        spec.return_period[i] = _interp_return_period(gms, afes, rtgm)
        spec.return_period_67[i] = _interp_return_period(gms, afes, rtgm * 2 / 3)
    return spec


def develop_2point_design_values(sa: np.ndarray, period: np.ndarray):
    """Calculate two-point spectral design values per ASCE 7-22 Section 21.4.

    Returns
    -------
    (sds, sd1) : short-period and 1-second spectral design values.
    """
    sa = np.asarray(sa, dtype=float)
    period = np.asarray(period, dtype=float)

    # Sds: 90% of the peak between 0.2 and 5 seconds
    range_s = (period >= 0.2) & (period <= 5.0)
    sds = 0.9 * np.max(sa[range_s])

    # Sd1: max of Sa(1.0s) and 90% of the peak of T*Sa between 1 and 2 seconds
    range_1 = (period >= 1.0) & (period <= 2.0)
    sd1 = max(
        float(np.max(sa[period == 1.0])),
        0.9 * float(np.max(period[range_1] * sa[range_1])),
    )
    return float(sds), float(sd1)
