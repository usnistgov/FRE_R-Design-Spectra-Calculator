"""Risk-targeted ground motion (RTGM) calculation.

Port of MATLAB ``+usgs/fn_calc_rtgm.m``: integrates a lognormal fragility
shape with a hazard curve to find the ground motion anchor that produces a
target lifetime risk of exceedance.
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import norm


def calc_rtgm(
    sa: np.ndarray,
    afe: np.ndarray,
    beta: float,
    poe: float,
    risk_target: float,
    t: float,
) -> float:
    """Compute the risk-targeted ground motion for one hazard curve.

    Parameters
    ----------
    sa : Sa values of the hazard curve (period dependent), increasing.
    afe : annual frequency of exceedance for each Sa value.
    beta : fragility curve lognormal standard deviation.
    poe : probability of exceedance given target shaking
        (e.g., 0.10 for a 10% chance of exceedance given the RTGM).
    risk_target : target risk of exceedance across the timespan
        (e.g., 0.01 for 1% risk).
    t : timespan of risk in years (e.g., design life of building).

    Returns
    -------
    rtgm : Sa value of the risk-targeted ground motion.
    """
    sa = np.asarray(sa, dtype=float)
    afe = np.asarray(afe, dtype=float)

    # Define search grid of candidate anchor points (MATLAB: sa(1):0.01:sa(end))
    n_steps = int(np.floor((sa[-1] - sa[0]) / 0.01 + 1e-10))
    x = sa[0] + 0.01 * np.arange(n_steps + 1)

    # Define fragility: for each candidate anchor x, the lognormal median such
    # that P(exceed | Sa = x) = poe
    z = norm.ppf(poe)
    theta = np.exp(np.log(x) - z * beta)  # (n_x,)

    # Fragility pdf evaluated at hazard-curve Sa points: shape (n_x, n_sa)
    frag_pdf = _lognpdf(sa[None, :], np.log(theta)[:, None], beta)

    # Integrate hazard curve and fragility to get annual frequency of exceedance
    haz_x_frag_pdf = frag_pdf * afe[None, :]
    cum_afe = np.trapezoid(haz_x_frag_pdf, sa, axis=1)  # (n_x,)

    # Lifetime risk over t years
    risk = 1 - (1 - cum_afe) ** t

    # Find RTGM: interpolate log(x) at the target risk using unique risk points
    _, first_idx = np.unique(risk, return_index=True)
    filt = np.sort(first_idx)  # 'stable' unique: first occurrences, original order
    f = interp1d(risk[filt], np.log(x[filt]), kind="linear", fill_value="extrapolate")
    return float(np.exp(f(risk_target)))


def _lognpdf(x, mu, sigma):
    """Lognormal pdf matching MATLAB ``lognpdf(x, mu, sigma)``."""
    return np.exp(-((np.log(x) - mu) ** 2) / (2 * sigma**2)) / (
        x * sigma * np.sqrt(2 * np.pi)
    )
