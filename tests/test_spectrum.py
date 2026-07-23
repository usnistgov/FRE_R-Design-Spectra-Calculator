"""Unit tests for the deterministic spectrum / design-value math.

These do not touch the network or the cache; they exercise the ported MATLAB
logic on synthetic inputs.
"""

import numpy as np

from frer.spectrum import T_VEC, develop_2point_design_values


def test_2point_sds_is_90pct_of_peak():
    period = T_VEC
    sa = np.ones_like(period)
    sa[period == 0.3] = 2.0  # peak inside the 0.2-5s window
    sds, _ = develop_2point_design_values(sa, period)
    assert sds == 0.9 * 2.0


def test_2point_sd1_takes_max_of_two_rules():
    period = np.array([0.2, 1.0, 1.5, 2.0])
    # Sa(1.0) = 0.5; 0.9*max(T*Sa) over [1,2] with T=1.5, Sa=0.6 -> 0.9*0.9=0.81
    sa = np.array([1.0, 0.5, 0.6, 0.4])
    _, sd1 = develop_2point_design_values(sa, period)
    assert np.isclose(sd1, max(0.5, 0.9 * 1.5 * 0.6))


def test_2point_matches_hand_calc():
    period = np.array([0.2, 0.5, 1.0, 2.0, 5.0])
    sa = np.array([1.5, 1.2, 0.8, 0.4, 0.1])
    sds, sd1 = develop_2point_design_values(sa, period)
    assert np.isclose(sds, 0.9 * 1.5)
    # max(Sa(1)=0.8, 0.9*max(1*0.8, 2*0.4)=0.9*0.8=0.72) -> 0.8
    assert np.isclose(sd1, 0.8)
