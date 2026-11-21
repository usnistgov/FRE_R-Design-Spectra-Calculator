"""Site table loading and site-class expansion.

There used to be three site sets -- usgs_160, nehrp_34 and frtc_testbed -- and
a column per set in ``sites.csv`` saying which rows belonged to which. The
other two are gone: every nehrp_34 site was already a usgs_160 site (all 34
were the same rows), and frtc_testbed was 9 street addresses that are not NSHM
test sites, so no published data release covers them. usgs_160 is the whole
table now, and a site no longer opts into a set or names its own site class.
"""

from pathlib import Path

import pandas as pd

from .params import DEFAULT_SITE_CLASSES

SITE_SET = "usgs_160"


def load_sites(sites_csv: Path) -> pd.DataFrame:
    """Load sites.csv.

    Rows are returned in file order (ascending id), matching the order the
    original MATLAB tool processed them.
    """
    return pd.read_csv(sites_csv).reset_index(drop=True)


def site_class_list(site_row=None) -> list[tuple[str, float]]:
    """Return [(site_class, vs30), ...] to pull/calculate for one site.

    Every site gets the same classes. The argument is accepted and ignored so
    that callers reading a site table row by row need not special-case this.
    """
    return list(DEFAULT_SITE_CLASSES.items())
