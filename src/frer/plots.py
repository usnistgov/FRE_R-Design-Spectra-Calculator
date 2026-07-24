"""Plotting: design spectra and FREr-vs-DE scatter comparisons."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from .spectrum import two_point_spectrum

# MATLAB default color order
CMAP = [
    (0.0000, 0.4470, 0.7410),
    (0.8500, 0.3250, 0.0980),
    (0.9290, 0.6940, 0.1250),
    (0.4940, 0.1840, 0.5560),
    (0.4660, 0.6740, 0.1880),
    (0.3010, 0.7450, 0.9330),
    (0.6350, 0.0780, 0.1840),
]


def draw_spectra(ax, result):
    """Draw one site's DE and FREr spectra, multi-period and 2-point, on ``ax``.

    ``result`` is a :class:`frer.single.SiteResult`.
    """
    fr_t, fr_sa = two_point_spectrum(result.sdsfr, result.sd1fr)
    ls_t, ls_sa = two_point_spectrum(result.sds, result.sd1)

    ax.loglog(result.de_period, result.de_sa, "-k", label="DE: MPS")
    ax.loglog(ls_t, ls_sa, "--k", label="DE: 2-point")
    ax.loglog(
        result.fr_spectrum.period,
        result.fr_spectrum.sa,
        "-",
        color=CMAP[0],
        label="FRE$_R$: MPS",
    )
    ax.loglog(fr_t, fr_sa, "--", color=CMAP[0], label="FRE$_R$: 2-point")

    ax.set_xlim(0.01, 5)
    ax.set_xlabel("Period (s)")
    ax.set_ylabel("Sa (g)")
    ax.legend(loc="lower left")
    ax.grid(True, which="both", alpha=0.4)
    ax.set_title(f"{result.version_id}: Site Class {result.site_class}")


def build_spectra_figure(results):
    """Build a log-log spectra figure, one panel per code version.

    ``results`` is a sequence of :class:`frer.single.SiteResult` for the same
    site. Comparing versions gets a panel each rather than eight lines on one
    axes, so each panel stays as readable as the single-version plot.

    Built as a bare :class:`~matplotlib.figure.Figure` rather than through
    ``pyplot`` so it never enters pyplot's global registry: callers get a figure
    they own outright, with nothing to close and no shared state to trip over.
    """
    results = list(results)
    fig = Figure(figsize=(6.4 * len(results), 4.8))
    axes = fig.subplots(1, len(results), squeeze=False)
    for ax, result in zip(axes[0], results, strict=True):
        draw_spectra(ax, result)
    fig.tight_layout()
    return fig


def plot_spectra(plot_dir: Path, site_name: str, result):
    """Save a single-site, single-version spectra plot as a PNG."""
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig = build_spectra_figure([result])
    fig.savefig(plot_dir / f"{site_name} - Site Class {result.site_class}.png", dpi=150)


def plot_scatter_comparisons(plot_dir: Path, table):
    """Scatter plots comparing FREr with DE design values by site class."""
    plot_dir.mkdir(parents=True, exist_ok=True)

    pairs = [
        ("Sds", "Sdsfr", "S$_{DS}$ (g)", "S$_{DSfr}$ (g)"),
        ("Sd1", "Sd1fr", "S$_{D1}$ (g)", "S$_{D1fr}$ (g)"),
    ]
    regions = sorted(table["region"].unique())

    for site_class in sorted(table["site_class"].unique()):
        in_class = table["site_class"] == site_class
        for x_col, y_col, x_lab, y_lab in pairs:
            fig, ax = plt.subplots()
            for i, region in enumerate(regions):
                sel = in_class & (table["region"] == region)
                ax.scatter(
                    table.loc[sel, x_col],
                    table.loc[sel, y_col],
                    color=CMAP[i % len(CMAP)],
                    label=region,
                )
            lim = np.ceil(table.loc[in_class, x_col].max() * 10) / 10
            ax.plot([0, lim], [0, lim], "--k")
            ax.plot([0, lim], [0, 0.8 * lim], ":k")
            ax.plot([0, lim], [0, 1.2 * lim], ":k", label="+/- 20%")
            ax.set_xlim(0, lim)
            ax.set_ylim(0, lim)
            ax.set_title(f"Site Class {site_class}")
            ax.legend(loc="upper left")
            ax.set_xlabel(x_lab)
            ax.set_ylabel(y_lab)
            fig.savefig(plot_dir / f"{x_col}_v_{y_col}_{site_class}.png", dpi=150)
            plt.close(fig)
