"""Plotting: design spectra and FREr-vs-DE scatter comparisons."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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


def _two_point_spectrum(sds: float, sd1: float):
    """Build the 2-point code spectrum polyline (T, Sa) for plotting."""
    ts = sd1 / sds
    t0 = 0.2 * ts
    t_range = np.arange(ts + 0.1, 5.0 + 1e-9, 0.1)
    t_vals = np.concatenate([[0.01, t0, ts], t_range])  # 0.01 for log axis
    sa_vals = np.concatenate([[0.4 * sds, sds, sds], sd1 / t_range])
    return t_vals, sa_vals


def plot_spectra(
    plot_dir: Path,
    site_name: str,
    site_class: str,
    version_id: str,
    rt_spectrum_fr,
    ls_period,
    ls_sa_de,
    sds: float,
    sd1: float,
    sdsfr: float,
    sd1fr: float,
):
    """Log-log plot of the DE and FREr multi-point and 2-point spectra."""
    plot_dir.mkdir(parents=True, exist_ok=True)

    fr_t, fr_sa = _two_point_spectrum(sdsfr, sd1fr)
    ls_t, ls_sa = _two_point_spectrum(sds, sd1)

    fig, ax = plt.subplots()
    ax.loglog(ls_period, ls_sa_de, "-k", label="DE: MPS")
    ax.loglog(ls_t, ls_sa, "--k", label="DE: 2-point")
    ax.loglog(
        rt_spectrum_fr.period,
        rt_spectrum_fr.sa,
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
    ax.set_title(f"{version_id}: Site Class {site_class}")
    fig.savefig(plot_dir / f"{site_name} - Site Class {site_class}.png", dpi=150)
    plt.close(fig)


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
