#!/usr/bin/env python
"""
Diagnostic summary of a Lumen simulation bank.

Produces a multi-page PDF with:
  Page 1: Parameter prior distributions (histograms)
  Page 2: Conditional prior checks (γ_max vs γ_min, correlations)
  Page 3: SED quality statistics (flux ranges, NaN/zero rates)
  Page 4: Observation statistics (points per sim, band coverage)
  Page 5: Noise and upper limit diagnostics
  Page 6: Example SEDs (random subset)

Also prints a text summary to stdout.

Usage:
    python diagnose_bank.py bank.h5
    python diagnose_bank.py bank.h5 --output diagnostics.pdf
"""

import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from collections import Counter

MODEL_NAMES = {0: "1A", 1: "1B", 2: "2A", 3: "2B", 4: "3A", 5: "3B"}

BAND_COLORS = {
    "radio": "#2176AE", "submm": "#57B8FF", "ir": "#B66D0D",
    "optical": "#FBB13C", "xray": "#D72638", "gamma": "#7B2D8E",
}


def load_all(h5_path):
    """Load everything from the bank into a dict."""
    d = {}
    with h5py.File(h5_path, "r") as f:
        d["n_sims"] = f.attrs["n_sims"]
        d["n_slots"] = f.attrs["n_slots"]
        d["nx"] = f.attrs.get("nx", "?")
        d["ngamma"] = f.attrs.get("ngamma", "?")
        d["seed"] = f.attrs.get("seed", "?")

        d["nu"] = f["slot_frequencies"][:]
        d["true_flux"] = f["true_flux"][:]
        d["obs_flux"] = f["obs_flux"][:]
        d["obs_err"] = f["obs_err"][:]
        d["obs_mask"] = f["obs_mask"][:]
        d["bands"] = np.array([
            b.decode() if isinstance(b, bytes) else b
            for b in f["slot_bands"][:]
        ])
        d["slot_names"] = np.array([
            n.decode() if isinstance(n, bytes) else n
            for n in f["slot_names"][:]
        ])

        p = f["params"]
        d["G0"] = p["G0"][:]
        d["p"] = p["p"][:]
        d["theta"] = p["theta"][:]
        d["gamma_min"] = p["gamma_min"][:]
        d["gamma_max"] = p["gamma_max"][:]
        d["Rj"] = p["Rj"][:]
        d["Lj"] = p["Lj"][:]
        d["l"] = p["l"][:]
        d["z"] = p["z"][:]
        d["eta_e"] = p["eta_e"][:]
        d["model"] = p["model"][:].astype(int)

    return d


def text_summary(d):
    """Print text diagnostics to stdout."""
    N = d["n_sims"]
    mask = d["obs_mask"]
    true = d["true_flux"]

    n_obs_per_sim = np.sum(mask != 0, axis=1)
    n_det_per_sim = np.sum(mask > 0, axis=1)
    n_ul_per_sim = np.sum(mask < 0, axis=1)

    n_failed = np.sum(np.all(true == 0, axis=1))
    n_nan = np.sum(np.any(np.isnan(true), axis=1))
    n_allzero_obs = np.sum(n_obs_per_sim == 0)

    # Flux dynamic range (non-zero true fluxes)
    pos = true[true > 0]

    sep = "=" * 65
    print(f"\n{sep}")
    print(f"  SIMULATION BANK DIAGNOSTICS: {N} simulations")
    print(sep)
    print(f"  File settings: nx={d['nx']}, nγ={d['ngamma']}, seed={d['seed']}")
    print(f"  Instrument slots: {d['n_slots']}")
    print()

    print(f"  --- SED Quality ---")
    print(f"  Failed (all-zero true flux): {n_failed} ({n_failed/N:.1%})")
    print(f"  Contains NaN:                {n_nan} ({n_nan/N:.1%})")
    print(f"  No observations generated:   {n_allzero_obs} ({n_allzero_obs/N:.1%})")
    if len(pos) > 0:
        print(f"  True flux range:  {pos.min():.2e} — {pos.max():.2e} erg/s/cm²")
        print(f"  True flux median: {np.median(pos):.2e}")
    print()

    print(f"  --- Observation Statistics ---")
    print(f"  Points per sim:  {n_obs_per_sim.min()} – {n_obs_per_sim.max()}"
          f"  (median {np.median(n_obs_per_sim):.0f})")
    print(f"  Detections/sim:  {n_det_per_sim.min()} – {n_det_per_sim.max()}"
          f"  (median {np.median(n_det_per_sim):.0f})")
    print(f"  Upper lim/sim:   {n_ul_per_sim.min()} – {n_ul_per_sim.max()}"
          f"  (median {np.median(n_ul_per_sim):.0f})")
    print()

    # Per-band fill rate
    print(f"  --- Band Fill Rates ---")
    unique_bands = list(dict.fromkeys(d["bands"]))  # preserve order
    for band in unique_bands:
        band_slots = np.array([b == band for b in d["bands"]])
        n_band_slots = band_slots.sum()
        filled = np.sum(mask[:, band_slots] != 0)
        total = N * n_band_slots
        det = np.sum(mask[:, band_slots] > 0)
        ul = np.sum(mask[:, band_slots] < 0)
        ul_frac = ul / max(filled, 1)
        print(f"    {band:>8s}: {n_band_slots} slots, "
              f"{filled/total:.0%} filled, "
              f"{ul_frac:.0%} are UL")

    print()
    print(f"  --- Model Distribution ---")
    counts = Counter(d["model"])
    for m in sorted(counts):
        print(f"    Model {MODEL_NAMES.get(m, m)}: {counts[m]:>6d}  ({counts[m]/N:.1%})")

    # SNR statistics (detections only)
    det_mask = mask > 0
    obs_f = d["obs_flux"]
    obs_e = d["obs_err"]
    snr_all = []
    for i in range(N):
        for j in range(d["n_slots"]):
            if det_mask[i, j] and obs_e[i, j] > 0:
                snr_all.append(obs_f[i, j] / obs_e[i, j])
    snr_all = np.array(snr_all)
    if len(snr_all) > 0:
        print()
        print(f"  --- SNR of Detections ---")
        print(f"  Range:  {snr_all.min():.1f} – {snr_all.max():.1f}")
        print(f"  Median: {np.median(snr_all):.1f}")
        pcts = np.percentile(snr_all, [5, 25, 75, 95])
        print(f"  Percentiles [5,25,75,95]: [{pcts[0]:.1f}, {pcts[1]:.1f}, "
              f"{pcts[2]:.1f}, {pcts[3]:.1f}]")

    print(sep)
    return snr_all


def page_param_histograms(d, fig):
    """Page 1: Parameter prior distributions."""
    fig.suptitle("Parameter Prior Distributions", fontsize=14, y=0.98)

    hist_specs = [
        ("G0",        d["G0"],                False, r"$\Gamma_0$"),
        ("p",         d["p"],                 False, r"$p$"),
        ("θ",         d["theta"],             False, r"$\theta$ [deg]"),
        ("γ_min",     d["gamma_min"],         True,  r"$\gamma_{\rm min}$"),
        ("γ_max",     d["gamma_max"],         True,  r"$\gamma_{\rm max}$"),
        ("Rj",        d["Rj"],                True,  r"$R_j$ [cm]"),
        ("Lj",        d["Lj"],                True,  r"$L_j$ [erg/s]"),
        ("l",         d["l"],                 True,  r"$\ell$ [cm]"),
        ("z",         d["z"],                 False, r"$z$"),
        ("η_e",       d["eta_e"],             True,  r"$\eta_e$"),
        ("model",     d["model"],             False, "Model"),
    ]

    n = len(hist_specs)
    ncols = 4
    nrows = (n + ncols - 1) // ncols

    for i, (name, vals, logscale, label) in enumerate(hist_specs):
        ax = fig.add_subplot(nrows, ncols, i + 1)

        if name == "model":
            counts = Counter(vals)
            models = sorted(counts.keys())
            ax.bar([MODEL_NAMES.get(m, str(m)) for m in models],
                   [counts[m] for m in models], color='steelblue', edgecolor='k')
            ax.axhline(len(vals) / 6, color='r', ls='--', lw=0.8, label='uniform')
            ax.legend(fontsize=7)
        else:
            plot_vals = np.log10(vals) if logscale else vals
            ax.hist(plot_vals, bins=50, color='steelblue', edgecolor='none', alpha=0.8)
            if logscale:
                label = f"log₁₀({label})"

        ax.set_xlabel(label, fontsize=9)
        ax.set_ylabel("count", fontsize=8)
        ax.tick_params(labelsize=7)

    fig.tight_layout(rect=[0, 0, 1, 0.95])


def page_conditional_priors(d, fig):
    """Page 2: Conditional prior checks."""
    fig.suptitle("Conditional Priors & Correlations", fontsize=14, y=0.98)

    # γ_max vs γ_min — should show the gap
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.scatter(np.log10(d["gamma_min"]), np.log10(d["gamma_max"]),
                s=1, alpha=0.3, c='steelblue', rasterized=True)
    # Exclusion zone: log(γ_max) < log(γ_min) + 1
    x_line = np.linspace(1, 4, 100)
    ax1.plot(x_line, x_line + 1, 'r--', lw=1.2, label='min gap (1 dec)')
    ax1.plot(x_line, x_line, 'k:', lw=0.8, label='γ_max = γ_min')
    ax1.set_xlabel(r"log₁₀($\gamma_{\rm min}$)", fontsize=10)
    ax1.set_ylabel(r"log₁₀($\gamma_{\rm max}$)", fontsize=10)
    ax1.legend(fontsize=8)
    ax1.set_title("γ gap enforcement", fontsize=10)

    # Lj vs Rj
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.scatter(np.log10(d["Rj"]), np.log10(d["Lj"]),
                s=1, alpha=0.3, c='steelblue', rasterized=True)
    ax2.set_xlabel(r"log₁₀($R_j$) [cm]", fontsize=10)
    ax2.set_ylabel(r"log₁₀($L_j$) [erg/s]", fontsize=10)
    ax2.set_title("Lj vs Rj (should be uncorrelated)", fontsize=10)

    # G0 vs theta
    ax3 = fig.add_subplot(2, 2, 3)
    sc = ax3.scatter(d["theta"], d["G0"], s=1, alpha=0.3,
                     c=np.log10(d["Lj"]), cmap='viridis', rasterized=True)
    ax3.set_xlabel(r"$\theta$ [deg]", fontsize=10)
    ax3.set_ylabel(r"$\Gamma_0$", fontsize=10)
    ax3.set_title("G0 vs θ (colored by log Lj)", fontsize=10)
    fig.colorbar(sc, ax=ax3, label=r"log₁₀($L_j$)", shrink=0.8)

    # Electron index p vs γ range
    ax4 = fig.add_subplot(2, 2, 4)
    gamma_range = np.log10(d["gamma_max"]) - np.log10(d["gamma_min"])
    ax4.scatter(d["p"], gamma_range, s=1, alpha=0.3,
                c='steelblue', rasterized=True)
    ax4.set_xlabel(r"$p$", fontsize=10)
    ax4.set_ylabel(r"log₁₀($\gamma_{\rm max}/\gamma_{\rm min}$) [decades]", fontsize=10)
    ax4.set_title("Spectral index vs energy range", fontsize=10)

    fig.tight_layout(rect=[0, 0, 1, 0.95])


def page_sed_quality(d, fig):
    """Page 3: SED quality — flux distributions and failure modes."""
    fig.suptitle("SED Quality Diagnostics", fontsize=14, y=0.98)
    true = d["true_flux"]
    N = d["n_sims"]

    # True flux histogram (all positive values)
    ax1 = fig.add_subplot(2, 2, 1)
    pos = true[true > 0]
    if len(pos) > 0:
        ax1.hist(np.log10(pos), bins=80, color='steelblue',
                 edgecolor='none', alpha=0.8)
    ax1.set_xlabel(r"log₁₀($\nu F_\nu$) [erg/s/cm²]", fontsize=10)
    ax1.set_ylabel("count", fontsize=9)
    ax1.set_title("True flux distribution (all slots, all sims)", fontsize=10)

    # Fraction of zero-flux slots per simulation
    ax2 = fig.add_subplot(2, 2, 2)
    frac_zero = np.mean(true <= 0, axis=1)
    ax2.hist(frac_zero, bins=50, color='tomato', edgecolor='none', alpha=0.8)
    ax2.set_xlabel("Fraction of slots with flux ≤ 0", fontsize=10)
    ax2.set_ylabel("count", fontsize=9)
    ax2.set_title("Zero-flux fraction per sim", fontsize=10)
    ax2.axvline(np.median(frac_zero), color='k', ls='--', lw=1,
                label=f'median={np.median(frac_zero):.2f}')
    ax2.legend(fontsize=8)

    # True flux dynamic range per sim
    ax3 = fig.add_subplot(2, 2, 3)
    dyn_ranges = []
    for i in range(N):
        p = true[i][true[i] > 0]
        if len(p) > 1:
            dyn_ranges.append(np.log10(p.max() / p.min()))
    if dyn_ranges:
        ax3.hist(dyn_ranges, bins=50, color='steelblue', edgecolor='none', alpha=0.8)
    ax3.set_xlabel("Dynamic range [decades]", fontsize=10)
    ax3.set_ylabel("count", fontsize=9)
    ax3.set_title("Flux dynamic range per sim", fontsize=10)

    # Peak flux frequency — where does the SED peak?
    ax4 = fig.add_subplot(2, 2, 4)
    nu = d["nu"]
    peak_nu = []
    for i in range(N):
        if np.any(true[i] > 0):
            peak_nu.append(np.log10(nu[np.argmax(true[i])]))
    if peak_nu:
        ax4.hist(peak_nu, bins=50, color='steelblue', edgecolor='none', alpha=0.8)
    ax4.set_xlabel(r"log₁₀($\nu_{\rm peak}$) [Hz]", fontsize=10)
    ax4.set_ylabel("count", fontsize=9)
    ax4.set_title("Peak frequency distribution", fontsize=10)

    fig.tight_layout(rect=[0, 0, 1, 0.95])


def page_obs_statistics(d, fig):
    """Page 4: Observation statistics."""
    fig.suptitle("Observation & Instrument Statistics", fontsize=14, y=0.98)
    mask = d["obs_mask"]
    N = d["n_sims"]

    n_obs = np.sum(mask != 0, axis=1)
    n_det = np.sum(mask > 0, axis=1)
    n_ul = np.sum(mask < 0, axis=1)

    # Points per sim
    ax1 = fig.add_subplot(2, 2, 1)
    bins = np.arange(n_obs.min() - 0.5, n_obs.max() + 1.5)
    ax1.hist(n_obs, bins=bins, color='steelblue', edgecolor='k', alpha=0.8,
             label='total')
    ax1.hist(n_det, bins=bins, color='forestgreen', edgecolor='none',
             alpha=0.5, label='detections')
    ax1.hist(n_ul, bins=bins, color='tomato', edgecolor='none',
             alpha=0.5, label='upper limits')
    ax1.set_xlabel("Points per simulation", fontsize=10)
    ax1.set_ylabel("count", fontsize=9)
    ax1.legend(fontsize=8)
    ax1.set_title("Observation count distribution", fontsize=10)

    # Per-slot fill rate
    ax2 = fig.add_subplot(2, 2, 2)
    fill_rate = np.mean(mask != 0, axis=0)
    slot_idx = np.arange(d["n_slots"])
    colors = [BAND_COLORS.get(b, 'gray') for b in d["bands"]]
    ax2.bar(slot_idx, fill_rate, color=colors, edgecolor='none')
    ax2.set_xlabel("Slot index", fontsize=10)
    ax2.set_ylabel("Fill rate", fontsize=9)
    ax2.set_title("Per-slot fill rate", fontsize=10)
    # Add slot names at angle
    ax2.set_xticks(slot_idx)
    ax2.set_xticklabels(d["slot_names"], rotation=90, fontsize=4)

    # UL fraction per band
    ax3 = fig.add_subplot(2, 2, 3)
    unique_bands = list(dict.fromkeys(d["bands"]))
    band_ul_fracs = []
    band_labels = []
    band_cs = []
    for band in unique_bands:
        bmask = np.array([b == band for b in d["bands"]])
        filled = np.sum(mask[:, bmask] != 0)
        ul = np.sum(mask[:, bmask] < 0)
        if filled > 0:
            band_ul_fracs.append(ul / filled)
        else:
            band_ul_fracs.append(0)
        band_labels.append(band)
        band_cs.append(BAND_COLORS.get(band, 'gray'))

    ax3.barh(band_labels, band_ul_fracs, color=band_cs, edgecolor='k', alpha=0.8)
    ax3.set_xlabel("Upper limit fraction", fontsize=10)
    ax3.set_title("UL fraction by band", fontsize=10)
    ax3.set_xlim(0, 1)

    # Band co-occurrence: how often do we have radio + xray, radio + optical, etc.
    ax4 = fig.add_subplot(2, 2, 4)
    has_band = {}
    for band in unique_bands:
        bmask = np.array([b == band for b in d["bands"]])
        has_band[band] = np.any(mask[:, bmask] != 0, axis=1)

    # Pairwise co-occurrence matrix
    nb = len(unique_bands)
    cooc = np.zeros((nb, nb))
    for i, b1 in enumerate(unique_bands):
        for j, b2 in enumerate(unique_bands):
            cooc[i, j] = np.mean(has_band[b1] & has_band[b2])

    im = ax4.imshow(cooc, cmap='Blues', vmin=0, vmax=1)
    ax4.set_xticks(range(nb))
    ax4.set_yticks(range(nb))
    ax4.set_xticklabels(unique_bands, fontsize=8, rotation=45)
    ax4.set_yticklabels(unique_bands, fontsize=8)
    for i in range(nb):
        for j in range(nb):
            ax4.text(j, i, f"{cooc[i,j]:.0%}", ha='center', va='center',
                     fontsize=6, color='white' if cooc[i, j] > 0.5 else 'black')
    ax4.set_title("Band co-occurrence", fontsize=10)
    fig.colorbar(im, ax=ax4, shrink=0.8)

    fig.tight_layout(rect=[0, 0, 1, 0.95])


def page_noise_snr(d, snr_all, fig):
    """Page 5: Noise and SNR diagnostics."""
    fig.suptitle("Noise & SNR Diagnostics", fontsize=14, y=0.98)
    mask = d["obs_mask"]
    true = d["true_flux"]
    obs = d["obs_flux"]
    err = d["obs_err"]

    # SNR histogram
    ax1 = fig.add_subplot(2, 2, 1)
    if len(snr_all) > 0:
        ax1.hist(np.clip(snr_all, 0, 50), bins=80, color='steelblue',
                 edgecolor='none', alpha=0.8)
        ax1.axvline(np.median(snr_all), color='k', ls='--',
                    label=f'median={np.median(snr_all):.1f}')
        ax1.legend(fontsize=8)
    ax1.set_xlabel("SNR (detections)", fontsize=10)
    ax1.set_ylabel("count", fontsize=9)
    ax1.set_title("SNR distribution", fontsize=10)

    # Fractional error histogram
    ax2 = fig.add_subplot(2, 2, 2)
    det = mask > 0
    frac_err = []
    for i in range(d["n_sims"]):
        for j in range(d["n_slots"]):
            if det[i, j] and obs[i, j] > 0:
                frac_err.append(err[i, j] / obs[i, j])
    if frac_err:
        ax2.hist(np.clip(frac_err, 0, 1.5), bins=80, color='steelblue',
                 edgecolor='none', alpha=0.8)
    ax2.set_xlabel("σ / flux (detections)", fontsize=10)
    ax2.set_ylabel("count", fontsize=9)
    ax2.set_title("Fractional error distribution", fontsize=10)

    # Residual: (obs - true) / err for detections
    ax3 = fig.add_subplot(2, 2, 3)
    residuals = []
    for i in range(d["n_sims"]):
        for j in range(d["n_slots"]):
            if det[i, j] and err[i, j] > 0:
                residuals.append((obs[i, j] - true[i, j]) / err[i, j])
    if residuals:
        residuals = np.array(residuals)
        ax3.hist(np.clip(residuals, -5, 5), bins=100, color='steelblue',
                 edgecolor='none', alpha=0.8, density=True)
        # Overlay standard normal
        xx = np.linspace(-5, 5, 200)
        ax3.plot(xx, np.exp(-xx**2/2) / np.sqrt(2*np.pi), 'r-', lw=1.5,
                 label=r'$\mathcal{N}(0,1)$')
        ax3.legend(fontsize=8)
        ax3.set_title(f"Pull distribution (μ={np.mean(residuals):.2f}, "
                      f"σ={np.std(residuals):.2f})", fontsize=10)
    ax3.set_xlabel("(obs − true) / σ", fontsize=10)
    ax3.set_ylabel("density", fontsize=9)

    # SNR by band
    ax4 = fig.add_subplot(2, 2, 4)
    unique_bands = list(dict.fromkeys(d["bands"]))
    snr_by_band = {b: [] for b in unique_bands}
    for i in range(d["n_sims"]):
        for j in range(d["n_slots"]):
            if det[i, j] and err[i, j] > 0:
                snr_by_band[d["bands"][j]].append(obs[i, j] / err[i, j])

    band_data = [np.array(snr_by_band[b]) for b in unique_bands if len(snr_by_band[b]) > 0]
    band_labels_filt = [b for b in unique_bands if len(snr_by_band[b]) > 0]
    if band_data:
        bp = ax4.boxplot(band_data, labels=band_labels_filt, patch_artist=True,
                         showfliers=False, whis=[5, 95])
        for patch, band in zip(bp['boxes'], band_labels_filt):
            patch.set_facecolor(BAND_COLORS.get(band, 'gray'))
            patch.set_alpha(0.7)
    ax4.set_ylabel("SNR", fontsize=10)
    ax4.set_title("SNR by band", fontsize=10)

    fig.tight_layout(rect=[0, 0, 1, 0.95])


def page_example_seds(d, fig, n_examples=9):
    """Page 6: Random example SEDs."""
    fig.suptitle("Example Simulations (random subset)", fontsize=14, y=0.98)
    N = d["n_sims"]
    rng = np.random.default_rng(0)
    indices = rng.choice(N, min(n_examples, N), replace=False)

    ncols = 3
    nrows = (n_examples + ncols - 1) // ncols

    for k, idx in enumerate(indices):
        ax = fig.add_subplot(nrows, ncols, k + 1)
        nu = d["nu"]
        true = d["true_flux"][idx]
        obs_f = d["obs_flux"][idx]
        obs_e = d["obs_err"][idx]
        obs_m = d["obs_mask"][idx]

        # True SED
        pos = true > 0
        if pos.any():
            ax.plot(nu[pos], true[pos], '-', color='0.75', lw=0.8)

        # Detections
        det = obs_m > 0
        if det.any():
            colors = [BAND_COLORS.get(d["bands"][j], 'gray')
                      for j in range(d["n_slots"]) if det[j]]
            ax.errorbar(nu[det], obs_f[det], yerr=obs_e[det],
                        fmt='o', ms=3, elinewidth=0.8, capsize=1.5,
                        color='steelblue', ecolor='steelblue')

        # Upper limits
        ul = obs_m < 0
        if ul.any():
            ax.scatter(nu[ul], obs_f[ul], marker='v', s=15,
                       color='tomato', zorder=3)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylim(bottom=1e-20, top=1e-7)
        ax.tick_params(labelsize=6)

        n_obs = int((obs_m != 0).sum())
        n_det_i = int(det.sum())
        n_ul_i = int(ul.sum())
        model = int(d["model"][idx])
        ax.set_title(f"#{idx}  M{MODEL_NAMES[model]}  "
                     f"{n_det_i}d+{n_ul_i}ul", fontsize=8)

        if k >= (nrows - 1) * ncols:
            ax.set_xlabel(r"$\nu$ [Hz]", fontsize=7)
        if k % ncols == 0:
            ax.set_ylabel(r"$\nu F_\nu$", fontsize=7)

    fig.tight_layout(rect=[0, 0, 1, 0.95])


def main(h5_path, output):
    d = load_all(h5_path)
    snr_all = text_summary(d)

    print(f"\nGenerating {output}...")
    with PdfPages(output) as pdf:
        for page_fn, label in [
            (page_param_histograms,  "priors"),
            (page_conditional_priors, "conditionals"),
            (page_sed_quality,       "SED quality"),
            (page_obs_statistics,    "observations"),
            (lambda d, fig: page_noise_snr(d, snr_all, fig), "noise/SNR"),
            (page_example_seds,      "examples"),
        ]:
            fig = plt.figure(figsize=(12, 8))
            page_fn(d, fig)
            pdf.savefig(fig, dpi=150)
            plt.close(fig)
            print(f"  ✓ {label}")

    print(f"\nSaved {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnose a Lumen simulation bank")
    parser.add_argument("bank", help="Path to HDF5 bank file")
    parser.add_argument("--output", "-o", default="diagnostics.pdf")
    args = parser.parse_args()

    main(args.bank, args.output)
