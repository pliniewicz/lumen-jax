#!/usr/bin/env python
"""
Visualize the nth simulation from the HDF5 bank.

Usage:
    python view_sim.py 42
    python view_sim.py 42 --bank bank.h5 --save
"""

import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Band colors
BAND_COLORS = {
    "radio":   "#2176AE",
    "submm":   "#57B8FF",
    "ir":      "#B66D0D",
    "optical": "#FBB13C",
    "xray":    "#D72638",
    "gamma":   "#7B2D8E",
}

BAND_ORDER = ["radio", "submm", "ir", "optical", "xray", "gamma"]

MODEL_NAMES = {0: "1A", 1: "1B", 2: "2A", 3: "2B", 4: "3A", 5: "3B"}


def view_sim(idx, h5_path="bank.h5", save=False):
    with h5py.File(h5_path, "r") as f:
        n_sims = f.attrs["n_sims"]
        if idx < 0 or idx >= n_sims:
            print(f"Index {idx} out of range [0, {n_sims-1}]")
            return

        nu       = f["slot_frequencies"][:]
        true     = f["true_flux"][idx]
        obs_flux = f["obs_flux"][idx]
        obs_err  = f["obs_err"][idx]
        obs_mask = f["obs_mask"][idx]
        bands    = [b.decode() if isinstance(b, bytes) else b for b in f["slot_bands"][:]]
        names    = [n.decode() if isinstance(n, bytes) else n for n in f["slot_names"][:]]

        p = f["params"]
        G0    = p["G0"][idx]
        p_idx = p["p"][idx]
        theta = p["theta"][idx]
        gmin  = p["gamma_min"][idx]
        gmax  = p["gamma_max"][idx]
        Rj    = p["Rj"][idx]
        Lj    = p["Lj"][idx]
        l     = p["l"][idx]
        z     = p["z"][idx]
        eta_e = p["eta_e"][idx]
        model = int(p["model"][idx])

    # Separate detections and upper limits
    det_mask = obs_mask > 0
    ul_mask  = obs_mask < 0
    n_det = int(det_mask.sum())
    n_ul  = int(ul_mask.sum())
    n_obs = n_det + n_ul

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(10, 6))

    # True SED (faint line through all slots with positive flux)
    pos = true > 0
    ax.plot(nu[pos], true[pos], '-', color='0.75', lw=1.2, zorder=1,
            label='True SED')

    # Detections with error bars, colored by band
    for band in BAND_ORDER:
        mask_b = det_mask & np.array([b == band for b in bands])
        if not mask_b.any():
            continue
        ax.errorbar(
            nu[mask_b], obs_flux[mask_b], yerr=obs_err[mask_b],
            fmt='o', ms=7, color=BAND_COLORS[band], ecolor=BAND_COLORS[band],
            elinewidth=1.5, capsize=3, zorder=3, label=f'{band} det',
        )

    # Upper limits (downward triangles)
    for band in BAND_ORDER:
        mask_b = ul_mask & np.array([b == band for b in bands])
        if not mask_b.any():
            continue
        ax.scatter(
            nu[mask_b], obs_flux[mask_b],
            marker='v', s=60, color=BAND_COLORS[band],
            edgecolors='k', linewidths=0.5, zorder=3,
            label=f'{band} UL',
        )

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$\nu$ [Hz]', fontsize=13)
    ax.set_ylabel(r'$\nu F_\nu$ [erg/s/cm$^2$]', fontsize=13)

    ax.set_ylim(bottom=1e-30)

    # Parameter box
    param_text = (
        f"Sim #{idx}  —  Model {MODEL_NAMES[model]}\n"
        f"G₀={G0:.1f}   p={p_idx:.2f}   θ={theta:.1f}°   z={z:.2f}\n"
        f"γ_min={gmin:.0e}   γ_max={gmax:.0e}\n"
        f"Lj={Lj:.1e} erg/s   Rj={Rj:.1e} cm\n"
        f"l={l:.1e} cm   η_e={eta_e:.3f}\n"
        f"{n_det} detections, {n_ul} upper limits"
    )
    ax.text(0.02, 0.02, param_text, transform=ax.transAxes,
            fontsize=8.5, verticalalignment='bottom',
            fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.4', fc='white', alpha=0.85))

    # Annotate instrument names near points
    for j in range(len(nu)):
        if obs_mask[j] != 0:
            ax.annotate(
                names[j], (nu[j], obs_flux[j]),
                fontsize=5.5, color='0.4', rotation=45,
                xytext=(3, 6), textcoords='offset points',
            )

    ax.legend(fontsize=8, ncol=2, loc='upper right')
    ax.set_title(f'Simulation #{idx}  ({n_obs} observed points)', fontsize=12)
    fig.tight_layout()

    if save:
        fname = f"sim_{idx}.png"
        fig.savefig(fname, dpi=180)
        print(f"Saved {fname}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="View a simulation from the bank")
    parser.add_argument("index", type=int, help="Simulation index")
    parser.add_argument("--bank", default="bank.h5")
    parser.add_argument("--save", action="store_true", help="Save PNG instead of showing")
    args = parser.parse_args()

    view_sim(args.index, args.bank, args.save)
