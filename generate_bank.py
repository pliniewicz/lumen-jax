#!/usr/bin/env python
"""
Generate simulation bank for SBI training.

Produces an HDF5 file with:
  - Raw SED fluxes at instrument slot frequencies
  - Parameter vectors with conditional priors
  - Realistic band-dependent noise and upper limits
  - Discrete model labels

Usage:
    python generate_bank.py --n_sims 100000 --output bank.h5
"""

import argparse
import time
import os
import numpy as np
import h5py

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from lumen import (
    make_params, make_cosmology, observed_sed,
    MODEL_1A, MODEL_1B, MODEL_2A, MODEL_2B, MODEL_3A, MODEL_3B,
    kpc, MODEL_NAMES,
)


# ==================================================================
#  Instrument catalog
# ==================================================================

INSTRUMENTS = {
    # --- RADIO ---
    "LOFAR_LBA":    {"nu_Hz": [54e6],                "prob": 0.10, "band": "radio"},
    "LOFAR_HBA":    {"nu_Hz": [144e6],               "prob": 0.20, "band": "radio"},
    "GMRT_150":     {"nu_Hz": [150e6],               "prob": 0.20, "band": "radio"},
    "GMRT_325":     {"nu_Hz": [325e6],               "prob": 0.18, "band": "radio"},
    "GMRT_610":     {"nu_Hz": [610e6],               "prob": 0.15, "band": "radio"},
    "RACS":         {"nu_Hz": [888e6],               "prob": 0.15, "band": "radio"},
    "MeerKAT_L":    {"nu_Hz": [1.28e9],              "prob": 0.12, "band": "radio"},
    "VLA_L":        {"nu_Hz": [1.4e9],               "prob": 0.80, "band": "radio"},
    "VLA_C":        {"nu_Hz": [4.86e9],              "prob": 0.70, "band": "radio"},
    "VLA_X":        {"nu_Hz": [8.46e9],              "prob": 0.50, "band": "radio"},
    "VLA_U":        {"nu_Hz": [14.94e9],             "prob": 0.20, "band": "radio"},
    "VLA_K":        {"nu_Hz": [22.46e9],             "prob": 0.15, "band": "radio"},
    "VLA_Q":        {"nu_Hz": [43.34e9],             "prob": 0.08, "band": "radio"},
    "ALMA_B3":      {"nu_Hz": [100e9],               "prob": 0.10, "band": "submm"},
    "ALMA_B6":      {"nu_Hz": [230e9],               "prob": 0.08, "band": "submm"},
    "ALMA_B7":      {"nu_Hz": [345e9],               "prob": 0.05, "band": "submm"},
    # --- INFRARED ---
    "Spitzer_58":   {"nu_Hz": [5.17e13],             "prob": 0.15, "band": "ir"},
    "Spitzer_36":   {"nu_Hz": [8.33e13],             "prob": 0.20, "band": "ir"},
    "JWST_F444W":   {"nu_Hz": [6.76e13],             "prob": 0.03, "band": "ir"},
    "JWST_F200W":   {"nu_Hz": [1.50e14],             "prob": 0.05, "band": "ir"},
    "Ground_K":     {"nu_Hz": [1.39e14],             "prob": 0.10, "band": "ir"},
    "JWST_F115W":   {"nu_Hz": [2.61e14],             "prob": 0.05, "band": "optical"},
    # --- OPTICAL / UV ---
    "HST_F814W":    {"nu_Hz": [3.68e14],             "prob": 0.35, "band": "optical"},
    "Ground_R":     {"nu_Hz": [4.68e14],             "prob": 0.15, "band": "optical"},
    "HST_F606W":    {"nu_Hz": [4.95e14],             "prob": 0.30, "band": "optical"},
    "HST_F475W":    {"nu_Hz": [6.32e14],             "prob": 0.15, "band": "optical"},
    "HST_F300W":    {"nu_Hz": [9.99e14],             "prob": 0.08, "band": "optical"},
    # --- X-RAY ---
    "Chandra_05":   {"nu_Hz": [1.21e17],             "prob": 0.85, "band": "xray"},
    "Chandra_1":    {"nu_Hz": [2.42e17],             "prob": 0.85, "band": "xray"},
    "Chandra_7":    {"nu_Hz": [1.69e18],             "prob": 0.50, "band": "xray"},
    "XMM_soft":     {"nu_Hz": [1.21e17],             "prob": 0.20, "band": "xray"},
    "XMM_hard":     {"nu_Hz": [2.42e18],             "prob": 0.15, "band": "xray"},
    "NuSTAR":       {"nu_Hz": [4.84e18],             "prob": 0.03, "band": "xray"},
    # --- GAMMA-RAY ---
    "Fermi_100MeV": {"nu_Hz": [2.42e22],             "prob": 0.05, "band": "gamma"},
    "Fermi_10GeV":  {"nu_Hz": [2.42e24],             "prob": 0.03, "band": "gamma"},
}

# Build flat slot arrays
SLOT_NAMES = []
SLOT_FREQUENCIES = []
SLOT_PROBS = []
SLOT_BANDS = []

for name, info in INSTRUMENTS.items():
    for nu in info["nu_Hz"]:
        SLOT_NAMES.append(name)
        SLOT_FREQUENCIES.append(nu)
        SLOT_PROBS.append(info["prob"])
        SLOT_BANDS.append(info["band"])

SLOT_FREQUENCIES = np.array(SLOT_FREQUENCIES)
SLOT_PROBS = np.array(SLOT_PROBS)
SLOT_BANDS = np.array(SLOT_BANDS)
N_SLOTS = len(SLOT_FREQUENCIES)

# Band-dependent fractional noise (σ/flux)
NOISE_CONFIG = {
    "radio":   (0.05, 0.15),   # 5-15% calibration-dominated
    "submm":   (0.08, 0.20),   # 8-20% ALMA calibration + atmosphere
    "ir":      (0.10, 0.30),   # 10-30% background subtraction
    "optical": (0.10, 0.35),   # 10-35% faint knots against host galaxy
    "xray":    (0.20, 0.50),   # 20-50% low counts, Gaussian approx
    "gamma":   (0.30, 0.60),   # 30-60% very few photons
}

# Probability of a detection being an upper limit, by band
UL_PROB = {
    "radio":   0.02,
    "submm":   0.10,
    "ir":      0.15,
    "optical": 0.25,   # optical jets often undetected → UL
    "xray":    0.10,
    "gamma":   0.60,   # gamma-ray jets rarely detected
}

ALL_MODELS = [MODEL_1A, MODEL_1B, MODEL_2A, MODEL_2B, MODEL_3A, MODEL_3B]


# ==================================================================
#  Parameter sampling (conditional priors)
# ==================================================================

def sample_parameters(rng, n):
    """
    Draw n parameter vectors with physically motivated priors.

    Returns dict of arrays, each shape (n,).
    """
    params = {}

    # Continuous parameters
    params["G0"]    = rng.uniform(3.0, 25.0, n)
    params["p"]     = rng.uniform(1.5, 3.5, n)
    params["theta"] = rng.uniform(2.0, 30.0, n)

    # Conditional: gamma_min log-uniform, gamma_max at least 1 decade above
    log_gmin = rng.uniform(1.0, 4.0, n)
    log_gmax = rng.uniform(log_gmin + 1.0, 8.0)
    params["gamma_min"] = 10.0 ** log_gmin
    params["gamma_max"] = 10.0 ** log_gmax

    # Log-uniform for multi-decade quantities
    params["Rj"] = 10.0 ** rng.uniform(np.log10(0.1 * kpc), np.log10(50 * kpc), n)
    params["Lj"] = 10.0 ** rng.uniform(44.0, 49.0, n)
    params["l"]  = 10.0 ** rng.uniform(np.log10(1 * kpc), np.log10(100 * kpc), n)

    # Fixed for now (could also sample)
    params["q_ratio"] = np.ones(n)
    params["z"]       = rng.uniform(0.1, 4.0, n)
    params["eta_e"]   = 10.0 ** rng.uniform(-2.0, 0.0, n)   # 0.01 to 1.0

    # Discrete model: uniform over 6 models
    params["model"] = rng.choice(ALL_MODELS, n)

    return params


# ==================================================================
#  Instrument selection
# ==================================================================

def sample_instrument_mask(rng, min_points=3, max_points=15):
    """
    Draw a realistic set of observed instrument slots.

    Returns boolean mask of shape (N_SLOTS,).
    """
    # Independent Bernoulli draw per slot
    included = rng.random(N_SLOTS) < SLOT_PROBS

    # Ensure minimum coverage: force VLA 1.4 GHz + at least one Chandra
    if included.sum() < min_points:
        # Find indices for key instruments
        for fallback_name in ["VLA_L", "VLA_C", "Chandra_05", "Chandra_1"]:
            for j, sn in enumerate(SLOT_NAMES):
                if sn == fallback_name:
                    included[j] = True

    # Thin if too many
    if included.sum() > max_points:
        active = np.where(included)[0]
        keep = rng.choice(active, max_points, replace=False)
        included[:] = False
        included[keep] = True

    return included


# ==================================================================
#  Noise and upper limits
# ==================================================================

def add_noise_and_limits(rng, true_flux, mask, min_snr_for_det=2.0):
    """
    Add band-dependent Gaussian noise and mark some points as upper limits.

    Parameters
    ----------
    true_flux : (N_SLOTS,) true νFν values
    mask : (N_SLOTS,) bool, which slots are observed

    Returns
    -------
    obs_flux : (N_SLOTS,) noisy flux (0 where unobserved)
    obs_err  : (N_SLOTS,) 1σ error (0 where unobserved)
    obs_mask : (N_SLOTS,) +1 detection, -1 upper limit, 0 unobserved
    """
    obs_flux = np.zeros(N_SLOTS, dtype=np.float64)
    obs_err  = np.zeros(N_SLOTS, dtype=np.float64)
    obs_mask = np.zeros(N_SLOTS, dtype=np.float32)

    for j in range(N_SLOTS):
        if not mask[j]:
            continue

        band = SLOT_BANDS[j]
        frac_lo, frac_hi = NOISE_CONFIG[band]

        # Random fractional error for this observation
        frac_err = rng.uniform(frac_lo, frac_hi)
        sigma = frac_err * abs(true_flux[j])
        sigma = max(sigma, 1e-300)  # guard

        # Add Gaussian noise
        noisy = true_flux[j] + rng.normal(0, sigma)

        # Decide if this is an upper limit
        is_ul = rng.random() < UL_PROB[band]

        # Also: if SNR < threshold, force upper limit (realistic)
        snr = abs(noisy) / sigma if sigma > 0 else 0.0
        if snr < min_snr_for_det:
            is_ul = True

        if is_ul:
            # Upper limit: report the ~2σ level above background
            obs_flux[j] = abs(noisy) + rng.uniform(1.0, 2.5) * sigma
            obs_err[j]  = sigma
            obs_mask[j] = -1.0
        else:
            obs_flux[j] = max(noisy, 1e-300)
            obs_err[j]  = sigma
            obs_mask[j] = +1.0

    return obs_flux, obs_err, obs_mask


# ==================================================================
#  Main generation loop
# ==================================================================

def generate_bank(n_sims, output_path, nx=48, ngamma=64, seed=42,
                  flux_lo=1e-20, flux_hi=1e-10, min_detectable=3,
                  oversample=2.0):
    """
    Generate bank with rejection sampling for physical plausibility.

    Parameters
    ----------
    flux_lo, flux_hi : float
        Plausible νFν range [erg/s/cm²]. SEDs with peak flux outside
        this range are rejected. Typical quasar jet knots have peak
        νFν ~ 1e-16 to 1e-12, but we keep a wider range for diversity.
    min_detectable : int
        Minimum number of slots with flux > 1e-19 (roughly Chandra
        sensitivity limit). Ensures every training SED has enough
        "in principle detectable" points.
    oversample : float
        Draw this many extra parameter sets to compensate for rejects.
    """
    rng = np.random.default_rng(seed)

    # We oversample parameters, then keep only the first n_sims that pass
    n_draw = int(n_sims * oversample)
    print(f"Sampling {n_draw} parameter vectors (target: {n_sims} after cuts)...")
    params = sample_parameters(rng, n_draw)

    cosmo = make_cosmology(71.0, 0.27, 0.73)

    # Output arrays (pre-allocate for n_sims, fill incrementally)
    all_true_flux = np.zeros((n_sims, N_SLOTS), dtype=np.float64)
    all_obs_flux  = np.zeros((n_sims, N_SLOTS), dtype=np.float64)
    all_obs_err   = np.zeros((n_sims, N_SLOTS), dtype=np.float64)
    all_obs_mask  = np.zeros((n_sims, N_SLOTS), dtype=np.float32)
    # Parameter arrays for accepted sims
    accepted_params = {k: np.zeros(n_sims, dtype=params[k].dtype) for k in params}

    nu_jax = jnp.array(SLOT_FREQUENCIES)

    # Warm-up JIT for each model
    print("Warming up JIT (one call per model)...")
    for model in ALL_MODELS:
        p_warmup = make_params(
            G0=10.0, q_ratio=1.0, p=2.5, theta=12.0,
            gamma_min=1e2, gamma_max=1e6,
            Rj=10*kpc, Lj=1e48, l=10*kpc,
            z=2.5, eta_e=0.1, model=int(model),
        )
        _ = observed_sed(nu_jax, p_warmup, cosmo, nx=nx, ngamma=ngamma).block_until_ready()

    print(f"Generating {n_sims} plausible simulations ({N_SLOTS} instrument slots)...")
    print(f"  Flux cuts: [{flux_lo:.0e}, {flux_hi:.0e}] erg/s/cm²")
    print(f"  Min detectable slots: {min_detectable}")
    t0 = time.perf_counter()

    n_accepted = 0
    n_tried = 0
    n_nan = 0
    n_flux_lo = 0
    n_flux_hi = 0
    n_too_few = 0
    draw_idx = 0

    while n_accepted < n_sims:
        # If we exhaust the initial draw, sample more
        if draw_idx >= n_draw:
            extra = int(n_sims * 0.5)
            print(f"  Exhausted initial draw, sampling {extra} more...")
            extra_params = sample_parameters(rng, extra)
            for k in params:
                params[k] = np.concatenate([params[k], extra_params[k]])
            n_draw += extra

        i = draw_idx
        draw_idx += 1
        n_tried += 1

        # Build JetParams
        p_i = make_params(
            G0=params["G0"][i],
            q_ratio=params["q_ratio"][i],
            p=params["p"][i],
            theta=params["theta"][i],
            gamma_min=params["gamma_min"][i],
            gamma_max=params["gamma_max"][i],
            Rj=params["Rj"][i],
            Lj=params["Lj"][i],
            l=params["l"][i],
            z=params["z"][i],
            eta_e=params["eta_e"][i],
            model=int(params["model"][i]),
        )

        # Compute full SED at all slot frequencies
        try:
            sed = np.asarray(
                observed_sed(nu_jax, p_i, cosmo, nx=nx, ngamma=ngamma).block_until_ready()
            )
        except Exception:
            n_nan += 1
            continue

        # --- Rejection criteria ---

        # 1. NaN or all-zero
        if np.any(np.isnan(sed)) or np.all(sed <= 0):
            n_nan += 1
            continue

        # 2. Peak flux too low (undetectable by any instrument)
        peak_flux = np.max(sed[sed > 0]) if np.any(sed > 0) else 0
        if peak_flux < flux_lo:
            n_flux_lo += 1
            continue

        # 3. Peak flux too high (unphysical for a resolved jet feature)
        if peak_flux > flux_hi:
            n_flux_hi += 1
            continue

        # 4. Too few slots with detectable flux
        n_detectable = np.sum(sed > flux_lo)
        if n_detectable < min_detectable:
            n_too_few += 1
            continue

        # --- Accepted! ---
        j = n_accepted

        all_true_flux[j] = sed

        # Store accepted parameters
        for k in params:
            accepted_params[k][j] = params[k][i]

        # Draw instrument mask
        inst_mask = sample_instrument_mask(rng)

        # Add noise and upper limits
        obs_flux, obs_err, obs_mask = add_noise_and_limits(rng, sed, inst_mask)
        all_obs_flux[j] = obs_flux
        all_obs_err[j]  = obs_err
        all_obs_mask[j] = obs_mask

        n_accepted += 1

        # Progress
        if n_accepted % 500 == 0:
            elapsed = time.perf_counter() - t0
            rate = n_accepted / elapsed
            eta = (n_sims - n_accepted) / rate
            accept_rate = n_accepted / n_tried
            print(f"  [{n_accepted:>7d}/{n_sims}]  "
                  f"{rate:.1f} sims/s  ETA {eta/60:.1f}min  "
                  f"accept {accept_rate:.0%}  "
                  f"(lo={n_flux_lo} hi={n_flux_hi} "
                  f"few={n_too_few} nan={n_nan})")

    elapsed = time.perf_counter() - t0
    accept_rate = n_accepted / n_tried
    print(f"\nDone: {n_accepted} accepted / {n_tried} tried "
          f"({accept_rate:.0%} accept rate, {elapsed:.0f}s)")
    print(f"  Rejected — too faint: {n_flux_lo}, too bright: {n_flux_hi}, "
          f"too few slots: {n_too_few}, NaN/zero: {n_nan}")

    # --- Write HDF5 ---
    print(f"Writing {output_path}...")
    with h5py.File(output_path, "w") as f:
        # Instrument catalog (metadata)
        f.create_dataset("slot_frequencies", data=SLOT_FREQUENCIES)
        f.attrs["n_slots"] = N_SLOTS
        f.attrs["n_sims"] = n_sims
        f.attrs["nx"] = nx
        f.attrs["ngamma"] = ngamma
        f.attrs["seed"] = seed
        f.attrs["flux_lo"] = flux_lo
        f.attrs["flux_hi"] = flux_hi
        f.attrs["accept_rate"] = accept_rate

        # Store slot names as variable-length strings
        dt = h5py.string_dtype()
        f.create_dataset("slot_names", data=np.array(SLOT_NAMES, dtype=object), dtype=dt)
        f.create_dataset("slot_bands", data=np.array(SLOT_BANDS, dtype=object), dtype=dt)

        # SEDs
        f.create_dataset("true_flux", data=all_true_flux)      # (N, N_SLOTS) raw νFν
        f.create_dataset("obs_flux",  data=all_obs_flux)       # (N, N_SLOTS) noisy νFν
        f.create_dataset("obs_err",   data=all_obs_err)        # (N, N_SLOTS) 1σ errors
        f.create_dataset("obs_mask",  data=all_obs_mask)       # (N, N_SLOTS) +1/-1/0

        # Parameters (only accepted ones!)
        grp = f.create_group("params")
        for key in ["G0", "p", "theta", "gamma_min", "gamma_max",
                     "Rj", "Lj", "l", "q_ratio", "z", "eta_e"]:
            grp.create_dataset(key, data=accepted_params[key])
        grp.create_dataset("model", data=accepted_params["model"].astype(np.int32))

    size_mb = os.path.getsize(output_path) / 1e6
    print(f"Saved {output_path}  ({size_mb:.1f} MB)")


# ==================================================================
#  CLI
# ==================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SBI simulation bank")
    parser.add_argument("--n_sims", type=int, default=100_000)
    parser.add_argument("--output", type=str, default="bank.h5")
    parser.add_argument("--nx", type=int, default=48)
    parser.add_argument("--ngamma", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--flux_lo", type=float, default=1e-20,
                        help="Min peak νFν [erg/s/cm²] to accept (default 1e-20)")
    parser.add_argument("--flux_hi", type=float, default=1e-10,
                        help="Max peak νFν [erg/s/cm²] to accept (default 1e-10)")
    parser.add_argument("--min_detectable", type=int, default=3,
                        help="Min slots with flux > flux_lo (default 3)")
    parser.add_argument("--oversample", type=float, default=2.0,
                        help="Oversample factor for rejection (default 2.0)")
    args = parser.parse_args()

    generate_bank(args.n_sims, args.output, args.nx, args.ngamma, args.seed,
                  flux_lo=args.flux_lo, flux_hi=args.flux_hi,
                  min_detectable=args.min_detectable,
                  oversample=args.oversample)
