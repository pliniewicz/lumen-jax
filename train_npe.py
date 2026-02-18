#!/usr/bin/env python
"""
Train NPE (Neural Posterior Estimation) on Lumen simulation bank.

Reads the HDF5 file produced by generate_bank.py, encodes observations
into fixed-length vectors, and trains a normalizing flow to estimate
p(θ | observation).

Usage:
    python train_npe.py --bank bank.h5 --epochs 300
    python train_npe.py --bank bank.h5 --epochs 300 --device cuda

Requirements:
    pip install sbi torch h5py numpy
"""

import argparse
import time
import numpy as np
import h5py
import torch
import torch.nn as nn
from pathlib import Path


# ==================================================================
#  1. Data loading and encoding
# ==================================================================

# Parameter names in the order they appear in the θ vector
CONTINUOUS_PARAM_NAMES = [
    "G0", "p", "theta",
    "log10_gamma_min", "log10_gamma_max",
    "log10_Rj", "log10_Lj", "log10_l",
    "z", "log10_eta_e",
]
MODEL_NAMES = ["1A", "1B", "2A", "2B", "3A", "3B"]
N_MODELS = 6


def load_bank(h5_path):
    """
    Load simulation bank from HDF5.

    Returns
    -------
    theta : (N, D_theta) float32 — continuous params (log-scaled) + model one-hot
    x     : (N, 3*N_SLOTS) float32 — encoded observations
    meta  : dict with slot info
    """
    print(f"Loading {h5_path}...")
    with h5py.File(h5_path, "r") as f:
        n_sims = f.attrs["n_sims"]
        n_slots = f.attrs["n_slots"]
        slot_freqs = f["slot_frequencies"][:]

        # Raw arrays
        obs_flux = f["obs_flux"][:]     # (N, n_slots), linear νFν
        obs_err  = f["obs_err"][:]      # (N, n_slots)
        obs_mask = f["obs_mask"][:]     # (N, n_slots) +1/-1/0

        # Parameters
        p = f["params"]
        G0        = p["G0"][:]
        p_idx     = p["p"][:]
        theta_deg = p["theta"][:]
        gmin      = p["gamma_min"][:]
        gmax      = p["gamma_max"][:]
        Rj        = p["Rj"][:]
        Lj        = p["Lj"][:]
        l         = p["l"][:]
        z         = p["z"][:]
        eta_e     = p["eta_e"][:]
        model     = p["model"][:].astype(int)

    # --- Build θ vector ---
    # Continuous: log-scale where appropriate
    theta_cont = np.column_stack([
        G0,
        p_idx,
        theta_deg,
        np.log10(gmin),
        np.log10(gmax),
        np.log10(Rj),
        np.log10(Lj),
        np.log10(l),
        z,
        np.log10(eta_e),
    ]).astype(np.float32)

    n_cont = theta_cont.shape[1]

    # One-hot model
    model_onehot = np.zeros((n_sims, N_MODELS), dtype=np.float32)
    model_onehot[np.arange(n_sims), model] = 1.0

    theta = np.concatenate([theta_cont, model_onehot], axis=1)  # (N, n_cont+6)

    # --- Encode observations: 3 channels × n_slots ---
    # Channel 0: log10(flux) where observed, 0 otherwise
    # Channel 1: log10(error) where observed, 0 otherwise
    # Channel 2: mask (+1, -1, 0)

    x = np.zeros((n_sims, 3 * n_slots), dtype=np.float32)

    for i in range(n_sims):
        for j in range(n_slots):
            if obs_mask[i, j] != 0:
                flux_val = max(obs_flux[i, j], 1e-300)
                err_val  = max(obs_err[i, j], 1e-300)
                x[i, j]             = np.log10(flux_val)  # channel 0
                x[i, n_slots + j]   = np.log10(err_val)   # channel 1
                x[i, 2*n_slots + j] = obs_mask[i, j]      # channel 2

    # Filter out failed sims (all zeros)
    good = np.any(obs_mask != 0, axis=1)
    theta = theta[good]
    x = x[good]
    print(f"  {good.sum()} / {n_sims} valid simulations loaded")
    print(f"  θ shape: {theta.shape}  ({n_cont} continuous + {N_MODELS} model one-hot)")
    print(f"  x shape: {x.shape}  (3 × {n_slots} slots)")

    meta = {
        "n_slots": n_slots,
        "slot_freqs": slot_freqs,
        "n_cont": n_cont,
        "param_names": CONTINUOUS_PARAM_NAMES + [f"M_{m}" for m in MODEL_NAMES],
    }

    return theta, x, meta


# ==================================================================
#  2. Embedding network
# ==================================================================

class SlotEmbedding(nn.Module):
    """
    Process the 3-channel instrument-slot observation into a
    fixed-size embedding.

    Input:  (batch, 3 * n_slots)
    Output: (batch, embed_dim)

    Uses 1D convolutions that treat the frequency axis as spatial,
    so neighboring slots share information. The fixed ordering means
    the network learns instrument-specific features positionally.
    """

    def __init__(self, n_slots, embed_dim=128):
        super().__init__()
        self.n_slots = n_slots
        self.conv = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(8),
            nn.Flatten(),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 8, 256),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(256, embed_dim),
        )

    def forward(self, x):
        # x: (batch, 3*n_slots) → reshape to (batch, 3, n_slots)
        batch = x.shape[0]
        x3 = x.reshape(batch, 3, self.n_slots)
        h = self.conv(x3)
        return self.fc(h)


# ==================================================================
#  3. Training
# ==================================================================

def train_npe(theta, x, meta, embed_dim=128, n_transforms=6,
              hidden_features=128, batch_size=256, lr=5e-4,
              max_epochs=300, val_frac=0.05, device="cpu",
              save_path="npe_posterior.pt"):
    """
    Train NPE with neural spline flow and custom embedding.
    """
    from sbi.inference import NPE
    from sbi.neural_nets import posterior_nn
    from sbi.utils import BoxUniform

    n_slots = meta["n_slots"]

    # Build embedding net
    embedding = SlotEmbedding(n_slots=n_slots, embed_dim=embed_dim)

    # Build density estimator (zuko NSF — nflows is unmaintained)
    density_estimator = posterior_nn(
        model="zuko_nsf",
        embedding_net=embedding,
        hidden_features=hidden_features,
        num_transforms=n_transforms,
    )

    # To tensors
    theta_t = torch.tensor(theta, dtype=torch.float32)
    x_t = torch.tensor(x, dtype=torch.float32)

    # sbi requires a prior; build a wide BoxUniform that covers the
    # training data range (the prior is only used for posterior sampling
    # bounds, not for the loss, since we use pre-simulated data).
    theta_min = theta_t.min(dim=0).values - 0.5
    theta_max = theta_t.max(dim=0).values + 0.5
    prior = BoxUniform(low=theta_min, high=theta_max)

    # Train/val split
    n_val = max(int(len(theta) * val_frac), 1000)
    n_train = len(theta) - n_val

    print(f"\nTraining NPE:")
    print(f"  {n_train} training / {n_val} validation examples")
    print(f"  θ dimension: {theta.shape[1]}")
    print(f"  x dimension: {x.shape[1]}")
    print(f"  Embedding: {n_slots} slots → {embed_dim}d")
    print(f"  Flow: {n_transforms} zuko_nsf transforms, {hidden_features} hidden")
    print(f"  Device: {device}")
    print()

    # NPE — prior is required in modern sbi
    inference = NPE(
        prior=prior,
        density_estimator=density_estimator,
        device=device,
    )
    inference.append_simulations(theta_t, x_t)

    t0 = time.perf_counter()
    density_estimator = inference.train(
        training_batch_size=batch_size,
        learning_rate=lr,
        max_num_epochs=max_epochs,
        stop_after_epochs=30,       # early stopping patience
        show_train_summary=True,
    )
    elapsed = time.perf_counter() - t0
    print(f"\nTraining completed in {elapsed/60:.1f} min")

    # Build posterior object
    posterior = inference.build_posterior(density_estimator)

    # Save
    torch.save({
        "posterior_state": density_estimator.state_dict(),
        "meta": meta,
        "embed_dim": embed_dim,
        "n_transforms": n_transforms,
        "hidden_features": hidden_features,
    }, save_path)
    print(f"Saved trained model to {save_path}")

    return posterior


# ==================================================================
#  4. Inference helpers
# ==================================================================

def encode_real_observation(slot_freqs, nu_obs, flux_obs, err_obs,
                            is_upper):
    """
    Encode real sparse observations into the fixed-slot format.

    Parameters
    ----------
    slot_freqs : (N_SLOTS,) instrument slot frequencies [Hz]
    nu_obs     : (M,) observed frequencies [Hz]
    flux_obs   : (M,) νFν [erg/s/cm²]
    err_obs    : (M,) σ on νFν
    is_upper   : (M,) bool array

    Returns
    -------
    x : (1, 3*N_SLOTS) tensor
    """
    n_slots = len(slot_freqs)
    x = np.zeros(3 * n_slots, dtype=np.float32)
    log_slots = np.log10(slot_freqs)

    for j in range(len(nu_obs)):
        i = np.argmin(np.abs(log_slots - np.log10(nu_obs[j])))

        x[i]             = np.log10(max(flux_obs[j], 1e-300))
        x[n_slots + i]   = np.log10(max(err_obs[j], 1e-300))
        x[2*n_slots + i] = -1.0 if is_upper[j] else +1.0

    return torch.tensor(x, dtype=torch.float32).unsqueeze(0)


def run_posterior(posterior, x_obs, n_samples=50_000):
    """
    Sample from posterior and parse results.

    Returns
    -------
    dict with keys: samples, medians, ci_lo, ci_hi, model_probs, best_model
    """
    samples = posterior.sample((n_samples,), x=x_obs).numpy()

    n_cont = samples.shape[1] - N_MODELS
    continuous = samples[:, :n_cont]
    model_logits = samples[:, n_cont:]

    # Model probabilities from posterior samples
    model_ids = np.argmax(model_logits, axis=1)
    from collections import Counter
    counts = Counter(model_ids)
    model_probs = {
        int(mid): count / len(model_ids)
        for mid, count in sorted(counts.items())
    }

    # Continuous param summaries
    medians = np.median(continuous, axis=0)
    ci_lo = np.percentile(continuous, 16, axis=0)
    ci_hi = np.percentile(continuous, 84, axis=0)

    return {
        "samples": samples,
        "medians": medians,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "model_probs": model_probs,
        "best_model": max(model_probs, key=model_probs.get),
    }


def print_results(results, param_names, true_theta=None):
    """Pretty-print posterior summary."""
    n_cont = len(results["medians"])
    names = param_names[:n_cont]

    print("\n" + "=" * 65)
    print("  POSTERIOR SUMMARY")
    print("=" * 65)

    for i, name in enumerate(names):
        med = results["medians"][i]
        lo = results["ci_lo"][i]
        hi = results["ci_hi"][i]
        true_str = f"  (true: {true_theta[i]:.3f})" if true_theta is not None else ""
        print(f"  {name:>18s}: {med:8.3f}  [{lo:.3f}, {hi:.3f}]{true_str}")

    print(f"\n  Model probabilities:")
    for mid, prob in sorted(results["model_probs"].items()):
        tag = ""
        if true_theta is not None:
            true_model = int(np.argmax(true_theta[n_cont:n_cont+N_MODELS]))
            if mid == true_model:
                tag = " ← true"
        print(f"    Model {MODEL_NAMES[mid]}: {prob:.1%}{tag}")

    print(f"\n  Best model: {MODEL_NAMES[results['best_model']]}")


# ==================================================================
#  5. Validation on held-out simulations
# ==================================================================

def validate(posterior, theta, x, meta, n_test=100, n_samples=10_000):
    """
    Run posterior on held-out examples and compute coverage.
    """
    n_cont = meta["n_cont"]
    rng = np.random.default_rng(999)
    idx = rng.choice(len(theta), n_test, replace=False)

    coverages_68 = np.zeros(n_cont)
    coverages_95 = np.zeros(n_cont)
    model_correct = 0

    for i in idx:
        x_obs = torch.tensor(x[i:i+1], dtype=torch.float32)
        samples = posterior.sample((n_samples,), x=x_obs).numpy()

        cont_samples = samples[:, :n_cont]
        true_cont = theta[i, :n_cont]

        for j in range(n_cont):
            lo68 = np.percentile(cont_samples[:, j], 16)
            hi68 = np.percentile(cont_samples[:, j], 84)
            lo95 = np.percentile(cont_samples[:, j], 2.5)
            hi95 = np.percentile(cont_samples[:, j], 97.5)

            if lo68 <= true_cont[j] <= hi68:
                coverages_68[j] += 1
            if lo95 <= true_cont[j] <= hi95:
                coverages_95[j] += 1

        # Model accuracy
        model_logits = samples[:, n_cont:]
        pred_model = np.argmax(np.mean(model_logits, axis=0))
        true_model = np.argmax(theta[i, n_cont:])
        if pred_model == true_model:
            model_correct += 1

    coverages_68 /= n_test
    coverages_95 /= n_test

    print("\n" + "=" * 65)
    print(f"  CALIBRATION CHECK ({n_test} held-out sims)")
    print("=" * 65)
    names = meta["param_names"][:n_cont]
    for j, name in enumerate(names):
        c68 = coverages_68[j]
        c95 = coverages_95[j]
        flag68 = " ⚠" if abs(c68 - 0.68) > 0.10 else " ✓"
        flag95 = " ⚠" if abs(c95 - 0.95) > 0.10 else " ✓"
        print(f"  {name:>18s}:  68% CI covers {c68:.0%}{flag68}   "
              f"95% CI covers {c95:.0%}{flag95}")

    print(f"\n  Model accuracy: {model_correct}/{n_test} = "
          f"{model_correct/n_test:.0%}")
    print("  (Ideal: 68% CI covers 68%, 95% CI covers 95%)")


# ==================================================================
#  6. CLI
# ==================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NPE for Lumen SBI")
    parser.add_argument("--bank", default="bank.h5", help="HDF5 simulation bank")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--n_transforms", type=int, default=6)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--validate", action="store_true",
                        help="Run calibration check after training")
    parser.add_argument("--save", default="npe_posterior.pt")
    args = parser.parse_args()

    # Load
    theta, x, meta = load_bank(args.bank)

    # Train
    posterior = train_npe(
        theta, x, meta,
        embed_dim=args.embed_dim,
        n_transforms=args.n_transforms,
        batch_size=args.batch_size,
        lr=args.lr,
        max_epochs=args.epochs,
        device=args.device,
        save_path=args.save,
    )

    # Validate
    if args.validate:
        validate(posterior, theta, x, meta)

    # Demo: inference on last held-out sim
    print("\n--- Demo inference on one held-out sim ---")
    demo_idx = -1
    x_obs = torch.tensor(x[demo_idx:], dtype=torch.float32)
    results = run_posterior(posterior, x_obs)
    print_results(results, meta["param_names"], true_theta=theta[demo_idx])
