#!/usr/bin/env python
"""Quick test: synchrotron + IC/CMB SED."""

import jax.numpy as jnp
import matplotlib.pyplot as plt

from lumen import (
    make_params, make_cosmology,
    observed_synchrotron, observed_ic, observed_sed,
    MODEL_1A, MODEL_1B, MODEL_2A, MODEL_2B, MODEL_3A, MODEL_3B, 
    kpc,
)

params = make_params(
    G0=10.0, q_ratio=1.0, p=2.5, theta=12.0,
    gamma_min=1e2, gamma_max=1e6,
    Rj=10*kpc, Lj=1e48, l=10*kpc,
    z=2.5, eta_e=0.1, model=MODEL_1B,
)
cosmo = make_cosmology(71.0, 0.27, 0.73)

# nu = jnp.logspace(8, 26, 300)
NX, NG = 256, 256

# nuFnu_syn = observed_synchrotron(nu, params, cosmo, nx=NX, ngamma=NG)
# nuFnu_ic  = observed_ic(nu, params, cosmo, nx=NX, ngamma=NG)
# nuFnu_tot = observed_sed(nu, params, cosmo, nx=NX, ngamma=NG)

# fig, ax = plt.subplots(figsize=(9, 5.5))
# ax.loglog(nu, nuFnu_tot, 'k-',  lw=2.2, label='Total')
# ax.loglog(nu, nuFnu_syn, 'b--', lw=1.4, label='Synchrotron')
# ax.loglog(nu, nuFnu_ic,  'b-', lw=1.4, label='IC/CMB')
# ax.set_xlabel(r'$\nu$ [Hz]')
# ax.set_ylabel(r'$\nu F_\nu$ [erg/s/cm$^2$]')
# ax.set_title('Lumen-JAX: synchrotron + IC/CMB')
# ax.set_ylim(bottom=1e-20, top=1e-12)
# ax.legend()
# fig.tight_layout()
# fig.savefig('sed_test.pdf', dpi=150)
# print('Saved sed_test.pdf')

# --- Benchmark ---
import time

# nu_bench = jnp.logspace(8, 26, 200)
#
# # Warm up (compile)
# _ = observed_sed(nu_bench, params, cosmo, nx=NX, ngamma=NG).block_until_ready()
#
# N_runs = 1000
# t0 = time.perf_counter()
# for _ in range(N_runs):
#     observed_sed(nu_bench, params, cosmo, nx=NX, ngamma=NG).block_until_ready()
# t1 = time.perf_counter()
# dt = (t1 - t0) / N_runs
#
# print(f"\nBenchmark: full SED (synch+IC), 200 freqs, nx={NX}, nγ={NG}")
# print(f"  {dt*1e3:.2f} ms per evaluation  ({N_runs} runs averaged)")

# --- Bulk simulation → HDF5 ---
import h5py
import numpy as np
import sys

N_sims = int(sys.argv[1]) if len(sys.argv) > 1 else 50
nu_out = jnp.logspace(3, 26, 200)

# Parameter ranges: (name, lo, hi, log-scale?)
param_ranges = [
    ("G0",        3.0,    25.0,   False),
    ("p",         2.0,    3.5,    False),
    ("theta",     5.0,    40.0,   False),
    ("gamma_min", 1e1,    1e4,    True),
    ("gamma_max", 1e4,    1e8,    True),
    ("Rj",        0.1*kpc, 50*kpc, True),
    ("Lj",        1e44,   1e49,   True),
    ("l",         1*kpc,  100*kpc, True),
    ("z",         0,        6,    False),
]

# Fixed params not being sampled
fixed_q_ratio = 1.0
# fixed_z = 2.5
fixed_eta_e = 0.1
fixed_model = MODEL_1B

rng = np.random.default_rng(seed=12345)

# Draw samples
samples = {}
for name, lo, hi, logscale in param_ranges:
    if logscale:
        samples[name] = 10 ** rng.uniform(np.log10(lo), np.log10(hi), N_sims)
    else:
        samples[name] = rng.uniform(lo, hi, N_sims)

log_gamma_min = rng.uniform(1, 3, N_sims) 
log_gamma_gap = 1
log_gamma_max = rng.uniform(log_gamma_min + log_gamma_gap, 6) 

samples["gamma_min"] = 10 ** log_gamma_min
samples["gamma_max"] = 10 ** log_gamma_max

models = [MODEL_1A, MODEL_1B, MODEL_2A, MODEL_2B, MODEL_3A, MODEL_3B]
samples["model"] = rng.choice(models, N_sims)

print(f"\nBulk run: {N_sims} simulations, {len(nu_out)} freqs each...")
all_seds = np.zeros((N_sims, len(nu_out)))

t0 = time.perf_counter()
for i in range(N_sims):
    p_i = make_params(
        G0=samples["G0"][i],
        q_ratio=fixed_q_ratio,
        p=samples["p"][i],
        theta=samples["theta"][i],
        gamma_min=samples["gamma_min"][i],
        gamma_max=samples["gamma_max"][i],
        Rj=samples["Rj"][i],
        Lj=samples["Lj"][i],
        l=samples["l"][i],
        z=samples["z"][i],
        # z=fixed_z,
        eta_e=fixed_eta_e,
        # model=fixed_model,
        model=int(samples["model"][i])
    )
    sed_i = observed_sed(nu_out, p_i, cosmo, nx=NX, ngamma=NG).block_until_ready()
    all_seds[i] = np.asarray(sed_i)

dt_bulk = time.perf_counter() - t0
print(f"  Done in {dt_bulk:.1f}s  ({dt_bulk/N_sims*1e3:.1f} ms/sim)")

# Write HDF5
outfile = "bulk_seds.h5"
with h5py.File(outfile, "w") as f:
    f.create_dataset("nu", data=np.asarray(nu_out))
    f.create_dataset("nuFnu", data=all_seds)
    # Store each sampled parameter
    grp = f.create_group("params")
    for name, _, _, _ in param_ranges:
        grp.create_dataset(name, data=samples[name])
    grp.create_dataset("model", data=samples["model"].astype(np.int32))
    # Store fixed values as attributes
    grp.attrs["q_ratio"] = fixed_q_ratio
    # grp.attrs["z"] = fixed_z
    grp.attrs["eta_e"] = fixed_eta_e
    # grp.attrs["model"] = fixed_model

print(f"  Saved {outfile}  ({N_sims} SEDs × {len(nu_out)} freqs)")
