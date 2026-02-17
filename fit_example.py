#!/usr/bin/env python
"""
Example: synchrotron SED computation and gradient-based fitting with Lumen-JAX.

Demonstrates:
  1. Forward model evaluation (simulation mode)
  2. Autodiff of χ² w.r.t. jet parameters
  3. Gradient-based optimization via scipy (L-BFGS-B)
  4. Easy transition to HMC/NUTS sampling (sketch)

Usage:
    python fit_example.py
"""

import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

import lumen
from lumen import (
    make_params, JetParams,
    MODEL_1A, MODEL_1B, MODEL_NAMES,
    kpc, observed_synchrotron, synchrotron_spectrum,
    make_cosmology,
    chi_squared, SEDData, SEDDataPoint,
)


# ==================================================================
#  1.  SIMULATION MODE — forward model
# ==================================================================

print("=" * 60)
print("  Lumen-JAX: Synchrotron SED skeleton")
print("=" * 60)

# Define jet parameters
params = make_params(
    G0=10.0,
    q_ratio=1.0,
    p=2.5,
    theta=12.0,         # degrees
    gamma_min=1e2,
    gamma_max=1e6,
    Rj=10 * kpc,
    Lj=1e48,            # erg/s
    l=10 * kpc,
    z=2.5,
    eta_e=0.1,
    model=MODEL_1B,
)

cosmo = make_cosmology(71.0, 0.27, 0.73)

# Frequency grid: radio to X-ray
nu_array = jnp.logspace(8, 20, 100)

print("\nComputing synchrotron spectrum (first call compiles)...")
import time
t0 = time.time()
nuFnu = observed_synchrotron(nu_array, params, cosmo, nx=48, ngamma=64)
t1 = time.time()
print(f"  First call (compile + run): {t1 - t0:.2f}s")

t0 = time.time()
nuFnu = observed_synchrotron(nu_array, params, cosmo, nx=48, ngamma=64)
t1 = time.time()
print(f"  Second call (cached JIT):   {t1 - t0:.4f}s")

print(f"\n  ν range:   {nu_array[0]:.1e} – {nu_array[-1]:.1e} Hz")
print(f"  νFν range: {float(jnp.min(nuFnu[nuFnu > 0])):.2e} – {float(jnp.max(nuFnu)):.2e} erg/s/cm²")


# ==================================================================
#  2.  AUTODIFF — gradients of χ² for free
# ==================================================================

print("\n" + "=" * 60)
print("  Autodiff demo: ∂χ²/∂θ")
print("=" * 60)

# Create some synthetic "data" by adding noise to the model
key = jax.random.PRNGKey(42)
noise = 0.2 * nuFnu * jax.random.normal(key, shape=nuFnu.shape)
fake_data_flux = nuFnu + noise
fake_data_err = 0.15 * jnp.abs(nuFnu) + 1e-20

# Build an SEDData object at a subset of frequencies
idx = jnp.array([10, 30, 50, 70, 90])  # 5 "data points"
data = SEDData([
    SEDDataPoint(
        float(nu_array[i]),
        float(fake_data_flux[i]),
        float(fake_data_err[i]),
    )
    for i in idx
])

# Define the objective as a function of a flat parameter vector u
# u = [log10(Lj), G0, theta]
# This is the function we'll differentiate through.

def objective(u):
    """χ² as a function of free parameters u = [log10(Lj), G0, theta]."""
    log_Lj, G0, theta = u

    p = make_params(
        G0=G0,
        q_ratio=1.0,
        p=2.5,
        theta=theta,
        gamma_min=1e2,
        gamma_max=1e6,
        Rj=10 * kpc,
        Lj=10.0 ** log_Lj,
        l=10 * kpc,
        z=2.5,
        eta_e=0.1,
        model=MODEL_1B,
    )

    model_flux = observed_synchrotron(data.frequencies, p, cosmo, nx=32, ngamma=48)
    return chi_squared(model_flux, data)


# Compute χ² and its gradient simultaneously
u_init = jnp.array([48.0, 10.0, 12.0])

print("\nComputing χ² and ∇χ² (first call compiles)...")
t0 = time.time()
chi2_val, grad_val = jax.value_and_grad(objective)(u_init)
t1 = time.time()
print(f"  Compile + run: {t1 - t0:.2f}s")

t0 = time.time()
chi2_val, grad_val = jax.value_and_grad(objective)(u_init)
t1 = time.time()
print(f"  Cached:        {t1 - t0:.4f}s")

print(f"\n  χ² = {float(chi2_val):.4f}")
print(f"  ∂χ²/∂log₁₀(Lj) = {float(grad_val[0]):.4e}")
print(f"  ∂χ²/∂G0         = {float(grad_val[1]):.4e}")
print(f"  ∂χ²/∂θ          = {float(grad_val[2]):.4e}")


# ==================================================================
#  3.  GRADIENT-BASED OPTIMIZATION (scipy L-BFGS-B)
# ==================================================================

print("\n" + "=" * 60)
print("  Gradient-based optimization: L-BFGS-B via jaxopt")
print("=" * 60)

try:
    from jaxopt import ScipyBoundedMinimize

    # Perturb the starting point away from truth
    u_start = jnp.array([47.5, 8.0, 15.0])

    solver = ScipyBoundedMinimize(
        fun=objective,
        method="L-BFGS-B",
        maxiter=200,
        tol=1e-8,
    )

    bounds_lo = jnp.array([45.0, 5.0, 5.0])
    bounds_hi = jnp.array([49.0, 25.0, 35.0])

    print(f"  Starting from: log10(Lj)={float(u_start[0])}, G0={float(u_start[1])}, θ={float(u_start[2])}")

    t0 = time.time()
    result = solver.run(u_start, bounds=(bounds_lo, bounds_hi))
    t1 = time.time()

    u_best = result.params
    print(f"\n  Converged in {t1 - t0:.2f}s")
    print(f"  log10(Lj) = {float(u_best[0]):.4f}  (true: 48.0)")
    print(f"  G0        = {float(u_best[1]):.4f}  (true: 10.0)")
    print(f"  θ         = {float(u_best[2]):.4f}  (true: 12.0)")
    print(f"  χ²        = {float(objective(u_best)):.4f}")

except ImportError:
    print("  [jaxopt not installed — pip install jaxopt]")
    print("  Skipping gradient-based optimization demo.")


# ==================================================================
#  4.  SKETCH: NUTS/HMC sampling (requires numpyro or blackjax)
# ==================================================================

print("\n" + "=" * 60)
print("  Next steps: MCMC sampling")
print("=" * 60)

print("""
  With the differentiable χ² above, you can plug directly into:

  • numpyro:   NUTS sampler with automatic mass matrix adaptation
               import numpyro; numpyro.infer.MCMC(numpyro.infer.NUTS(...))

  • blackjax:  Lightweight JAX-native HMC/NUTS
               import blackjax; blackjax.nuts(log_prob, ...)

  • flowMC:    Normalizing-flow enhanced MCMC for multimodal posteriors
               Ideal for the degeneracies in jet parameter estimation.

  • sbi:       Simulation-based inference (amortized posterior estimation)
               Train a neural density estimator on Lumen simulations.

  All of these benefit from the exact gradients JAX provides.
  No finite-difference approximations needed.
""")

print("Done.")
