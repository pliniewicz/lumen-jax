"""
Jet profiles: velocity, magnetic field, pressure, Doppler factor.

Mirrors profiles.jl. All functions are JAX-traceable w.r.t. continuous
parameters (x, theta, G0, etc.).  The `model` field is treated as a
**static** integer — switching models triggers re-compilation, which
is the intended workflow (model is fixed per fit run).

Convention: x ∈ [0, 1] is the normalized radial coordinate inside the jet.
"""

import jax
import jax.numpy as jnp
from functools import partial

from .constants import c
from .integration import get_gl, map_to_interval, gl_quad


# ------------------------------------------------------------------ #
#  Core profiles                                                      #
# ------------------------------------------------------------------ #

def Gamma_profile(x, G0):
    """Bulk Lorentz factor Γ(x) = 1 + (G0 - 1)(1 - x²)."""
    return 1.0 + (G0 - 1.0) * (1.0 - x * x)


def b_profile(x, model: int):
    """
    Magnetic field profile b(x).

    model 0,1 (1A/1B):  b = x
    model 2,3 (2A/2B):  b = 10x² / (1 + 9x⁸)
    model 4,5 (3A/3B):  b = 151x / (1 + 150x⁴)

    NOTE: `model` must be a plain Python int, not a JAX tracer.
    This is enforced by marking it static in the JIT boundary.
    """
    x = jnp.clip(x, 0.0, 1.0)
    model = int(model)  # ensure Python int (not tracer)
    if model in (0, 1):
        return x
    elif model in (2, 3):
        return 10.0 * x**2 / (1.0 + 9.0 * x**8)
    elif model in (4, 5):
        return 151.0 * x / (1.0 + 150.0 * x**4)
    else:
        return jnp.ones_like(x)


def f_profile(x, G0, model: int):
    """f(x) = [b(x) / Γ(x)]²."""
    b = b_profile(x, model)
    G = Gamma_profile(x, G0)
    return (b / G) ** 2


def p_profile(x, G0, q_ratio, model: int, n: int = 64):
    """
    Total pressure profile p(x).

    p(x) = 1 + q [ 1 - f(x) + 2 ∫_x^1 f(s)/s ds ]

    Uses Gauss-Legendre quadrature for the integral.
    """
    x = jnp.maximum(x, 1e-15)
    nodes, weights = get_gl(n)
    s_nodes, scale = map_to_interval(nodes, x, 1.0)

    # f(s)/s evaluated at quadrature nodes
    f_vals = f_profile(s_nodes, G0, model) / s_nodes
    integral = gl_quad(f_vals, weights, scale)

    return 1.0 + q_ratio * (1.0 - f_profile(x, G0, model) + 2.0 * integral)


# ------------------------------------------------------------------ #
#  Doppler factor                                                     #
# ------------------------------------------------------------------ #

def lorentz_beta(Gamma):
    """β = √(1 - 1/Γ²)."""
    return jnp.sqrt(1.0 - 1.0 / (Gamma * Gamma))


def doppler(Gamma, theta_deg):
    """Doppler factor δ = 1 / [Γ(1 - β cos θ)]."""
    beta = lorentz_beta(Gamma)
    cos_theta = jnp.cos(jnp.radians(theta_deg))
    return 1.0 / (Gamma * (1.0 - beta * cos_theta))


def Doppler_factor(x, G0, theta_deg):
    """Doppler factor at radial position x."""
    return doppler(Gamma_profile(x, G0), theta_deg)


# ------------------------------------------------------------------ #
#  Magnetic field normalization: PB1, B1                              #
# ------------------------------------------------------------------ #

def _pressure_boundary_integrand(x, G0, q_ratio, model, n_p=64):
    """x · β · Γ² · p(x)  — integrand for PB1."""
    model = int(model)
    Gamma = Gamma_profile(x, G0)
    beta = lorentz_beta(Gamma)
    P = p_profile(x, G0, q_ratio, model, n=n_p)
    return x * beta * Gamma**2 * P


def PB1(params, n: int = 64, n_p: int = 64):
    """
    Magnetic pressure normalization at the jet boundary.

    PB1 = q · Lj / (8π c Rj² · ∫₀¹ x β Γ² p dx)
    """
    nodes, weights = get_gl(n)
    xs, scale = map_to_interval(nodes, 0.0, 1.0)

    # Vectorize over quadrature nodes
    model = int(params.model)
    integrand_fn = jax.vmap(
        lambda x: _pressure_boundary_integrand(
            x, params.G0, params.q_ratio, model, n_p
        )
    )
    f_vals = integrand_fn(xs)
    integral = gl_quad(f_vals, weights, scale)

    return params.q_ratio * params.Lj / (8.0 * jnp.pi * c * params.Rj**2 * integral)


def B1(params, n: int = 64, n_p: int = 64):
    """Magnetic field at boundary: B1 = √(8π PB1)."""
    return jnp.sqrt(8.0 * jnp.pi * PB1(params, n, n_p))
