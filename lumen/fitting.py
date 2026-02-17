"""
SED data I/O and χ² computation.

Mirrors fitting.jl.  Data structures are plain Python (not JAX-traced)
since they represent fixed observations.  The χ² function *is*
JAX-traceable w.r.t. model parameters, enabling gradient-based
optimization and HMC sampling.
"""

import numpy as np
import jax.numpy as jnp
from dataclasses import dataclass
from typing import List, Optional


# ------------------------------------------------------------------ #
#  Data containers                                                    #
# ------------------------------------------------------------------ #

@dataclass
class SEDDataPoint:
    nu: float           # frequency [Hz]
    nuFnu: float        # νFν [erg/s/cm²]
    nuFnu_err: float    # error on νFν
    is_upper: bool = False


@dataclass
class SEDData:
    points: List[SEDDataPoint]

    @property
    def frequencies(self):
        return jnp.array([p.nu for p in self.points])

    @property
    def fluxes(self):
        return jnp.array([p.nuFnu for p in self.points])

    @property
    def errors(self):
        return jnp.array([p.nuFnu_err for p in self.points])

    @property
    def upper_limit_mask(self):
        return jnp.array([p.is_upper for p in self.points])

    def detections(self):
        return SEDData([p for p in self.points if not p.is_upper])

    def upper_limits(self):
        return SEDData([p for p in self.points if p.is_upper])

    def __len__(self):
        return len(self.points)


# ------------------------------------------------------------------ #
#  Loading                                                            #
# ------------------------------------------------------------------ #

def load_sed(filename: str, delimiter=',', skip_header: int = 1,
             log_data: bool = False) -> SEDData:
    """
    Load SED from file.  Columns: ν, νFν, νFν_err [, is_upper].

    When log_data=True, input columns are log10 values and errors
    are propagated: σ_lin = 10^val · ln(10) · σ_log.
    """
    raw = np.loadtxt(filename, delimiter=delimiter, skiprows=skip_header)
    n, ncols = raw.shape

    points = []
    for i in range(n):
        is_upper = bool(raw[i, 3] == 1) if ncols >= 4 else False

        if log_data:
            nu = 10.0 ** raw[i, 0]
            nuFnu = 10.0 ** raw[i, 1]
            nuFnu_err = nuFnu * np.log(10) * raw[i, 2] if ncols >= 3 else 0.1 * nuFnu
        else:
            nu = raw[i, 0]
            nuFnu = raw[i, 1]
            nuFnu_err = raw[i, 2] if ncols >= 3 else 0.1 * nuFnu

        points.append(SEDDataPoint(nu, nuFnu, nuFnu_err, is_upper))

    return SEDData(points)


def load_sed_Fnu(filename: str, unit: str = 'mJy',
                 delimiter=',', skip_header: int = 1) -> SEDData:
    """
    Load SED with Fν columns.  Columns: ν [Hz], Fν, ΔFν [, is_upper].
    Converts to νFν internally.
    """
    from .cosmology import Fnu_to_nuFnu

    raw = np.loadtxt(filename, delimiter=delimiter, skiprows=skip_header)
    n, ncols = raw.shape

    points = []
    for i in range(n):
        nu = raw[i, 0]
        Fnu = raw[i, 1]
        dFnu = raw[i, 2] if ncols >= 3 else 0.1 * Fnu
        is_upper = bool(raw[i, 3] == 1) if ncols >= 4 else False

        nuFnu = float(Fnu_to_nuFnu(nu, Fnu, unit))
        nuFnu_err = float(Fnu_to_nuFnu(nu, dFnu, unit))
        points.append(SEDDataPoint(nu, nuFnu, nuFnu_err, is_upper))

    return SEDData(points)


# ------------------------------------------------------------------ #
#  χ² computation (JAX-traceable w.r.t. model_flux)                   #
# ------------------------------------------------------------------ #

def chi_squared(model_flux, data: SEDData):
    """
    Compute χ² between model νFν and data.

    Upper limits contribute only when the model exceeds them.
    This function is JAX-traceable w.r.t. model_flux (for autodiff).

    Parameters
    ----------
    model_flux : jax array, shape (N,)
        Model νFν at the data frequencies.
    data : SEDData
        Observed data.

    Returns
    -------
    chi2 : scalar
    """
    obs = data.fluxes
    err = data.errors
    is_ul = data.upper_limit_mask

    residual = (model_flux - obs) / err

    # Detections: always contribute
    det_chi2 = jnp.where(~is_ul, residual**2, 0.0)

    # Upper limits: contribute only if model > data
    ul_chi2 = jnp.where(is_ul & (model_flux > obs), residual**2, 0.0)

    return jnp.sum(det_chi2 + ul_chi2)


def chi_squared_reduced(model_flux, data: SEDData, n_free: int):
    """χ²/dof where dof = N_detections - N_free_params."""
    n_det = int(jnp.sum(~data.upper_limit_mask))
    dof = max(n_det - n_free, 1)
    return chi_squared(model_flux, data) / dof
