"""
Physical constants in CGS (Gaussian) units.

Mirrors constants.jl from the Julia implementation.
"""

# Fundamental constants (CGS-Gaussian)
c = 2.99792458e10           # speed of light [cm/s]
q = 4.80320425e-10          # elementary charge [statC]
h = 6.62607015e-27          # Planck constant [erg·s]
hbar = 1.054571817e-27      # reduced Planck constant [erg·s]
me = 9.1093837015e-28       # electron mass [g]
r0 = 2.8179403262e-13       # classical electron radius [cm]
kB = 1.380649e-16           # Boltzmann constant [erg/K]
sigma_T = 6.6524587e-25     # Thomson cross-section [cm²]

# Astrophysical constants
T_CMB = 2.725               # CMB temperature [K]
U_CMB = 4.2e-13             # CMB energy density [erg/cm³]
eps_CMB = 2e-9              # CMB characteristic photon energy [dimensionless, in me c² units]
pc = 3.08567758128e18       # parsec [cm]
kpc = 3.08567758128e21      # kiloparsec [cm]

# Convenience
LCONST = 1e46               # luminosity normalization [erg/s]
c_km_s = 299792.458         # speed of light [km/s]
Mpc_cm = 3.08567758e24      # cm per Mpc
