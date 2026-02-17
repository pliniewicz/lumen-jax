import h5py
import numpy as np

with h5py.File("bulk_seds.h5", "r") as f:
    nu = f["nu"][:]                    # (200,)
    nuFnu = f["nuFnu"][:]              # (N_sims, 200)
    
    # Sampled parameters as a dict
    params = {k: f["params"][k][:] for k in f["params"]}
    
    # Fixed values
    # z = f["params"].attrs["z"]
    # model = f["params"].attrs["model"]
    # params["model"] = f["params"]["model"][:]  # array of ints: [1, 4, 0, 3, ...]

from lumen import (
        make_params, observed_sed,
        )

import sys
import matplotlib.pyplot as plt

num = int(sys.argv[1]) - 1 if len(sys.argv) > 1 else 0
num_total = len(nuFnu[:])

fig, ax = plt.subplots(figsize=(9, 5.5))
ax.loglog(nu, np.clip(nuFnu[num, :], 1e-30, None), 'k-', lw=2.2, label='Total SED')
ax.set_xlabel(r'$\nu$ [Hz]')
ax.set_ylabel(r'$\nu F_\nu$ [erg/s/cm$^2$]')
ax.set_title(f"Model {params['model'][num]}")
fig.suptitle(f'Lumen-JAX: synchrotron + IC/CMB simulation {num+1}/{num_total}')
ax.set_ylim(bottom=1e-30)
# ax.text(1e13, 1e-31, "Hi")
plt.figtext(0.10, 0.03, f"z={np.round(params['z'][num],2)}", ha="center", fontsize=11)
plt.figtext(0.20, 0.03, f"theta={np.round(params['theta'][num],3)}", ha="center", fontsize=11)
plt.figtext(0.90, 0.03, f"G0={np.round(params['G0'][num], 3)}", ha="center", fontsize=11)
plt.figtext(0.10, 0.90, f"gmin={np.round(params['gamma_min'][num],1)}", ha="center", fontsize=11)
plt.figtext(0.25, 0.90, f"gmax={np.round(params['gamma_max'][num],1)}", ha="center", fontsize=11)
plt.figtext(0.40, 0.90, f"p={np.round(params['p'][num],3)}", ha="center", fontsize=11)
plt.figtext(0.70, 0.90, f"Rj={np.round(params['Rj'][num]/3.0857e21, 3)}", ha="center", fontsize=11)
plt.figtext(0.80, 0.90, f"l={np.round(params['l'][num]/3.0857e21, 3)}", ha="center", fontsize=11)
plt.figtext(0.90, 0.90, f"Lj={np.round(params['Lj'][num]/1e45, 3)}", ha="center", fontsize=11)

ax.legend()
# fig.tight_layout()
fig.savefig('simulation_quick.pdf', dpi=150)
fig.savefig(f'simulation_{num+1}.pdf', dpi=150)
