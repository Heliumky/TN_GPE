import numpy as np
import matplotlib.pyplot as plt
import plotsetting as ps

steps= 5000

data = np.loadtxt(f'GD2_CPUTIME_step_{steps}.txt', dtype=np.complex128)
ens_GD2 = data[:,1].real
ts2 = data[:,0].real
Exact_E = 0
# Plot energy

fig2, ax2 = plt.subplots()
ax2.relim()
ax2.autoscale_view()
#ax2.plot(range(len(ens_TDVP)), np.abs(np.array(ens_TDVP)-Exact_E), label=f'TDVP, $D_{{\psi^{{2}}}}={maxdim_psi2}$')
ax2.plot(range(len(ens_GD2)), np.abs(np.array(ens_GD2) - Exact_E), label=f'GD2, $D_{{\psi^{{2}}}}=40$, $\\mathrm{{Steps}}={steps}$')
#ax2.plot(range(len(ens_DMRG)), np.abs(np.array(ens_DMRG)-Exact_E), label='DMRG step')
ax2.set_xlabel(r"Steps", loc="center")
ax2.set_ylabel(r"$<\mu(step)>$", loc="center")
ax2.set_yscale('log')
ax2.legend()
ps.set(ax2,tick_length=2)
plt.savefig(f"E_step_{steps}.pdf", format='pdf')

#en = ens_GD2[-1]        # replace by the exact energy
fig2, ax2 = plt.subplots()
ax2.relim()
ax2.autoscale_view()
#ax2.plot(ts0, ens_TDVP, marker='.', label=f'TDVP Wall time,$D_{{\psi^{{2}}}}={maxdim_psi2}$')
ax2.plot(ts2, ens_GD2, marker='+', label=f'GD2 Wall time, $D_{{\psi^{{2}}}}={40}$, $\\mathrm{{Steps}}={steps}$')
#ax2.plot(ts_dmrg, ens_DMRG, marker='x', label=f'DMRG Wall time,$D_{{\psi^{{2}}}}={maxdim_psi2}$')
#ax2.plot(ts0, abs(ens_TDVP-en), marker='.', label='TDVP')
#ax2.plot(ts2, abs(ens_GD2-en), marker='+', label='GD2')
ax2.set_xlabel(r"Wall time(s)", loc="center")
ax2.set_ylabel(r"$<\mu(t)>$", loc="center")
ax2.set_yscale('log')
ax2.legend()
ps.set(ax2,tick_length=2)
plt.savefig(f"E_Walltime_{steps}.pdf", format='pdf')
#plt.show()
