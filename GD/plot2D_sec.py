import sys, copy, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../NumpyTensorTools')))
import pickle
import qtt_tools as qtt
import numpy as np
import npmps
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib
from matplotlib.patches import Patch
import plot_utility_jax as pltut
import plotsetting as ps
from matplotlib.ticker import MultipleLocator, MaxNLocator
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Add this before any plotting commands
# plt.rcParams['text.usetex'] = False


steps = 1400 
#input_mps_path = f"TDVP_step_{steps}.pkl"
#input_mps_path = f"tci_initial.pkl"
input_mps_path = f"GD2_mps_step_{steps}.pkl"
with open(input_mps_path, 'rb') as file:
    data = pickle.load(file)
mps = data

x1 = -21
x2 = 21
#mps = qtt.normalize_MPS_by_integral (mps, x1, x2, Dim=2)
for i in range(4):
    mps = qtt.kill_site_2D(mps, 80,dtype = np.complex128)

mps = qtt.normalize_MPS_by_integral (mps, x1, x2, Dim=2)
N = len(mps)//2
Ndx = 2**N
rescale = (x2-x1)/Ndx
shift = x1
print(npmps.MPS_dims(mps))
bxs = list(pltut.BinaryNumbers(N))
bys = list(pltut.BinaryNumbers(N))

xs = pltut.bin_to_dec_list (bxs, rescale, shift)
ys = pltut.bin_to_dec_list (bys, rescale, shift)
X, Y = np.meshgrid (xs, ys)

Z = pltut.get_2D_mesh_eles_mps(mps, bxs, bys, batch_size=2**15)
# Flatten arrays into columns
X_flat = X.flatten()
Y_flat = Y.flatten()
Z_flat = Z.flatten()

# Split into real and imaginary parts
Z_real = np.real(Z_flat)
Z_imag = np.imag(Z_flat)

# Stack into 4 columns: x, y, Re(psi), Im(psi)
#data_to_save = np.column_stack((X_flat, Y_flat, Z_real, Z_imag))
data_to_save = np.column_stack((Z_real, Z_imag))
# Save to .dat (tab-separated, MATLAB-friendly)
np.savetxt("psi2D_complex.txt", data_to_save, fmt="%.10e", delimiter="\t")
Z = np.abs(Z)**2
fig, ax = plt.subplots()
ax.relim()
ax.autoscale_view()
ax.tick_params(axis='both', which='major', labelsize=20)
ax.plot(X[Ndx//2,:],Z[Ndx//2,:], label=r'$| \psi |^{2}$', color='black')
ax.legend(loc='upper right', fontsize=14)
#ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.yaxis.set_major_locator(MultipleLocator(0.01))
ax.set_xlabel(r'$x$', fontsize=20)
ax.set_ylabel(r'$| \mathrm{\psi} |^{2}$', fontsize=20)
#ax.set_aspect('equal', adjustable='box')
#ps.set_tick_inteval(ax.xaxis, major_itv=3, minor_itv=1)
ps.text(ax, x=0.1, y=0.9, t="(a)", fontsize=20)
ax.set_xlim(x1, x2)
#ps.set(ax)
plt.savefig("2D_psi2_sec.pdf", transparent=False)



fig2,ax2 = plt.subplots()
ax2.relim()
ax2.autoscale_view()
ax2.tick_params(axis='both', which='major', labelsize=10)
surfxy = ax2.contourf(X, Y, Z, cmap=cm.coolwarm, antialiased=False)
ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
ax2.set_xlabel(r'$x$', rotation=0, fontsize=20)
ax2.set_ylabel(r'$y$', rotation=0, fontsize=20)
ax2.set_aspect('equal', adjustable='box')
#fake2Dline = matplotlib.lines.Line2D([0], [0], linestyle="none", c='y', marker='o')
#ax.legend([fake2Dline], [r'$|\psi|^2_{TDVP}$'], numpoints=1)
cbar = fig2.colorbar(surfxy)
cbar.ax.tick_params(labelsize=20)
ps.set_tick_inteval(ax2.yaxis, major_itv=5, minor_itv=1)
ps.set_tick_inteval(ax2.xaxis, major_itv=5, minor_itv=1)
ax2.set_ylim(x1, x2)
ax2.set_xlim(x1, x2)
ps.text(ax2, x=0.1, y=0.9, t="(b)", fontsize=20)
#ps.set(ax2)
plt.savefig(f"{input_mps_path}.pdf", bbox_inches='tight')
