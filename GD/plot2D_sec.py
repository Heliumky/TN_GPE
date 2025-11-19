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


steps = 1000 
#input_mps_path = f"TDVP_step_{steps}.pkl"
#input_mps_path = f"tci_initial.pkl"
#input_mps_path = f"tci_initial_comp.pkl"
input_mps_path = f"GD2_mps_step_{steps}.pkl"
with open(input_mps_path, 'rb') as file:
    data = pickle.load(file)
mps = data

x1 = -21
x2 = 21
#mps = qtt.normalize_MPS_by_integral (mps, x1, x2, Dim=2)
for i in range(9):
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

#result = get_2D_mesh_eles_mps(mps, bxs, bys, mode='low_memory')

# 对于中等问题（10-100万点），使用平衡模式
#result = get_2D_mesh_eles_mps(mps, bxs, bys, mode='balanced', batch_size=128)

# 让小问题（<10万点），使用优化模式
#result = get_2D_mesh_eles_mps(mps, bxs, bys, mode='optimized', batch_size=256)

# 或者让系统自动选择
#result = get_2D_mesh_eles_mps(mps, bxs, bys, mode='auto')
#get_2D_mesh_eles_mps(mps, bxs, bys, mode='balanced', batch_size=128)
Z = pltut.get_2D_mesh_eles_mps(mps, bxs, bys, batch_size=8192)
# Flatten arrays into columns
X_flat = X.flatten()
Y_flat = Y.flatten()
Z_flat = Z.flatten()

# Stack them: each row = [x, y, z]
data_to_save = np.column_stack((X_flat, Y_flat, Z_flat))

# Save to .dat file (space-separated, MATLAB compatible)
np.savetxt("psi2_2D.dat", data_to_save, fmt="%.8e", delimiter="\t", header="x\ty\t|psi|^2")
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

