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
import plot_utility as pltut
import plotsetting as ps
from matplotlib.ticker import MultipleLocator, MaxNLocator
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

steps= 2000
#input_mps_path = f"TDVP_step_{steps}.pkl"
#input_mps_path = f"tci_initial.pkl"
input_mps_path = f"GD2_mps_step_{steps}.pkl"
with open(input_mps_path, 'rb') as file:
    data = pickle.load(file)
mps = data

x1 = -6
x2 = 6
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

Z = pltut.get_2D_mesh_eles_mps (mps, bxs, bys)
Z = np.abs(Z)**2
fig, ax = plt.subplots()
ax.relim()
ax.autoscale_view()
ax.tick_params(axis='both', which='major', labelsize=25)
ax.plot(X[Ndx//2,:],Z[Ndx//2,:], label=r'$| \psi |^{2}$', color='black')
#ax.legend(loc='upper right', fontsize=14)
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.yaxis.set_major_locator(MultipleLocator(0.01))
ax.set_xlabel(r'$x$', fontsize=25)
ax.set_ylabel(r'$| \psi |^{2}$', fontsize=25)
ax.set_aspect('equal', adjustable='box')
ps.set(ax)
plt.savefig("2D_psi2_sec.pdf", transparent=False)
fig,ax = plt.subplots()
ax.relim()
ax.autoscale_view()
ax.tick_params(axis='both', which='major', labelsize=25)
surfxy = ax.contourf(X, Y, Z, cmap=cm.coolwarm, antialiased=False)
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.set_xlabel(r'$x$', rotation=0, fontsize=25)
ax.set_ylabel(r'$y$', rotation=0, fontsize=25)
ax.set_aspect('equal', adjustable='box')
#fake2Dline = matplotlib.lines.Line2D([0], [0], linestyle="none", c='y', marker='o')
#ax.legend([fake2Dline], [r'$|\psi|^2_{TDVP}$'], numpoints=1)
cbar = fig.colorbar(surfxy)
cbar.ax.tick_params(labelsize=25)
#ps.set(ax)
plt.savefig(f"{input_mps_path}.pdf", bbox_inches='tight')

