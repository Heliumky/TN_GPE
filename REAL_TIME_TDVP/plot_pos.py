import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.ticker as ticker
import plotsetting as ps

# Load data (example data)

f10 = np.loadtxt("position_val.txt", dtype=complex)
#f30 = np.loadtxt("position_val_D20_n12.txt", dtype=complex)
# Create figure and axis with adjusted size
figax, ax = plt.subplots()

ax.relim()
ax.autoscale_view()
ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
#ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
# set title
#ax.set_title('Pure GP CPU TIME')
#print(gd30[:,0])
# Scatter plot for Intel Fortran and one-site TDVP
ax.plot(f10[:,0], f10[:,1], color='red',ls = "dashdot", label=r"TDVP, $\epsilon_{\psi^2} = 10^{-4}$",alpha= 0.8)
#ax.plot(f30[:,0], f30[:,1], color='red', label=r"TDVP, $\epsilon_{\psi^2}$ = 12",alpha= 0.6)
# Set labels and title
ax.set_xlim(0,10)
ax.set_ylim(1.00,1.5)
ax.set_xlabel(r'Time')
ax.set_ylabel(r'$\sqrt{\langle x^2 + y^2 \rangle}$',fontsize= 16)
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
#ax.set_ylim([1e-5, 3*1e-1])  # Limit y-axis from 0.1 to 1
ax.legend(loc = 'upper right', fontsize = 20)
# Set integer tick marks on x-axis
ps.text(ax, x=0.1, y=0.9, t="(a)",fontsize = 22)
ps.set(ax)
figax.savefig("pos_time.pdf", transparent=False)
plt.show()

