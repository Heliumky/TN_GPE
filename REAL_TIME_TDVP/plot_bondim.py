import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter
import plotsetting as ps

# Load data


tdvpds_n4_20 = np.loadtxt("psi2_dim.txt")
tdvpd_n4_20 = np.loadtxt("psi_dim.txt")




# Create figure
fig, ax = plt.subplots()

# Main plot settings
ax.relim()
ax.autoscale_view()
#ax.tick_params(axis='both', direction='in', length=6, width=0.7, grid_color='black', grid_alpha=0.5)
ax.set_ylim(5.8,10)
ax.set_xlim(0,8*1e3)
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))  
ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))      
ax.set_xlabel(r'$\mathrm{Step}$')
ax.set_ylabel(r'$D$')

# Main plot curves
ax.plot(range(len(tdvpd_n4_20)), tdvpd_n4_20, color='red',ls = "dashdot", label=r"TDVP, $\epsilon_{\psi} = 10^{-4}$",alpha= 0.8)


# Adjusted legend position (upper left to avoid overlapping)
#ax.legend(bbox_to_anchor=(0.6, 1), fontsize=14)

# Add text (b)
ps.text(ax, x=0.1, y=0.9, t="(b)", fontsize=22)

# Create inset plot
ax_inset = ps.new_panel(fig, left=0.4, bottom=0.38, width=0.52, height=0.52)
#ax_inset.tick_params(axis='both', direction='in', length=4, width=0.5)

# Ensure more x-ticks in the inset
ax_inset.yaxis.set_major_locator(MaxNLocator(integer=True))
ax_inset.xaxis.set_major_locator(MaxNLocator(nbins=5)) 
ax_inset.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax_inset.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
ax_inset.set_ylim(5,10)
ax_inset.set_xlim(0,8*1e3)
# Inset plot curves
ax_inset.plot(range(len(tdvpds_n4_20)), tdvpds_n4_20, color='red',ls = "dashdot", label=r"TDVP, $\epsilon_{\psi^2}$ = 4",alpha= 0.8)
#ax_inset.plot(range(len(tdvpds_n8_20)), tdvpds_n8_20, color='black',ls = "--", label=r"TDVP, $\epsilon_{\psi^2}$ = 8",alpha= 0.7)
#ax_inset.plot(range(len(tdvpds_n12_20)), tdvpds_n12_20, color='red', label=r"TDVP, $\epsilon_{\psi^2}$ = 12",alpha= 0.6)

# Inset labels
ax_inset.set_xlabel(r'$\mathrm{Step}$', fontsize=22)
ax_inset.set_ylabel(r'$\chi$', fontsize=22)

# Adjusted legend position (outside, top right)
#ax_inset.legend(loc="upper right", fontsize=8, frameon=False)

# Add text (c)
#ps.text(ax_inset, x=0.1, y=0.9, t="(c)", fontsize=22)

# Apply plot settings
ps.set(ax)
ps.set(ax_inset, fontsize=22)

# Save and show
plt.savefig("TDVP_dim_step.pdf", transparent=False)
plt.show()

