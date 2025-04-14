import numpy as np
import matplotlib.pyplot as plt
import plotsetting as ps
from matplotlib.ticker import MaxNLocator, MultipleLocator

steps = 20
step_iter = 10
#num_cpu = 28

fd_gd_data = {}
gd_dim_data = {}
gd_sclope_data = {}

for step in range(step_iter, int(steps+1), step_iter):
    filename = f"TDVP_CPUTIME_step_{step}.txt"
    filename1 = f"psi2_dim_{step}.txt"
    #filename2 = f"NewGD_energy_sclope_step{step}.txt"
    fd_gd_data[step] = np.loadtxt(filename, dtype=np.complex128)  
    gd_dim_data [step] = np.loadtxt(filename1, dtype=np.int32)  
    #gd_sclope_data[step] = np.loadtxt(filename2, dtype=np.float64)  

fd_gd_x = {}
fd_gd_y = {}
gd_dim_x = {}
#gd_sclope_x = {}
#gd_sclope_y = {}

for step in range(step_iter, int(steps+1), step_iter):
    fd_gd_x[step] = fd_gd_data[step][:, 0].real 
    fd_gd_y[step] = fd_gd_data[step][:, 1].real  
    gd_dim_x[step] = gd_dim_data [step]
    #gd_sclope_x[step] = gd_sclope_data[step][:, 0]
    #gd_sclope_y[step] = gd_sclope_data[step][:, 1]


sweep_cput = np.array([fd_gd_x[step] for step in range(step_iter, int(steps+1), step_iter)])

sweep_cput = np.cumsum(sweep_cput, axis=1)
for i in range(1, sweep_cput.shape[0]):
    sweep_cput[i, :] += sweep_cput[i - 1, -1]  

sweep_cput = sweep_cput.flatten()
sweep_mu = np.array([fd_gd_y[step] for step in range(step_iter, int(steps+1), step_iter)]).flatten()
sweep_dim = np.array([gd_dim_x[step] for step in range(step_iter, int(steps+1), step_iter)]).flatten()
#sweep_E = np.array([gd_sclope_x[step] for step in range(step_iter, int(steps+1), step_iter)]).flatten()
#sweep_sclope = np.array([gd_sclope_y[step] for step in range(step_iter, int(steps+1), step_iter)]).flatten()



# Save data to a file with five columns
np.savetxt(f'TDVP_data_{steps}.txt', np.column_stack((sweep_cput, sweep_mu, sweep_dim)))

print(np.array(sweep_cput).shape)

print(np.array(sweep_cput).shape)

fig2, ax2 = plt.subplots()
fig1, ax1 = plt.subplots()
#fig3, ax3 = plt.subplots()
#fig4, ax4 = plt.subplots()
ax2.relim()
ax2.autoscale_view()

ax1.relim()
ax1.autoscale_view()

ax1.plot(range(len(sweep_dim)), sweep_dim, label=r"TDVP", color='red')
ax2.plot(sweep_cput, sweep_mu, label="TDVP ", color='red')


ax1.set_ylabel(r"$\chi$", loc="center")
ax1.set_xlabel(r"$\mathrm{Steps}$", loc="center")


ax2.set_xlabel(r"$\mathrm{CPU\\ time (s)}$", loc="center")
ax2.set_ylabel(r"$\mathrm{\mu(t)}$", loc="center")
ax2.legend(fontsize=12)
ax1.legend(fontsize=12)
#ax3.legend(fontsize=12)
#ax4.legend(fontsize=12)
ps.set(ax2, tick_length=2)
ps.set(ax1, tick_length=2)
ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.show()

