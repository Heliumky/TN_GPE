import numpy as np
import matplotlib.pyplot as plt
import plotsetting as ps

#steps = 3000
#num_cpu = 28

fd_gd_data = {}


for step in range(500, 3501, 500):
    filename = f"GD2_CPUTIME_step_{step}.txt"#
    fd_gd_data[step] = np.loadtxt(filename, dtype=np.complex128)  
#fort_filename = f"CPUTIME-MU.txt"
#fortran_mu_cput = np.loadtxt(fort_filename , dtype=np.float64) 

#fort_cput = np.array(fortran_mu_cput[:, 0])
#fort_mu = np.array(fortran_mu_cput[:,1])


fd_gd_x = {}
fd_gd_y = {}

for step in range(500, 3501, 500):
    fd_gd_x[step] = fd_gd_data[step][:, 0].real 
    fd_gd_y[step] = fd_gd_data[step][:, 1].real  

sweep_cput = np.array([fd_gd_x[step] for step in range(500, 3501, 500)])
sweep_mu = np.array([fd_gd_y[step] for step in range(500, 3501, 500)]).flatten()

for i in range(1, sweep_cput.shape[0]):
    sweep_cput[i, :] += sweep_cput[i - 1, -1]  

sweep_cput = sweep_cput.flatten()

print(np.array(sweep_cput).shape)

fig2, ax2 = plt.subplots()
ax2.relim()
ax2.autoscale_view()

##steps_range = list(range(500, 14501, 500))

ax2.plot(sweep_cput, sweep_mu, label="GD2 ", color='red',)
#ax2.plot(fort_cput, fort_mu, label="fortran", color='blue')

ax2.set_xlabel(r"$\mathrm{CPU\\ time (s)}$", loc="center")
ax2.set_ylabel(r"$\mathrm{\mu(t)}$", loc="center")
ax2.legend(fontsize=12)
ps.set(ax2, tick_length=2)
plt.show()
#plt.savefig(f"D_Walltime_{steps}combi.pdf", format='pdf', transparent=True)
#print(sweep_cput)

