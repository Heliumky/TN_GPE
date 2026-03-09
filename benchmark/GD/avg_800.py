import numpy as np
import scipy.io as sio
import re
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter, AutoMinorLocator, MultipleLocator
import matplotlib.ticker as ticker
from matplotlib import cm
import plotsetting as ps  # Custom plotting settings module

matplotlib.rcParams['text.usetex'] = True 

ite = 20

# ============================================
# Step 1: Load GPELab results (BESP)
# ============================================
mat_data = sio.loadmat('GPELab_BESP_results.mat', struct_as_record=False, squeeze_me=True)
Outputs = mat_data['Outputs']

mu_mat = np.array(Outputs.Chemical_potential).flatten()
E_mat = np.array(Outputs.Energy).flatten()
CPU_time_mat = np.array(Outputs.CPU_time).flatten()

# ============================================
# Step 2: Load GPELab results (BEFD finer)
# ============================================
mat_FD_data = sio.loadmat('GPELab_BEFD_results.mat', struct_as_record=False, squeeze_me=True)
Outputs_FD = mat_FD_data['Outputs']

mu_mat_FD = np.array(Outputs_FD.Chemical_potential).flatten()
E_mat_FD = np.array(Outputs_FD.Energy).flatten()
CPU_time_mat_FD = np.array(Outputs_FD.CPU_time).flatten()

# ============================================
# Step 3: Load multi‑step data from text files (GD results)
# ============================================
steps = 10000
interval = 100

fd_gd_data = {}
gd_dim_data = {}
gd_sclope_data = {}

for step in range(interval, steps + 1, interval):
    filename = f"GD2_CPUTIME_step_{step}.txt"
    filename1 = f"NewGD2_psi2_dim_steps{step}.txt"
    filename2 = f"NewGD_energy_slope_step{step}.txt"

    try:
        fd_gd_data[step] = np.loadtxt(filename, dtype=np.complex128)
        gd_dim_data[step] = np.loadtxt(filename1, dtype=np.int32)
        gd_sclope_data[step] = np.loadtxt(filename2, dtype=np.float64)
    except OSError:
        print(f" Cannot read {filename} or its companion files. Skipping step {step}.")
        continue

# ============================================
# Step 4: Parse run.log (our own iteration data)
# ============================================
def parse_run_log(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        content = file.read()

    mu_pattern = r'mu(\d+)\s*=\s*([\d.]+)'
    etot_pattern = r'Etot(\d+)\s*=\s*([\d.]+)'
    cput_pattern = r'cput(\d+)\s*=\s*([\d.]+)'

    mu_matches = re.findall(mu_pattern, content)
    etot_matches = re.findall(etot_pattern, content)
    cput_matches = re.findall(cput_pattern, content)

    mu_dict = {int(step): float(value) for step, value in mu_matches}
    etot_dict = {int(step): float(value) for step, value in etot_matches}
    cput_dict = {int(step): float(value) for step, value in cput_matches}

    steps_sorted = sorted(mu_dict.keys())
    mu_array = np.array([mu_dict[s] for s in steps_sorted])
    etot_array = np.array([etot_dict[s] for s in steps_sorted])
    cput_array = np.array([cput_dict[s] for s in steps_sorted])

    return steps_sorted, mu_array, etot_array, cput_array

steps_log, mu_log, Etot_log, cput_log = parse_run_log('run.log')

# ============================================
# Step 5: Consolidate the loaded GD data
# ============================================
fd_gd_x = {}
fd_gd_y = {}
gd_dim_x = {}
gd_sclope_x = {}
gd_sclope_y = {}

for step in fd_gd_data:
    fd_gd_x[step] = fd_gd_data[step][:, 0].real
    fd_gd_y[step] = fd_gd_data[step][:, 1].real
    gd_dim_x[step] = gd_dim_data[step]
    gd_sclope_x[step] = gd_sclope_data[step][:, 0]
    gd_sclope_y[step] = gd_sclope_data[step][:, 1]

# Build cumulative CPU time over all steps
sorted_steps = sorted(fd_gd_x.keys())
sweep_cput = np.array([fd_gd_x[step] for step in sorted_steps])
sweep_cput = np.cumsum(sweep_cput, axis=1)
for i in range(1, sweep_cput.shape[0]):
    sweep_cput[i, :] += sweep_cput[i - 1, -1]
sweep_cput = sweep_cput.flatten()

# Flatten corresponding μ and energy data
sweep_mu = np.array([fd_gd_y[step] for step in sorted_steps]).flatten()
sweep_E  = np.array([gd_sclope_x[step] for step in sorted_steps]).flatten()

# ============================================
# Step 6: Save consolidated data
# ============================================
np.savetxt("compare_all_mu_E_vs_CPU.txt",
           np.column_stack((sweep_cput, sweep_mu, sweep_E)),
           header="CPU_time_cum  mu  E")
print(" Output written to compare_all_mu_E_vs_CPU.txt")

# ============================================
# Step 7: Compute average CPU times (total time / number of steps)
# ============================================
def avg_cpu_time(cpu_array, tot_step):
    """cpu_array is assumed to be cumulative CPU time per step, last element is total."""
    if len(cpu_array) == 0:
        return 0.0
    total = cpu_array[int(tot_step-1)]
    steps = len(cpu_array[:int(tot_step-1)])
    return total / steps

avgT_gd = avg_cpu_time(sweep_cput,799)
avgT_besp = avg_cpu_time(CPU_time_mat,799)
avgT_befd = avg_cpu_time(CPU_time_mat_FD,799)
avgT_ite = avg_cpu_time(cput_log,799)

# ============================================
# Step 8: Load wavefunctions for density plots
# ============================================
def extract_wavefunction(phi_raw, name="Phi"):
    """Extract complex wavefunction from loaded mat array."""
    if phi_raw.dtype == np.object_:
        phi_struct = phi_raw[0, 0]
        if isinstance(phi_struct, np.void):
            if 'real' in phi_struct.dtype.names and 'imag' in phi_struct.dtype.names:
                phi = phi_struct['real'] + 1j * phi_struct['imag']
                print(f"Reconstructed complex {name} from real/imag.")
            else:
                for field_name in phi_struct.dtype.names:
                    field = phi_struct[field_name]
                    if isinstance(field, np.ndarray) and field.size > 1:
                        phi = field
                        print(f"Using field '{field_name}' as {name}.")
                        break
                else:
                    raise ValueError(f"No suitable array found in {name}.")
        else:
            phi = phi_struct
    else:
        phi = phi_raw
    return phi

def prepare_density(psi, name="", default_L=42.0):
    """Convert wavefunction to 2D density map, trying to infer grid."""
    if psi is None:
        return None, None, None
    psi = np.array(psi)
    # Ensure complex, compute density
    density = np.abs(psi) ** 2
    shape = density.shape
    ndim = density.ndim

    if ndim == 2:
        # Already 2D, assume shape is (Ny, Nx)
        Ny, Nx = shape
        # Generate physical coordinates (assume square region)
        L = default_L
        x = np.linspace(-L/2, L/2, Nx)
        y = np.linspace(-L/2, L/2, Ny)
        X, Y = np.meshgrid(x, y)
        return X, Y, density
    elif ndim == 1:
        N = len(density)
        # Try to reshape to square (if N is perfect square)
        side = int(np.sqrt(N))
        if side * side == N:
            density_2d = density.reshape((side, side))
            L = default_L
            x = np.linspace(-L/2, L/2, side)
            y = np.linspace(-L/2, L/2, side)
            X, Y = np.meshgrid(x, y)
            print(f"{name}: reshaped 1D array of length {N} to {side}x{side} grid.")
            return X, Y, density_2d
        else:
            print(f"{name}: 1D array of length {N} cannot be reshaped to square. Using 1D plot fallback.")
            # Fallback: return 1D x coordinates
            L = default_L
            x = np.linspace(-L/2, L/2, N)
            return x, None, density
    else:
        print(f"{name}: unsupported dimension {ndim}. Cannot plot density.")
        return None, None, None

# ---- 8.1 QTT GD from psi2D_complex.txt ----
density_gd = None
try:
    psi_gd_txt = np.loadtxt('psi2D_complex.txt', dtype=np.float64)
    if psi_gd_txt.ndim == 2 and psi_gd_txt.shape[1] == 2:
        psi_gd = psi_gd_txt[:, 0] + 1j * psi_gd_txt[:, 1]
        X_gd, Y_gd, Z_gd = prepare_density(psi_gd, name="QTT GD")
        if X_gd is not None and Y_gd is not None:
            density_gd = (X_gd, Y_gd, Z_gd)  # 2D case
        else:
            density_gd = (X_gd, Z_gd)        # 1D case
    else:
        print("psi2D_complex.txt does not have 2 columns.")
except OSError:
    print("psi2D_complex.txt not found.")

# ---- 8.2 BESP from GPELab_BESP_results.mat ----
density_besp = None
try:
    phi_raw_besp = mat_data['Phi']
    phi_besp = extract_wavefunction(phi_raw_besp, "BESP")
    X_besp, Y_besp, Z_besp = prepare_density(phi_besp, name="BESP")
    if X_besp is not None and Y_besp is not None:
        density_besp = (X_besp, Y_besp, Z_besp)
    else:
        density_besp = (X_besp, Z_besp)
except (KeyError, ValueError) as e:
    print(f"Could not extract BESP wavefunction: {e}")

# ---- 8.3 BEFD from GPELab_BEFD_results.mat ----
density_befd = None
try:
    phi_raw_befd = mat_FD_data['Phi']
    phi_befd = extract_wavefunction(phi_raw_befd, "BEFD")
    X_befd, Y_befd, Z_befd = prepare_density(phi_befd, name="BEFD")
    if X_befd is not None and Y_befd is not None:
        density_befd = (X_befd, Y_befd, Z_befd)
    else:
        density_befd = (X_befd, Z_befd)
except (KeyError, ValueError) as e:
    print(f"Could not extract BEFD wavefunction: {e}")

# ---- 8.4 Benchmark from gpe_benchmark-Lx=42-Nx=2048-Om=0.946.mat ----
benchmark_file = 'gpe_benchmark-Lx=42-Nx=2048-Om=0.946.mat'
density_bench = None
try:
    bench_data = sio.loadmat(benchmark_file, struct_as_record=False, squeeze_me=True)
    phi_raw_bench = bench_data['u']
    phi_bench = extract_wavefunction(phi_raw_bench, "Benchmark")
    # Extract Lx from filename (assume square)
    match = re.search(r'Lx=([\d.]+)', benchmark_file)
    Lx_bench = float(match.group(1)) if match else 42.0
    X_bench, Y_bench, Z_bench = prepare_density(phi_bench, name="Benchmark", default_L=Lx_bench)
    if X_bench is not None and Y_bench is not None:
        density_bench = (X_bench, Y_bench, Z_bench)
    else:
        density_bench = (X_bench, Z_bench)
except Exception as e:
    print(f"Could not load benchmark file: {e}")

# ============================================
# Step 9: Plot comparison and densities
# ============================================
fig, axs = plt.subplots(2, 4, figsize=(24, 10))


# ---- First row: error plots (with avg time in labels) ----
# (a) μ error vs CPU time
axs[0,0].scatter(sweep_cput[::ite], np.abs(sweep_mu)[::ite], c='r', s=4,
                 label=f'QTT GD (avgT={avgT_gd:.2f} s)')
axs[0,0].scatter(sweep_cput[799], np.abs(sweep_mu)[799], c='r', s=48)

axs[0,0].scatter(CPU_time_mat[::ite], np.abs(mu_mat)[::ite], c='b', s=4,
                 label=f'BESP (avgT={avgT_besp:.2f} s)')
axs[0,0].scatter(CPU_time_mat[799], np.abs(mu_mat)[799], c='b', s=48)

axs[0,0].scatter(CPU_time_mat_FD[::ite], np.abs(mu_mat_FD)[::ite], c='g', s=4,
                 label=f'BEFD (avgT={avgT_befd:.2f} s)')
axs[0,0].scatter(CPU_time_mat_FD[799], np.abs(mu_mat_FD)[799], c='g', s=48)

axs[0,0].scatter(cput_log[::ite], np.abs(mu_log)[::ite], c='k', s=4,
                 label=f'ITE (avgT={avgT_ite:.2f} s)')
axs[0,0].scatter(cput_log[799], np.abs(mu_log)[799], c='k', s=48)


                 
axs[0,0].set_xlabel(r'CPU time (s)')
axs[0,0].set_xlim(0, 1e4)
axs[0,0].set_ylim(2.91, 3.80)
axs[0,0].set_ylabel(r'$\mathrm{\mu}$')
axs[0,0].xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
axs[0,0].ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
axs[0,0].text(0.05, 0.9, "(a)", fontsize=22, transform=axs[0,0].transAxes)
axs[0,0].legend(loc='upper right', fontsize=12, markerscale=6)

# (b) E error vs CPU time
axs[0,1].scatter(sweep_cput[::ite], np.abs(sweep_E)[::ite], c='r', s=4,
                 label=f'QTT GD')
axs[0,1].scatter(sweep_cput[799], np.abs(sweep_E)[799], c='r', s=48)    
             
axs[0,1].scatter(CPU_time_mat[::ite], np.abs(E_mat)[::ite], c='b', s=4,
                 label=f'BESP')
axs[0,1].scatter(CPU_time_mat[799], np.abs(E_mat)[799], c='b', s=48)

axs[0,1].scatter(CPU_time_mat_FD[::ite], np.abs(E_mat_FD)[::ite], c='g', s=4,
                 label=f'BEFD')
axs[0,1].scatter(CPU_time_mat_FD[799], np.abs(E_mat_FD)[799], c='g', s=48)

axs[0,1].scatter(cput_log[::ite], np.abs(Etot_log)[::ite], c='k', s=4,
                 label=f'IMT')
axs[0,1].scatter(cput_log[799], np.abs(Etot_log)[799], c='k', s=48)                 
                 
axs[0,1].set_xlabel(r'CPU time (s)')
axs[0,1].set_xlim(0, 1e4)
axs[0,1].set_ylim(2.25, 2.6)
axs[0,1].set_ylabel(r'$\mathrm{E}$')
axs[0,1].xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
axs[0,1].ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
axs[0,1].text(0.05, 0.9, "(b)", fontsize=22, transform=axs[0,1].transAxes)


# (c) μ error vs STEP
axs[0,2].scatter(range(len(sweep_cput))[::ite], np.abs(sweep_mu)[::ite], c='r', s=4,
                 label=f'QTT GD (avgT={avgT_gd:.2f} s)')
axs[0,2].scatter(range(len(CPU_time_mat))[::ite], np.abs(mu_mat)[::ite], c='b', s=4,
                 label=f'BESP (avgT={avgT_besp:.2f} s)')
axs[0,2].scatter(range(len(CPU_time_mat_FD))[::ite], np.abs(mu_mat_FD)[::ite], c='g', s=4,
                 label=f'BEFD (avgT={avgT_befd:.2f} s)')
axs[0,2].scatter(range(len(cput_log))[::ite], np.abs(mu_log)[::ite], c='k', s=4,
                 label=f'ITE (avgT={avgT_ite:.2f} s)')
axs[0,2].set_xlabel(r'Step')

axs[0,2].set_xlim(0, 8*1e2)
axs[0,2].set_ylim(2.91, 3.80)
axs[0,2].set_ylabel(r'$\mathrm{\mu}$')
axs[0,2].xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
axs[0,2].ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
axs[0,2].text(0.05, 0.9, "(c)", fontsize=22, transform=axs[0,2].transAxes)


# (d) E error vs STEP
axs[0,3].scatter(range(len(sweep_cput))[::ite], np.abs(sweep_E)[::ite], c='r', s=4,
                 label=f'QTT GD (avgT={avgT_gd:.2f} s)')
axs[0,3].scatter(range(len(CPU_time_mat))[::ite], np.abs(E_mat)[::ite], c='b', s=4,
                 label=f'BESP (avgT={avgT_besp:.2f} s)')
axs[0,3].scatter(range(len(CPU_time_mat_FD))[::ite], np.abs(E_mat_FD)[::ite], c='g', s=4,
                 label=f'BEFD (avgT={avgT_befd:.2f} s)')
axs[0,3].scatter(range(len(cput_log))[::ite], np.abs(Etot_log)[::ite], c='k', s=4,
                 label=f'IMT (avgT={avgT_ite:.2f} s)')
axs[0,3].set_xlabel(r'Step')
axs[0,3].set_xlim(0, 8*1e2)
axs[0,3].set_ylim(2.25, 2.6)
axs[0,3].set_ylabel(r'$\mathrm{E}$')
axs[0,3].xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
axs[0,3].ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
axs[0,3].text(0.05, 0.9, "(d)", fontsize=22, transform=axs[0,3].transAxes)


# ---- Second row: density plots ----
def plot_density(ax, density_data, title, pos_label):
    """Helper to plot density: if 2D do contourf, else 1D plot."""
    if density_data is None:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return
    if len(density_data) == 3:  # 2D case: X, Y, Z
        ax.relim()
        ax.autoscale_view()
        X, Y, Z = density_data
        surf = ax.contourf(X, Y, Z, cmap=cm.coolwarm, antialiased=False)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel(r'$x$', fontsize=20)
        ax.set_ylabel(r'$y$', fontsize=20)
        # Set nice tick intervals (using custom module if available)
        try:
            ps.set_tick_inteval(ax.yaxis, major_itv=5, minor_itv=1)
            ps.set_tick_inteval(ax.xaxis, major_itv=5, minor_itv=1)
        except AttributeError:
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.tick_params(axis='both', which='major', labelsize=20)
        # Add colorbar
        cbar = plt.colorbar(surf, ax=ax)
        cbar.ax.tick_params(labelsize=20)
        # Add label (e.g., (e))
        ax.text(0.05, 0.9, pos_label, fontsize=20, transform=ax.transAxes)
    elif len(density_data) == 2:  # 1D case: x, y
        x, y = density_data
        ax.plot(x, y, 'b-')
        ax.set_xlabel(r'$x$', fontsize=20)
        ax.set_ylabel(r'$|\psi|^2$', fontsize=20)
        ax.text(0.05, 0.9, pos_label, fontsize=20, transform=ax.transAxes)
    else:
        ax.text(0.5, 0.5, 'Invalid data', ha='center', va='center', transform=ax.transAxes)
    ax.set_title(title)

plot_density(axs[1,0], density_gd, 'QTT GD Step = 800', '(e)')
plot_density(axs[1,1], density_besp, 'BESP Step = 800', '(f)')
plot_density(axs[1,2], density_befd, 'BEFD Step = 800', '(g)')
plot_density(axs[1,3], density_bench, 'ITE Step = 800', '(h)')

plt.tight_layout()
plt.savefig("compare_plots_app.pdf", bbox_inches='tight')
plt.show()
