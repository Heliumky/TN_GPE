import sys, copy, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../NumpyTensorTools')))
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle
from matplotlib import cm
import plot_utility as pltut
import qtt_tools as qtt

# Step intervals for MPS data
step_intervals = np.arange(100, 1100, 100)  # 500, 1000, ..., 15000

# Initialize figure and axis
fig, ax = plt.subplots(figsize=(12, 9))
ax.set_xlabel(r'$x$', fontsize=25)
ax.set_ylabel(r'$y$', fontsize=25)

contour = None
cbar_contour = None

# Function to load MPS data and generate contour plot
def load_mps_data(step):
    filename = f"GD2_mps_step_{step}.pkl"
    try:
        with open(filename, 'rb') as file:
            data = pickle.load(file)
    except FileNotFoundError:
        print(f"MPS file {filename} not found, skipping...")
        return None, None, None
    
    mps = data  # Assuming the data is already normalized MPS
    x1, x2 = -8, 8
    N = len(mps) // 2
    Ndx = 2**N
    rescale = (x2 - x1) / Ndx
    shift = x1
    mps = qtt.normalize_MPS_by_integral (mps, x1, x2, Dim=2)
    # Generate binary index lists
    bxs = list(pltut.BinaryNumbers(N))
    bys = list(pltut.BinaryNumbers(N))
    
    # Convert binary indices to decimal coordinates
    xs = pltut.bin_to_dec_list(bxs, rescale, shift)
    ys = pltut.bin_to_dec_list(bys, rescale, shift)
    X, Y = np.meshgrid(xs, ys)
    
    # Compute MPS probability density
    Z = pltut.get_2D_mesh_eles_mps(mps, bxs, bys)
    Z = np.abs(Z)**2
    return X, Y, Z

# Function to update each frame
def update(frame):
    global contour, cbar_contour
    
    # Load MPS data based on step index
    step = step_intervals[min(frame, len(step_intervals) - 1)]
    X, Y, Z = load_mps_data(step)
    if X is not None:
        ax.clear()
        contour = ax.contourf(X, Y, Z, cmap=cm.coolwarm, antialiased=False)
        if not cbar_contour:
            cbar_contour = plt.colorbar(contour, ax=ax)
            cbar_contour.ax.tick_params(labelsize=25)
    
    ax.set_xlabel(r'$x$', fontsize=25)
    ax.set_ylabel(r'$y$', fontsize=25)
    ax.set_title(f'Step = {step}', fontsize=25)
    ax.tick_params(axis='both', which='major', labelsize=25)

    if frame == len(step_intervals) - 1:
        ani.event_source.stop()

# Create animation
ani = animation.FuncAnimation(
    fig,
    update,
    frames=len(step_intervals),
    interval=1000,
    repeat=False
)

plt.show()
