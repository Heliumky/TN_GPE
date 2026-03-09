import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib
from matplotlib.patches import Patch
import numpy as np
import scipy.io

for i in range(11, 12):
    site_num = 2**i
    try:
        mat_file = scipy.io.loadmat(f'gpe_benchmark-Lx=42-Nx=2048-Om=0.946-step=99900.mat')
    except FileNotFoundError:
        print(f"File not found for i={i}")
        continue  # Skip to the next iteration

    data = mat_file
    u = data['u']
    x = data['x'][0]
    y = data['y'][0]
    mu_t = np.real(data['mu_t'][0])
    cput = data['cput'][0]
    
    #X1, Y1 = np.meshgrid(x, y)
    #xflat = X1.flatten()
    #yflat = Y1.flatten()
    #uflat = u.flatten()
    #u2flat = (np.abs(u)**2).flatten()
    
    #print(u2flat.shape)
    
    #np.savetxt(f'YK_MU_site{i}.txt', np.real(mu_t), fmt='%4.12e') 
    #u_data = np.column_stack((xflat, yflat, np.real(uflat)))
    #u2_data = np.column_stack((xflat, yflat, np.real(u2flat)))
    #mu_cput = np.column_stack((cput, mu_t))
    
    #print(u_data)
    #np.savetxt(f'u_data_site{i}.txt', u_data, delimiter=' ')
    #np.savetxt(f'u2data_site{i}.txt', u2_data, delimiter=' ')
    np.savetxt(f'mu_cputime{i}.txt', mu_cput, delimiter=' ')
