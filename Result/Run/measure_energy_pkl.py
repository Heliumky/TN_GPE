import sys, copy, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../NumpyTensorTools')))
import numpy_dmrg as dmrg
import numpy as np
import matplotlib.pyplot as plt
import npmps
import plot_utility as pltut
import hamilt.hamilt_sho as sho
import gradient_descent_old as gd
import qtt_tools as qtt
import hamilt.hamilt_angular_momentum as ang
import tci
import time
import pickle
import gradient_descent_GP as gdGP
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib

def psi_sqr (psi):
    psi_op = qtt.MPS_to_MPO (psi)
    res = npmps.exact_apply_MPO (psi_op, psi)
    #res = npmps.svd_compress_MPS (res, cutoff=1e-12)
    return res

# psi2 is the initial guess of the result
def fit_psi_sqr (psi, psi2, maxdim, cutoff=1e-16):
    psi_op = qtt.MPS_to_MPO (psi)
    psi_op = npmps.conj (psi_op)
    fit = dmrg.fit_apply_MPO (psi_op, psi, psi2, numCenter=1, nsweep=1, maxdim=maxdim, cutoff=cutoff)
    #psi2_exact = psi_sqr (psi)
    #print_overlap (psi2_exact, fit)
    return fit

# def make_H_GP (H0, psi, psi2, g, maxdim):
#     #t1 = time.time()
#     psi = copy.copy(psi)                                # time
#    # psi2 = psi_sqr (psi)
#     #psi2 = fit_psi_sqr(psi, psi2, maxdim = maxdim, cutoff = cutoff)
#    # for i in range(len(psi2)):
#    #     print(psi2[i].shape)
#    # t2 = time.time()                                # time
#    # print('psi2 time',(t2-t1))                      # time

#     H_psi = qtt.MPS_to_MPO (psi2)
#     H_psi[0] *= g
#     H = npmps.sum_2MPO (H0, H_psi)
#     #H = npmps.svd_compress_MPO (H, cutoff=1e-12)
#     return H

# def make_H_GP2 (H0, psi, psi2, g, maxdim):   
#     psi2 = copy.copy(psi2)                             # time
#     H_psi = qtt.MPS_to_MPO (psi2)
#     H_psi[0] *= g
#     H = npmps.sum_2MPO (H0, H_psi)
#     #H = npmps.svd_compress_MPO (H, cutoff=1e-12)
#     return H

def fitfun_xy (x, y):
    f = cut_wave[x][y]
    return f

def fitfun2_xy (x, y):
    f = cut_density[x][y]
    return f

def inds_to_num (inds, dx, shift):
    bstr = ''
    for i in inds:
        bstr += str(i)
    return pltut.bin_to_dec (bstr, dx, shift)

def fitfun (inds):
    N = len(inds)//2
    Ndx = 2**N
    xinds, yinds = inds[:N], inds[N:]
    x = inds_to_num (xinds, dx=1, shift = 0)
    y = inds_to_num (yinds, dx=1, shift = 0)
    #print(xinds,yinds,x,y)
    return fitfun_xy (x, y)

def fitfun2 (inds):
    N = len(inds)//2
    Ndx = 2**N
    xinds, yinds = inds[:N], inds[N:]
    x = inds_to_num (xinds, dx=1, shift = 0)
    y = inds_to_num (yinds, dx=1, shift = 0)
    #print(xinds,yinds,x,y)
    return fitfun2_xy (x, y)

def get_init_state (N, x1, x2, maxdim):
    seed = 13
    np.random.seed(seed)
    mps = tci.tci (fitfun, 2*N, 2, maxdim=maxdim)
    mps = npmps.normalize_MPS (mps)
    return mps    


def get_init_statesqr (N, x1, x2, maxdim):
    mps = tci.tci (fitfun2, 2*N, 2, maxdim=maxdim)
    mps = npmps.normalize_MPS (mps)
    return mps  

#wave_func = np.loadtxt("im2d-fin-wf_GP.txt")
wave_func = np.loadtxt("im2d-fin-wf.txt")
position = np.loadtxt("im2d-den-10.txt")

real_wave = wave_func[:,0]
imag_wave = wave_func[:,1]
x_value =  position[:,0].reshape(int(2**8 + 1),int(2**8 + 1)).T
y_value = position[:,1].reshape(int(2**8 + 1),int(2**8 + 1)).T

wave = (real_wave + 1j*imag_wave).reshape(int(2**8 + 1),int(2**8 + 1)).T

density = np.abs(real_wave**2 + imag_wave**2).reshape(int(2**8 + 1),int(2**8 + 1)).T


# matplotlib.rcParams.update({'font.size': 20})
# matplotlib.rcParams['font.weight'] = 'normal'
# fig = plt.figure(figsize=(10, 8), frameon=False)
# extent = np.min(X1), np.max(X1), np.min(Y1), np.max(Y1)
# imxy = plt.imshow(psixy, cmap=plt.cm.viridis, alpha=.9, interpolation='bilinear', extent=extent)
# fig.colorbar(imxy, shrink=0.5, aspect=5)
# plt.title(r'$ N_{Cs_{133}} |\psi_{{Cs_{133}}}(x,y)| ^2$')  # title
# plt.xlabel(r'$x(\mu m)$', rotation=0)  # x label
# plt.ylabel(r'$y(\mu m)$', rotation=0)  # y label                
#filename = r'im-densityxy-Cs.png'
#plt.savefig(filename)
#plt.show()

#cut_x_value = X1[:256,:256]
#cut_y_value = Y1[:256,:256]
cut_wave = wave[:256,:256]
cut_density = density[:256,:256]

# matplotlib.rcParams.update({'font.size': 20})
# matplotlib.rcParams['font.weight'] = 'normal'
# fig = plt.figure(figsize=(10, 8), frameon=False)
# extent = np.min(cut_x_value), np.max(cut_x_value), np.min(cut_y_value), np.max(cut_y_value)
# imxy = plt.imshow(cut_density, cmap=plt.cm.viridis, alpha=.9, interpolation='bilinear', extent=extent)
# fig.colorbar(imxy, shrink=0.5, aspect=5)
# plt.title(r'$ N_{Cs_{133}} |\psi_{{Cs_{133}}}(x,y)| ^2$')  # title
# plt.xlabel(r'$x(\mu m)$', rotation=0)  # x label
# plt.ylabel(r'$y(\mu m)$', rotation=0)  # y label                
# #filename = r'im-densityxy-Cs.png'
# #plt.savefig(filename)
# plt.show()

#print(cut_wave.shape)

#print(extent)

if __name__ == '__main__':    
    N = 8
    x1,x2 = -6,6
    Ndx = 2**N
    dx = (x2-x1)/Ndx
    print('dx',dx)

    g = 100/dx**2 
    omega = 0.8

    maxdim = 20
    cutoff = 1e-12
    krylovDim = 10
    
    with open('7_vortex_tdvp.pkl', 'rb') as f:  
        data = pickle.load(f)
    #print(npmps.inner_MPS(psi,psi))
    #data = qtt.normalize_MPS_by_integral (data, x1, x2, Dim=2)
    H_SHO = sho.make_H (N, x1, x2)
    H_SHO = npmps.get_H_2D (H_SHO)
    H_SHO = npmps.change_dtype(H_SHO, complex)
    H_SHO[0] = 0.5*H_SHO[0]

    Lz = ang.Lz_MPO (N, x1, x2)
    Lz[0] *= -1*omega

    H0 = npmps.sum_2MPO (H_SHO, Lz)

    print('Non-interacting MPO dim, before compression:',npmps.MPO_dims(H0))
    #H0 = npmps.svd_compress_MPO (H0, cutoff=1e-12)
    print('Non-interacting MPO dim:',npmps.MPO_dims(H0))



    def absSqr (a):
        return abs(a)**2
    absSqr = np.vectorize(absSqr)
    
    def absVal (a):
        return abs(a)
    absVal = np.vectorize(absVal)
    
    # Initial MPS
    #psi = qtt.normalize_MPS_by_integral (psi, x1, x2, Dim=2)
    psi_data = data
    print("data inner = ", npmps.inner_MPS(psi_data,psi_data))
    print('Initial psi dim, before compression:',npmps.MPS_dims(psi_data))
    #psi_data = npmps.svd_compress_MPS (psi_data, cutoff=1e-12)
    psi2_data = psi_sqr (psi_data)
    psi2_MPO = qtt.MPS_to_MPO (psi2_data)
    psi2_MPO[0] *= g
    H = npmps.sum_2MPO (H0, psi2_MPO)
    
    psi2_data_fit = fit_psi_sqr(psi_data, psi2_data, maxdim = maxdim, cutoff = cutoff)
    psi2_MPO_fit = qtt.MPS_to_MPO (psi2_data_fit)
    psi2_MPO_fit[0] *= g
    H_fit = npmps.sum_2MPO (H0, psi2_MPO_fit)
    #print('Initial psi dim:',npmps.MPS_dims(psi))
    #psi = qtt.normalize_MPS_by_integral (psi, x1, x2, Dim=2)
    #pltut.plot_2D_proj (psi, x1, x2, ax=None, func=absSqr, label='psi_initsqr')
    #print("data inner E = ",np.abs(npmps.inner_MPO (psi_data, psi_data, H_data)))
    print("data inner psi2 MPO E = ",npmps.inner_MPO (psi_data, psi_data, psi2_MPO))
    print("data inner psi2_fit MPO E = ",npmps.inner_MPO (psi_data, psi_data, psi2_MPO_fit))
    print("data inner H MPO E = ",npmps.inner_MPO (psi_data, psi_data, H))
    print("data inner H_fit MPO E = ",npmps.inner_MPO (psi_data, psi_data, H_fit))
    print("data inner H0 E = ",npmps.inner_MPO (psi_data, psi_data, H0))
    print("data inner H_SHO E = ",npmps.inner_MPO (psi_data, psi_data, H_SHO))


    #print("data inner = ",npmps.inner_MPS(data,data)*dx**2)
    #data = qtt.normalize_MPS_by_integral (data, x1, x2, Dim=2)
    #print(np.abs(npmps.inner_MPO (data, data, H)))
    data = qtt.normalize_MPS_by_integral (data, x1, x2, Dim=2) 
    psi2_data[0] /= dx**2
    psi2_data_fit[0] /= dx**2
    pltut.plot_2D_proj (psi2_data, x1, x2, ax=None, func=absVal, label='psi2_direct_pkl')
    pltut.plot_2D_proj (psi2_data_fit, x1, x2, ax=None, func=absVal, label='psi2_fit_pkl')
    pltut.plot_2D_proj (data, x1, x2, ax=None, func=absSqr, label='psi_pkl')

    #print("data and fortran inner = ", npmps.inner_MPS(psi,psi_data))
    



