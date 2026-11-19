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
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib

def psi_sqr (psi):
    psi_op = qtt.MPS_to_MPO (psi)
    res = npmps.exact_apply_MPO (psi_op, psi)
    res = npmps.svd_compress_MPS (res, cutoff=1e-12)
    return res

# psi2 is the initial guess of the result
def fit_psi_sqr (psi, psi2, maxdim, cutoff=1e-16):
    psi_op = qtt.MPS_to_MPO (psi)
    psi_op = npmps.conj (psi_op)
    fit = dmrg.fit_apply_MPO (psi_op, psi, psi2, numCenter=1, nsweep=1, maxdim=maxdim, cutoff=cutoff)
    #psi2_exact = psi_sqr (psi)
    #print_overlap (psi2_exact, fit)
    return fit

def make_H_GP (H0, psi2, g):
    #t1 = time.time()                                # time
    #psi2 = psi_sqr (psi)
   # psi2 = fit_psi_sqr(psi, psi2, maxdim = maxdim, cutoff = cutoff)
   # for i in range(len(psi2)):
   #     print(psi2[i].shape)
   # t2 = time.time()                                # time
   # print('psi2 time',(t2-t1))                      # time

    H_psi = qtt.MPS_to_MPO (psi2)
    H_psi[0] *= g
    H = npmps.sum_2MPO (H0, H_psi)
    #H = npmps.svd_compress_MPO (H, cutoff=1e-12)
    return H


def fitfun_xy (x, y):
    f = cut_wave[y][x]
    return f

def fitfun2_xy (x, y):
    f = cut_density[y][x]
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
x_value =  position[:,0].reshape(int(2**8 + 1),int(2**8 + 1))
y_value = position[:,1].reshape(int(2**8 + 1),int(2**8 + 1))

wave = (real_wave + 1j*imag_wave).reshape(int(2**8 + 1),int(2**8 + 1))

density = (real_wave**2 + imag_wave**2).reshape(int(2**8 + 1),int(2**8 + 1))


X1 = x_value
Y1 = y_value
psixy = density


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

cut_x_value = X1[:256,:256]
cut_y_value = Y1[:256,:256]
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
    psi = npmps.dataarr_to_mps(dx*cut_wave.flatten(),16,2)
    print("load fortran inner = ", npmps.inner_MPS(psi,psi))
    #psi = qtt.normalize_MPS_by_integral (psi, x1, x2, Dim=2)
    print('Initial psi dim, before compression:',npmps.MPS_dims(psi))
    #psi = npmps.svd_compress_MPS (psi, cutoff=1e-12)
    #psi2 = psi_sqr (psi)
    psi2 = npmps.dataarr_to_mps(dx**2*cut_density.flatten(),16,2)
    pltut.plot_2D_proj (psi2, x1, x2, ax=None, func=absVal, label='psi_svd')
    #print("load fortran inner psi2 = ", npmps.inner_MPS(psi2,psi2))

    #psi2_fit = fit_psi_sqr(psi, psi2, maxdim = maxdim, cutoff = cutoff)
    #print("load fortran inner psi2_fit = ", npmps.inner_MPS(psi2_fit,psi2_fit))
    #pltut.plot_2D_proj (psi2_fit, x1, x2, ax=None, func=absVal, label='psi_svd_fit')


    #psi2_mpo = qtt.MPS_to_MPO (psi2)
    #psi2_fit_mpo = qtt.MPS_to_MPO (psi2_fit)
    H = make_H_GP(H0, psi2, g)
    #print('Initial psi dim:',npmps.MPS_dims(psi))
    #psi = qtt.normalize_MPS_by_integral (psi, x1, x2, Dim=2)
    #print("load fortran psi2 mpo E = ",g*npmps.inner_MPO (psi, psi, psi2_mpo))
    #print("load fortran fit psi2 mpo E = ",g*npmps.inner_MPO (psi, psi, psi2_fit_mpo))
    #psi = qtt.normalize_MPS_by_integral (psi, x1, x2, Dim=2) 
    #pltut.plot_2D_proj (psi2, x1, x2, ax=None, func=absVal, label='psi_svd')
    #pltut.plot_2D_proj (psi2_fit, x1, x2, ax=None, func=absVal, label='psi_svd_fit')
    #pltut.plot_2D_proj (psi2, x1, x2, ax=None, func=None, label='psi_svd')
    print("load fortran psi E = ",np.abs(npmps.inner_MPO (psi, psi, H)))
    #----------------------TCI---------------------------------------------------------------------------------
    
    # H_SHO = sho.make_H (N, x1, x2)
    # H_SHO = npmps.get_H_2D (H_SHO)
    # H_SHO = npmps.change_dtype(H_SHO, complex)
    # H_SHO[0] = 0.5*H_SHO[0]

    # Lz = ang.Lz_MPO (N, x1, x2)
    # Lz[0] *= -1*omega

    # H0 = npmps.sum_2MPO (H_SHO, Lz)

    # print('Non-interacting MPO dim, before compression:',npmps.MPO_dims(H0))

    # def absSqr (a):
    #     return abs(a)**2
    # absSqr = np.vectorize(absSqr)

    # # Initial MPS
    # psi_tci = get_init_state (N, x1, x2, maxdim=maxdim)
    # print("load TCI inner = ", npmps.inner_MPS(psi_tci,psi_tci))
    # #psi = qtt.normalize_MPS_by_integral (psi, x1, x2, Dim=2)
    # print('Initial psi dim, before compression:',npmps.MPS_dims(psi))
    # #psi2 = psi_sqr (psi)
    # psi2_tci = get_init_statesqr(N, x1, x2, maxdim=maxdim)
    # psi2_tci = fit_psi_sqr(psi_tci, psi2_tci, maxdim = maxdim, cutoff = cutoff)
    # psi2_tci_mpo = qtt.MPS_to_MPO (psi2_tci)
    # H_tci = make_H_GP (H0, psi_tci, psi2_tci, g, maxdim=maxdim)
    # #print('Initial psi dim:',npmps.MPS_dims(psi))
    # #psi = qtt.normalize_MPS_by_integral (psi, x1, x2, Dim=2)
    # pltut.plot_2D_proj (psi_tci, x1, x2, ax=None, func=absSqr, label='psi_tci')
    # print("load fortran psi E = ",g*npmps.inner_MPO (psi_tci, psi_tci, psi2_tci_mpo))
    # #print("load TCI and SVD inner = ", npmps.inner_MPS(psi_tci,psi))

    #print("original psi2 E=",dx**2*np.sum((cut_density*cut_density).flatten())*100)
    