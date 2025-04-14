import sys, copy, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../NumpyTensorTools')))
import numpy_dmrg as dmrg
import numpy as np
import matplotlib.pyplot as plt
import npmps
import plot_utility as pltut
import hamilt.hamilt_sho as sho
import qtt_tools as qtt
import hamilt.hamilt_angular_momentum as ang
import gradient_descent_GP as gdGP
import tci
import time
import pickle

def psi_sqr (psi):
    psi_op = qtt.MPS_to_MPO (psi)
    psi_op = npmps.conj (psi_op)
    res = npmps.exact_apply_MPO (psi_op, psi)
    return res

def fit_psi_sqr (psi, psi2, maxdim, cutoff):
    psi_op = qtt.MPS_to_MPO (psi)
    psi_op = npmps.conj (psi_op)
    fit = dmrg.fit_apply_MPO (psi_op, psi, psi2, numCenter=2, nsweep=1, maxdim=maxdim, cutoff=cutoff)
    return fit

def make_H_GP (H0, psi, psi2, g, maxdim_psi2, cutoff_mps2):
    psi2 = fit_psi_sqr(psi, psi2, maxdim_psi2, cutoff = cutoff_mps2)
    H_psi = qtt.MPS_to_MPO (psi2)
    H_psi[0] *= g
    H = npmps.sum_2MPO (H0, H_psi)
    return H, psi2

def gradient_descent2 (i, H0, psi, g, step_size, steps, maxdim, cutoff, maxdim_psi2, cutoff_psi2, psi2_update_length):
    psi, enss, ts = gdGP.gradient_descent_GP_MPS_new (i, steps, psi, H0, g, step_size, niter=3, maxdim=maxdim, cutoff=cutoff, maxdim_psi2=maxdim_psi2, cutoff_psi2=cutoff_psi2, linesearch=True, psi2_update_length=psi2_update_length)
    return psi, enss, ts

def fitfun_xy (x, y):
    f = np.sqrt(1/np.pi)*(x+1j*y)*np.exp(-(x**2+y**2)/2)
    return f

def inds_to_num (inds, dx, shift):
    bstr = ''
    for i in inds:
        bstr += str(i)
    return pltut.bin_to_dec (bstr, dx, shift)

def fitfun (inds):
    N = len(inds)//2
    Ndx = 2**N
    dx = (x2-x1)/Ndx
    shift = x1

    xinds, yinds = inds[:N], inds[N:]
    x = inds_to_num (xinds, dx, shift)
    y = inds_to_num (reversed(yinds), dx, shift)
    return fitfun_xy (x, y)

def get_init_other_state (N, x1, x2, maxdim, other):
    input_mps_path = f"{other}"
    with open(input_mps_path, 'rb') as file:
        data = pickle.load(file)
    mps = data
    mps = npmps.normalize_MPS (mps)
    return mps 

def get_init_state (N, x1, x2, maxdim):
    mps = tci.tci (fitfun, 2*N, 2, maxdim=maxdim)
    mps = npmps.normalize_MPS (mps)
    return mps   

def get_init_rand_state (N, x1, x2, maxdim, seed = 15, dtype=np.complex128):
    mps = npmps.random_MPS (2*N, 2, vdim=maxdim, seed=seed, dtype=np.complex128)
    mps = npmps.normalize_MPS (mps)
    return mps  

def check_hermitian (mpo):
    mm = npmps.MPO_to_matrix (mpo)
    t = np.linalg.norm(mm - mm.conj().T)
    print(t)
    assert t < 1e-10

def check_the_same (mpo1, mpo2):
    m1 = npmps.MPO_to_matrix(mpo1)
    m2 = npmps.MPO_to_matrix(mpo2)
    d = np.linalg.norm(m1-m2)
    print(d)
    assert d < 1e-10

def print_overlap (mps1, mps2):
    mps1 = copy.copy(mps1)
    mps2 = copy.copy(mps2)
    mps1 = npmps.normalize_MPS(mps1)
    mps2 = npmps.normalize_MPS(mps2)
    print('overlap',npmps.inner_MPS(mps1, mps2))

if __name__ == '__main__':    
    N = 8
    x1,x2 = -6,6
    Ndx = 2**N
    dx = (x2-x1)/Ndx
    print('dx',dx)

    g = 100/dx**2
    omega = 0.8
    maxdim = 10
    maxdim_psi2 = 1000000000
    cutoff_mps = 1e-12
    cutoff_mps2 = 1e-8
    psi2_update_length = 1
    steps= 10
    step_iter = 5

    H_SHO = sho.make_H (N, x1, x2)
    H_SHO = npmps.get_H_2D (H_SHO)
    H_SHO = npmps.change_dtype(H_SHO, complex)
    H_SHO[0] = 0.5*H_SHO[0]
    Lz = ang.Lz_MPO (N, x1, x2)
    Lz[0] *= -1*omega
    H0 = npmps.sum_2MPO (H_SHO, Lz)

    print('Non-interacting MPO dim, before compression:',npmps.MPO_dims(H0))
    H0 = npmps.svd_compress_MPO (H0, cutoff=1e-12)
    print('Non-interacting MPO dim:',npmps.MPO_dims(H0))


    def absSqr (a):
        return abs(a)**2
    absSqr = np.vectorize(absSqr)

    # Initial MPS
    psi = get_init_state (N, x1, x2, maxdim=maxdim)
    #psi = get_init_rand_state (N, x1, x2, maxdim=maxdim, seed = 15, dtype=np.complex128)
    #psi = get_init_other_state (N, x1, x2, maxdim, other)
    print('Initial psi dim, before compression:',npmps.MPS_dims(psi))
    psi = npmps.svd_compress_MPS (psi, cutoff=1e-30)
    print('Initial psi dim:',npmps.MPS_dims(psi))
    #psi = qtt.normalize_MPS_by_integral (psi, x1, x2, Dim=2)
    
    dt = dx**2/2
    psi_GD2, ens_GD2, ts2 = gradient_descent2 (step_iter, H0, psi, g, step_size=dt, steps=steps, maxdim=maxdim, cutoff=cutoff_mps, maxdim_psi2=maxdim_psi2, cutoff_psi2=cutoff_mps2, psi2_update_length=psi2_update_length)
