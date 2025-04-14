from ncon import ncon
import numpy as np
import npmps

def make_t_matrix (n, rescale=1.):
    return np.array([[0.,0.],[0.,rescale*2**n]])

def make_x_tensor (n, rescale=1.):
    I = np.array([[1.,0.],[0.,1.]])
    T = np.zeros((2,2,2,2))     # k1, ipr, i, k2
    T[0,:,:,0] = I
    T[1,:,:,1] = I
    T[1,:,:,0] = make_t_matrix (n, rescale)
    return T

def make_x_MPS_tensor (n, rescale=1.):
    T = np.zeros((2,2,2))     # k1, i, k2
    T[0,:,0] = [1.,1.]
    T[1,:,1] = [1.,1.]
    T[1,:,0] = [0.,rescale*2**n]
    return T

def make_x_LR (shift=0.):
    L = np.array([shift,1.])
    R = np.array([1.,0.])
    return L, R

def make_x_mps (N, x1, x2):
    Ndx = 2**N
    rescale = (x2-x1)/Ndx
    shift = x1

    mps = [make_x_MPS_tensor (n, rescale) for n in range(N)]
    L, R = make_x_LR (shift)
    mps[0] = ncon ([L,mps[0]], ((1,), (1,-1,-2))).reshape((1,2,2))
    mps[-1] = ncon ([R,mps[-1]], ((1,), (-1,-2,1))).reshape((2,2,1))
    return mps

def make_x_mpo (N, x1, x2):
    Ndx = 2**N
    rescale = (x2-x1)/Ndx
    shift = x1

    mpo = [make_x_tensor (n, rescale) for n in range(N)]
    L, R = make_x_LR (shift)
    mpo = npmps.absort_LR (mpo, L, R)
    return mpo

def make_xsqr_mps (N, x1, x2):
    mps = make_x_mps (N, x1, x2)
    mpo = make_x_mpo (N, x1, x2)
    re = npmps.exact_apply_MPO (mpo, mps)
    return re

def make_xsqr_mpo (N, x1, x2):
    Ndx = 2**N
    rescale = (x2-x1)/Ndx
    shift = x1

    H = []
    for n in range(N):
        x_tensor = make_x_tensor (n, rescale)
        x2_tensor = npmps.prod_MPO_tensor (x_tensor, x_tensor)
        H.append(x2_tensor)

    L_x, R_x = make_x_LR (shift)
    L = ncon([L_x,L_x], ((-1,),(-2,))).reshape(-1,)
    R = ncon([R_x,R_x], ((-1,),(-2,))).reshape(-1,)
    H = npmps.absort_LR (H, L, R)
    return H
