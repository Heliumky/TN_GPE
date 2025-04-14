import os, sys, copy
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import scipy as sp
from ncon import ncon
import npmps
import lanczos

# E is the left environment tensor
def left_environment_contraction_mpo (E, A1, W, A2):
    #  ----
    #  |  |--- -1
    #  |  |
    #  |  |--- -2
    #  |  |        -3
    #  |  |    1    |
    #  |  |---------O--- -4
    #  ----
    tmp = ncon((E,A1),((-1,-2,1),(1,-3,-4)))
    #  ----
    #  |  |--- -1
    #  |  |        -2
    #  |  |         |
    #  |  |---------O--- -3
    #  |  |    1    |
    #  |  |         | 2
    #  |  |---------O--- -4
    #  ----
    tmp = ncon((tmp,W),((-1,1,2,-4),(1,-2,2,-3)))
    #  ----
    #  |  |---------O--- -1
    #  |  |    1    |
    #  |  |         | 2
    #  |  |---------O--- -2
    #  |  |         |
    #  |  |         |
    #  |  |---------O--- -3
    #  ----
    tmp = ncon((tmp,np.conj(A2)),((1,2,-2,-3),(1,2,-1)))
    return tmp

# E is the righ environment tensor
def right_environment_contraction_mpo (E, A1, W, A2):
    #                 ----
    #           -1 ---|  |
    #                 |  |
    #           -2 ---|  |
    #       -3        |  |
    #        |    1   |  |
    #  -4 ---O--------|  |
    #                 ----
    tmp = ncon((E,A1),((-1,-2,1),(-4,-3,1)))
    #                 ----
    #           -1 ---|  |
    #       -2        |  |
    #        |        |  |
    #  -3 ---O--------|  |
    #        |    1   |  |
    #        | 2      |  |
    #  -4 ---O--------|  |
    #                 ----
    tmp = ncon((tmp,W),((-1,1,2,-4),(-3,-2,2,1)))
    #                 ----
    #  -1 ---O--------|  |
    #        |    1   |  |
    #        | 2      |  |
    #  -2 ---O--------|  |
    #        |        |  |
    #        |        |  |
    #  -3 ---O--------|  |
    #                 ----
    tmp = ncon((tmp,np.conj(A2)),((1,2,-2,-3),(-1,2,1)))
    return tmp

class LR_envir_tensors_mpo:
    def __init__ (self, N, dtype=float):
        self.dtype = dtype
        self.centerL = 0
        self.centerR = N-1
        self.LR = dict()
        for i in range(-1,N+1):
            self.LR[i] = None
        self.LR[-1] = np.ones((1,1,1),dtype=dtype)
        self.LR[N] = np.ones((1,1,1),dtype=dtype)

    def __getitem__(self, i):
        if i >= self.centerL and i <= self.centerR:
            print('environment tensor is not updated')
            print('centerL,centerR,i =',self.centerL,self.centerR,i)
            raise Exception
        return self.LR[i]

    def delete (self, i):
        self.centerL = min(self.centerL, i)
        self.centerR = max(self.centerR, i)

    # self.LR[i-1] and self.LR[i+1] are the left and right environments for the ith site
    # self.LR[i] for i=-1,...,centerL-1 are left environments;
    # self.LR[i] for i=centerR+1,...,N are right environments
    # MPS tensor indices = (l,i,r)
    # MPO tensor indices = (l,ip,i,r)
    # Left or right environment tensor = (up, mid, down)
    def update_LR (self, mps1, mps2, mpo, centerL, centerR=None):
        # Set dtype
        dtype = max(mps1[0].dtype, mps2[0].dtype, mpo[0].dtype)
        if dtype != self.dtype:
            for i in self.LR:
                if self.LR[i] != None:
                    self.LR[i] = self.LR[i].astype(dtype)
            self.dtype = dtype

        if centerR == None:
            centerR = centerL
        if centerL > centerR+1:
            print('centerL cannot be larger than centerR+1')
            print('centerL, centerR =',centerL, centerR)
            raise Exception
        # Update the left environments
        for p in range(self.centerL, centerL):
            self.LR[p] = left_environment_contraction_mpo (self.LR[p-1], mps1[p], mpo[p], mps2[p])
        # Update the right environments
        for p in range(self.centerR, centerR, -1):
            self.LR[p] = right_environment_contraction_mpo (self.LR[p+1], mps1[p], mpo[p], mps2[p])

        self.centerL = centerL
        self.centerR = centerR

# E is the left environment tensor
def left_environment_contraction_mps (E, A1, A2):
    #  ----
    #  |  |--- -1
    #  |  |
    #  |  |
    #  |  |        -2
    #  |  |         |
    #  |  |---------O--- -3
    #  ----    1
    tmp = ncon((E,A1),((-1,1),(1,-2,-3)))
    #  ----
    #  |  |---------O--- -1
    #  |  |    1    |
    #  |  |         |
    #  |  |       2 |
    #  |  |         |
    #  |  |         |
    #  |  |---------O--- -2
    #  ----
    tmp = ncon((tmp,np.conj(A2)),((1,2,-2),(1,2,-1)))
    return tmp

# E is the right environment tensor
def right_environment_contraction_mps (E, A1, A2):
    #                 ----
    #           -1 ---|  |
    #                 |  |
    #                 |  |
    #       -2        |  |
    #        |    1   |  |
    #  -3 ---O--------|  |
    #                 ----
    tmp = ncon((E,A1),((-1,1),(-3,-2,1)))
    #                 ----
    #  -1 ---O--------|  |
    #        |    1   |  |
    #        |        |  |
    #        | 2      |  |
    #        |        |  |
    #        |        |  |
    #  -2 ---O--------|  |
    #                 ----
    tmp = ncon((tmp,np.conj(A2)),((1,2,-2),(-1,2,1)))
    return tmp

class LR_envir_tensors_mps:
    def __init__ (self, N, mps1, mps2, dtype=float):
        self.dtype = dtype
        self.centerL = 0
        self.centerR = N-1
        self.LR = dict()
        for i in range(-1,N+1):
            self.LR[i] = None
        self.LR[-1] = np.ones((1,1),dtype=dtype)
        self.LR[N] = np.ones((1,1),dtype=dtype)

    def __getitem__(self, i):
        if i >= self.centerL and i <= self.centerR:
            print('environment tensor is not updated')
            print('centerL,centerR,i =',self.centerL,self.centerR,i)
            raise Exception
        return self.LR[i]

    def delete (self, i):
        self.centerL = min(self.centerL, i)
        self.centerR = max(self.centerR, i)

    # self.LR[i-1] and self.LR[i+1] are the left and right environments for the ith site
    # self.LR[i] for i=-1,...,centerL-1 are left environments;
    # self.LR[i] for i=centerR+1,...,N are right environments
    # MPS tensor indices = (l,i,r)
    # Left or right environment tensor = (up, mid, down)
    def update_LR (self, mps1, mps2, centerL, centerR=None):
        # Set dtype
        dtype = max(mps1[0].dtype, mps2[0].dtype, mpo[0].dtype)
        if dtype != self.dtype:
            for i in self.LR:
                if self.LR[i] != None:
                    self.LR[i] = self.LR[i].astype(dtype)
            self.dtype = dtype

        if centerR == None:
            centerR = centerL
        if centerL > centerR+1:
            print('centerL cannot be larger than centerR+1')
            print('centerL, centerR =',centerL, centerR)
            raise Exception
        # Update the left environments
        for p in range(self.centerL, centerL):
            self.LR[p] = left_environment_contraction_mps (self.LR[p-1], mps1[p], mps2[p])
        # Update the right environments
        for p in range(self.centerR, centerR, -1):
            self.LR[p] = right_environment_contraction_mps (self.LR[p+1], mps1[p], mps2[p])

        self.centerL = centerL
        self.centerR = centerR

# An effective Hamiltonian must:
# 1. Inherit <cytnx.LinOp> class
# 2. Has a function <matvec> that implements H|psi> operation
class eff_Hamilt_2sites (sp.sparse.linalg.LinearOperator):
    def __init__ (self, L, M1, M2, R):
        assert L.dtype == R.dtype
        self.L = L
        self.R = R
        self.M1 = M1
        self.M2 = M2

        dim = L.shape[0] * M1.shape[1] * M2.shape[1] * R.shape[0]
        self.shape = (dim,dim)
        self.dtype = L.dtype
        self.vshape = (L.shape[0], M1.shape[1], M2.shape[1], R.shape[0])

    # Define H|v> operation
    #
    #              2    3
    # |v> =        |____|
    #        1 ---(______)--- 4
    def apply (self, v):
        #  ----
        #  |  |--- -1
        #  |  |
        #  |  |--- -2
        #  |  |        -3   -4
        #  |  |         |____|
        #  |  |--------(______)--- -5
        #  ----    1
        tmp = ncon((self.L, v),((-1,-2,1),(1,-3,-4,-5)))
        #  ----
        #  |  |--- -1
        #  |  |        -2
        #  |  |         |
        #  |  |---------O--- -3
        #  |  |    1    |        -4
        #  |  |       2 |_________|
        #  |  |--------(__________)--- -5
        #  ----
        tmp = ncon((tmp,self.M1),((-1,1,2,-4,-5),(1,-2,2,-3)))
        #  ----
        #  |  |--- -1
        #  |  |        -2        -3
        #  |  |         |         |
        #  |  |---------O---------O--- -4
        #  |  |         |    1    |
        #  |  |         |_________| 2
        #  |  |--------(__________)--- -5
        #  ----
        tmp = ncon((tmp,self.M2),((-1,-2,1,2,-5),(1,-3,2,-4)))
        #  ----                           ----
        #  |  |--- -1               -4 ---|  |
        #  |  |        -2        -3       |  |
        #  |  |         |         |       |  |
        #  |  |---------O---------O-------|  |
        #  |  |         |         |   1   |  |
        #  |  |         |_________|       |  |
        #  |  |--------(__________)-------|  |
        #  ----                       2   ----
        tmp = ncon((tmp,self.R),((-1,-2,-3,1,2),(-4,1,2)))
        return tmp

    def _matvec (self, v):
        v = v.reshape(self.vshape)
        return self.apply(v)

# An effective Hamiltonian must:
# 2. Has a function <matvec> that implements H|psi> operation
class eff_Hamilt_1site (sp.sparse.linalg.LinearOperator):
    def __init__ (self, L, M, R):
        assert L.dtype == R.dtype
        self.L = L
        self.R = R
        self.M = M
        self.dtype = L.dtype
        self.vshape = (L.shape[0], M.shape[1], R.shape[0])
        dim = np.prod(self.vshape)
        self.shape = (dim,dim)

    # Define H|v> operation
    #
    #             2
    # |v> =       |
    #        1 ---O--- 3
    def apply (self, v):
        #  ----
        #  |  |--- -1
        #  |  |
        #  |  |--- -2
        #  |  |        -3
        #  |  |         |
        #  |  |---------O--- -4
        #  ----    1
        tmp = ncon((self.L, v),((-1,-2,1),(1,-3,-4)))
        #  ----
        #  |  |--- -1
        #  |  |        -2
        #  |  |         |
        #  |  |---------O--- -3
        #  |  |    1    |
        #  |  |       2 |
        #  |  |---------O--- -4
        #  ----
        tmp = ncon((tmp,self.M),((-1,1,2,-4),(1,-2,2,-3)))
        #  ----                  ----
        #  |  |--- -1      -3 ---|  |
        #  |  |        -2        |  |
        #  |  |         |        |  |
        #  |  |---------O--------|  |
        #  |  |         |    1   |  |
        #  |  |         |        |  |
        #  |  |---------O--------|  |
        #  ----              2   ----
        tmp = ncon((tmp,self.R),((-1,-2,1,2),(-3,1,2)))
        return tmp

    def _matvec (self, v):
        v = v.reshape(self.vshape)
        return self.apply(v)

# An effective Hamiltonian must:
# 1. Inherit <cytnx.LinOp> class
# 2. Has a function <matvec> that implements H|psi> operation
class eff_Hamilt_0site (sp.sparse.linalg.LinearOperator):
    def __init__ (self, L, R):
        assert L.dtype == R.dtype
        self.L = L
        self.R = R
        self.dtype = L.dtype
        self.vshape = (L.shape[0], R.shape[0])
        dim = np.prod(self.vshape)
        self.shape = (dim,dim)

    # Define H|v> operation
    #
    # |v> = 1 ---O--- 3
    def apply (self, v):
        #  ----
        #  |  |--- -1
        #  |  |
        #  |  |--- -2
        #  |  |
        #  |  |
        #  |  |---------O--- -3
        #  ----    1
        tmp = ncon((self.L, v),((-1,-2,1),(1,-3)))
        #  ----                  ----
        #  |  |--- -1      -2 ---|  |
        #  |  |                  |  |
        #  |  |                  |  |
        #  |  |------------------|  |
        #  |  |              1   |  |
        #  |  |                  |  |
        #  |  |---------O--------|  |
        #  ----              2   ----
        tmp = ncon((tmp,self.R),((-1,1,2),(-2,1,2)))
        return tmp

    def _matvec (self, v):
        v = v.reshape(self.vshape)
        return self.apply(v)

def get_eff_psi (psi, p, numCenter):
    if numCenter == 2:
        #
        #        -2    -3
        #         |  1  |
        #   -1 ---O-----O--- -4
        phi = ncon((psi[p],psi[p+1]),((-1,-2,1),(1,-3,-4)))
    elif numCenter == 1:
        #
        #         2
        #         |  
        #    1 ---O--- 3
        phi = psi[p]
    return phi

def get_sweeping_sites (N, numCenter):
    if numCenter == 2:
        sites = [range(N-1), range(N-2,-1,-1)]
    elif numCenter == 1:
        sites = [range(N), range(N-1,-1,-1)]
    return sites

def get_eff_H (LR, H, p, numCenter):
    if numCenter == 2:
        effH = eff_Hamilt_2sites (LR[p-1], H[p], H[p+1], LR[p+2])
    elif numCenter == 1:
        effH = eff_Hamilt_1site (LR[p-1], H[p], LR[p+1])
    elif numCenter == 0:
        effH = eff_Hamilt_0site (LR[p-1], LR[p])
    return effH

'''def update_mps_1site_old (A, i, mps, toRight):
    mps = copy.copy(mps)
    if toRight:
        if i == len(mps)-1:   # last site
            mps[i] = A
        else:
            #
            #           2                    2    
            #           |                    |  3   1     
            #  A = 1 ---A--- 3   =>     1 ---Q---------R--- 2
            mps[i], R = npmps.qr_decomp (A, rowrank=2, toRight=True)
            #
            #              -2
            #           1   |
            #  -1 ---R------O--- -3
            mps[i+1] = ncon((R,mps[i+1]), ((-1,1),(1,-2,-3)))
    else:
        if i == 0:          # first site
            mps[i] = A
        else:
            #
            #           2                              2
            #           |                       2   1  |
            #  A = 1 ---A--- 3   =>     1 ---R---------Q--- 3
            mps[i], R = npmps.qr_decomp (A, rowrank=1, toRight=False)
            #
            #       -2
            #        |   1
            #  -1 ---O------R--- -3
            mps[i-1] = ncon((mps[i-1],R), ((-1,-2,1),(1,-3)))
    return mps

def update_mps_1site (A, i, mps, toRight, maxdim=100000000, cutoff=0.):
    mps = copy.copy(mps)
    if toRight:
        if i == len(mps)-1:   # last site
            mps[i] = A
        else:
            #
            #           2                    2    
            #           |                    |  3   1     
            #  A = 1 ---A--- 3   =>     1 ---Q---------R--- 2
            mps[i], R, err = npmps.truncate_svd2 (A, rowrank=2, toRight=toRight, maxdim=maxdim, cutoff=cutoff)
            #
            #              -2
            #           1   |
            #  -1 ---R------O--- -3
            mps[i+1] = ncon((R,mps[i+1]), ((-1,1),(1,-2,-3)))
    else:
        if i == 0:          # first site
            mps[i] = A
        else:
            #
            #           2                              2
            #           |                       2   1  |
            #  A = 1 ---A--- 3   =>     1 ---R---------Q--- 3
            R, mps[i], err = npmps.truncate_svd2 (A, rowrank=1, toRight=toRight, maxdim=maxdim, cutoff=cutoff)
            #
            #       -2
            #        |   1
            #  -1 ---O------R--- -3
            mps[i-1] = ncon((mps[i-1],R), ((-1,-2,1),(1,-3)))
    return mps

def update_mpo_1site (A, i, mpo, toRight, maxdim=100000000, cutoff=0.):
    mpo = copy.copy(mpo)
    if toRight:
        if i == len(mpo)-1:   # last site
            mpo[i] = A
        else:
            #
            #           2                    2    
            #           |                    |  4   1     
            #  A = 1 ---A--- 4   =>     1 ---Q---------R--- 2
            #           |                    |
            #           3                    3
            mpo[i], R, err = npmps.truncate_svd2 (A, rowrank=3, toRight=toRight, maxdim=maxdim, cutoff=cutoff)
            #
            #              -2
            #           1   |
            #  -1 ---R------O--- -4
            #               |
            #              -3
            mpo[i+1] = ncon((R,mpo[i+1]), ((-1,1),(1,-2,-3,-4)))
    else:
        if i == 0:          # first site
            mpo[i] = A
        else:
            #
            #           2                              2
            #           |                       2   1  |
            #  A = 1 ---A--- 4   =>     1 ---R---------Q--- 4
            #           |                              |
            #           3                              3
            R, mpo[i], err = npmps.truncate_svd2 (A, rowrank=1, toRight=toRight, maxdim=maxdim, cutoff=cutoff)
            #
            #       -2
            #        |   1
            #  -1 ---O------R--- -4
            #        |
            #       -3
            mpo[i-1] = ncon((mpo[i-1],R), ((-1,-2,-3,1),(1,-4)))
    return mpo'''

def orthogonalize_MPS_tensor (mps, i, toRight, maxdim=100000000, cutoff=0.):
    res = copy.copy(mps)
    if toRight:
        # If i is the last site, do nothing
        if i != len(mps)-1:
            #
            #           2                     2    
            #           |                     |  3   1     
            #  A = 1 ---A1--- 3   =>     1 ---Q---------R--- 2
            res[i], R, err = npmps.truncate_svd2 (res[i], rowrank=2, toRight=toRight, maxdim=maxdim, cutoff=cutoff)
            #res[i], R = npmps.qr_decomp (res[i], rowrank=2, toRight=True)
            #
            #              -2
            #           1   |
            #  -1 ---R------A2--- -3
            res[i+1] = ncon((R,res[i+1]), ((-1,1),(1,-2,-3)))
    else:
        # If i is the first site, do nothing
        if i != 0:
            #
            #           2                               2
            #           |                       2   1   |
            #  A = 1 ---A1--- 3   =>     1 ---R---------Q--- 3
            R, res[i], err = npmps.truncate_svd2 (res[i], rowrank=1, toRight=toRight, maxdim=maxdim, cutoff=cutoff)
            #res[i], R = npmps.qr_decomp (res[i], rowrank=1, toRight=False)
            #
            #       -2
            #        |   1
            #  -1 ---A2------R--- -3
            res[i-1] = ncon((res[i-1],R), ((-1,-2,1),(1,-3)))
    return res

def dmrg (numCenter, psi, H, maxdims, cutoff, krylovDim=20, verbose=False):
    # Check the MPS and the MPO
    assert (len(psi) == len(H))
    npmps.check_MPO_links (H)
    npmps.check_MPS_links (psi)
    # Set dtype
    dtype = max(psi[0].dtype, H[0].dtype)

    psi = copy.copy(psi)

    # Define the links to update for a sweep
    # First do a left-to-right and then a right-to-left sweep
    Nsites = len(psi)
    ranges = get_sweeping_sites (Nsites, numCenter)

    # Get the environment tensors
    LR = LR_envir_tensors_mpo (Nsites, dtype)
    LR.update_LR (psi, psi, H, 0)

    ens, terrs = [], []
    N_update = len(ranges[0]) + len(ranges[1])
    for k in range(len(maxdims)):                                                            # For each sweep
        maxdim = maxdims[k]                                                                     # Read bond dimension
        terr = 0.
        for lr in [0,1]:
            for p in ranges[lr]:
                #
                #         2                   2      3
                #         |                   |______|
                #    1 ---O--- 3   or   1 ---(________)--- 4
                phi = get_eff_psi (psi, p, numCenter)
                dims = phi.shape
                phi = phi.reshape(-1)

                # Update the environment tensors
                if numCenter == 2:
                    LR.update_LR (psi, psi, H, p, p+1)
                elif numCenter == 1:
                    LR.update_LR (psi, psi, H, p)

                # Define the effective Hamiltonian
                effH = get_eff_H (LR, H, p, numCenter)

                # Find the ground state for the current bond
                en, phi = lanczos.lanczos_ground_state (effH, phi, k=krylovDim, dtype=dtype)
                phi = phi.reshape(dims)
                phi = phi / np.linalg.norm(phi)

                # Update tensors
                toRight = (lr==0)
                if numCenter == 2:
                    psi[p], psi[p+1], err = npmps.truncate_svd2 (phi, rowrank=2, toRight=toRight, maxdim=maxdim, cutoff=cutoff)
                    terr += err
                    LR.delete(p)
                    LR.delete(p+1)
                elif numCenter == 1:
                    psi[p] = phi
                    psi = orthogonalize_MPS_tensor (psi, p, toRight, maxdim=maxdim, cutoff=cutoff)
                    LR.delete(p)
                else:
                    raise Exception

        if verbose:
            print('Sweep',k,', maxdim='+str(maxdim),', MPS dim='+str(max(npmps.MPS_dims(psi))))
            print('\t','energy =',en, terr)

        ens.append(en);
        terrs.append (terr/N_update)
    return psi, ens, terrs

# Perform exp(-dt*H)|psi> by TDVP
def tdvp (numCenter, psi, H, dt, maxdims, cutoff, krylovDim=20, verbose=False):
    # Check the MPS and the MPO
    assert (len(psi) == len(H))
    npmps.check_MPO_links (H)
    npmps.check_MPS_links (psi)
    # Set dtype
    dtype = max(psi[0].dtype, H[0].dtype)

    psi = copy.copy(psi)

    # Define the links to update for a sweep
    # First do a left-to-right and then a right-to-left sweep
    Nsites = len(psi)
    ranges = get_sweeping_sites (Nsites, numCenter)

    # Get the environment tensors
    LR = LR_envir_tensors_mpo (Nsites)
    LR.update_LR (psi, psi, H, 0, -1)
    en = np.inner(LR[-1].reshape(-1), LR[0].reshape(-1))

    ens, terrs = [], []
    N_update = len(ranges[0]) + len(ranges[1])
    for k in range(len(maxdims)):                                                            # For each sweep
        maxdim = maxdims[k]                                                                     # Read bond dimension
        terr = []
        for lr in [0,1]:
            for p in ranges[lr]:
                #
                #         2                   2      3
                #         |                   |______|
                #    1 ---O--- 3   or   1 ---(________)--- 4
                phi = get_eff_psi (psi, p, numCenter)

                # Update the environment tensors
                if numCenter == 2:
                    LR.update_LR (psi, psi, H, p, p+1)
                elif numCenter == 1:
                    LR.update_LR (psi, psi, H, p)

                # --- Forward propagation ---
                # Define the effective Hamiltonian
                effH = get_eff_H (LR, H, p, numCenter)
                # Apply exp(-dt*H) to phi
                dims = phi.shape
                phi = phi.reshape(-1)
                phi = lanczos.lanczos_expm_multiply (effH, phi, -0.5*dt, k=krylovDim, dtype=dtype)
                phi = phi.reshape(dims)
                # Normalize phi
                phi = phi / np.linalg.norm(phi)

                # Update tensors
                toRight = (lr==0)
                if numCenter == 2:
                    psi[p], psi[p+1], err = npmps.truncate_svd2 (phi, rowrank=2, toRight=toRight, maxdim=maxdim, cutoff=cutoff)
                    terr.append(err)
                    LR.delete(p)
                    LR.delete(p+1)
                elif numCenter == 1:
                    psi[p] = phi
                    LR.delete(p)
                else:
                    raise Exception

                # --- Backward propagation ---
                # The end sites do not need to do backward propagation
                if (lr == 0 and p+numCenter-1 != Nsites-1) or (lr == 1 and p != 0):
                    next_p = p + toRight
                    if numCenter == 2:
                        # wavefunction for back propagation
                        phi = psi[next_p]
                        # Update the environment tensors
                        LR.update_LR (psi, psi, H, next_p)
                    elif numCenter == 1:
                        # wavefunction for back propagation
                        if toRight:
                            psi[p], R, err = npmps.truncate_svd2 (psi[p], rowrank=2, toRight=toRight, maxdim=maxdim, cutoff=cutoff)
                        else:
                            R, psi[p], err = npmps.truncate_svd2 (psi[p], rowrank=1, toRight=toRight, maxdim=maxdim, cutoff=cutoff)
                        phi = R
                        LR.delete(p)
                        # Update the environment tensors
                        LR.update_LR (psi, psi, H, next_p, next_p-1)

                    # Define the effective Hamiltonian
                    effH = get_eff_H (LR, H, next_p, numCenter-1)
                    # Apply exp(-dt*H) to phi
                    dims = phi.shape
                    phi = phi.reshape(-1)
                    phi = lanczos.lanczos_expm_multiply (effH, phi, 0.5*dt, k=krylovDim, dtype=dtype)
                    phi = phi.reshape(dims)
                    # Normalize phi
                    phi = phi / np.linalg.norm(phi)

                    # Update tensors
                    if numCenter == 2:
                        psi[next_p] = phi
                        LR.delete(next_p)
                    elif numCenter == 1:
                        if toRight:
                            #
                            #                -2
                            #             1   |
                            #  -1 ---phi------A--- -3
                            psi[p+1] = ncon((phi,psi[p+1]), ((-1,1),(1,-2,-3)))
                            LR.delete(p+1)
                        else:
                            #
                            #       -2
                            #        |   1
                            #  -1 ---A------phi--- -3
                            psi[p-1] = ncon((psi[p-1],phi), ((-1,-2,1),(1,-3)))
                            LR.delete(p-1)
                    else:
                        raise Exception

        if verbose:
            print('Sweep',k,', chi='+str(chi),', maxdim='+str(max(npmps.mps_dims(psi))))
            terr = np.mean(terr)
            print('\t','energy =',en, terr)

        ens.append(en);
        terrs.append (np.mean(terr))
    return psi, ens, terrs

# Compute mpo|mps> approximately
# It works better if the initial fit MPS is close to mps0
def fit_apply_MPO (mpo, mps, fitmps, numCenter, nsweep=1, maxdim=100000000, cutoff=0., normalize=False, returnLR=False):
    # Check the MPS and the MPO
    assert (len(mpo) == len(mps) == len(fitmps))
    npmps.check_MPO_links (mpo)
    npmps.check_MPS_links (mps)
    npmps.check_MPS_links (fitmps)

    # Define the links to update for a sweep
    # First do a left-to-right and then a right-to-left sweep
    N = len(mps)
    ranges = get_sweeping_sites (N, numCenter)

    fitmps = copy.copy(fitmps)

    # Get the environment tensors
    LR = LR_envir_tensors_mpo (N)

    for k in range(nsweep):                                                            # For each sweep
        for lr in [0,1]:
            for p in ranges[lr]:
                # Update the environment tensors
                if numCenter == 2:
                    LR.update_LR (mps, fitmps, mpo, p, p+1)
                elif numCenter == 1:
                    LR.update_LR (mps, fitmps, mpo, p)

                # Define the effective Hamiltonian
                effH = get_eff_H (LR, mpo, p, numCenter)

                #
                #         2                   2      3
                #         |                   |______|
                #    1 ---O--- 3   or   1 ---(________)--- 4
                phi_old = get_eff_psi (mps, p, numCenter)

                # Find the new state for the current bond
                phi = effH.apply (phi_old)
                if normalize:
                    phi = phi / np.linalg.norm(phi)

                # Update tensors
                toRight = (lr==0)
                if numCenter == 2:
                    fitmps[p], fitmps[p+1], err = npmps.truncate_svd2 (phi, rowrank=2, toRight=toRight, maxdim=maxdim, cutoff=cutoff)
                    LR.delete(p)
                    LR.delete(p+1)
                elif numCenter == 1:
                    fitmps[p] = phi
                    fitmps = orthogonalize_MPS_tensor (fitmps, p, toRight, maxdim=maxdim, cutoff=cutoff)
                    LR.delete(p)
                else:
                    raise Exception

    #LR.update_LR (mps, fitmps, mpo, 0, -1)
    #overlap = np.inner (LR[-1].reshape(-1), LR[0].reshape(-1))

    if returnLR:
        return fitmps, LR
    else:
        return fitmps


def get_sweep_sites_new (N, numCenter, site_beg, length=-1):
    if length == -1:
        length = N
    first = max(site_beg - length, 0)
    if numCenter == 2:
        last = min(site_beg+length-1, N-2)
    else:
        last = min(site_beg+length, N-1)

    res = []

    def add (lr, site, res):
        if len(res) > 0 and res[-1][1] == site:
            res[-1] = [lr, site]
        else:
            res.append([lr, site])

    # to right
    for site in range(site_beg, last):
        add (0, site, res)
    for site in range(last, first-1, -1):
        add (1, site, res)
    for site in range(first, site_beg):
        add (0, site, res)
    #print(res)
    return res

# Compute mpo|mps> approximately
# It works better if the initial fit MPS is close to mps0
def fit_apply_MPO_new (mpo, mps, fitmps, numCenter, nsweep=1, maxdim=100000000, cutoff=0., normalize=False, returnLR=False, LR=None, site=-1, psi2_update_length=-1):
    # Check the MPS and the MPO
    assert (len(mpo) == len(mps) == len(fitmps))
    npmps.check_MPO_links (mpo)
    npmps.check_MPS_links (mps)
    npmps.check_MPS_links (fitmps)

    # Define the links to update for a sweep
    # First do a left-to-right and then a right-to-left sweep
    N = len(mps)
    sweeps = get_sweep_sites_new (N, numCenter, site, psi2_update_length)

    fitmps = copy.copy(fitmps)

    # Get the environment tensors
    if LR == None:
        LR = LR_envir_tensors_mpo (N)

    for k in range(nsweep):                                                            # For each sweep
        for lr, p in sweeps:
                # Update the environment tensors
                if numCenter == 2:
                    LR.update_LR (mps, fitmps, mpo, p, p+1)
                elif numCenter == 1:
                    LR.update_LR (mps, fitmps, mpo, p)

                # Define the effective Hamiltonian
                effH = get_eff_H (LR, mpo, p, numCenter)

                #
                #         2                   2      3
                #         |                   |______|
                #    1 ---O--- 3   or   1 ---(________)--- 4
                phi_old = get_eff_psi (mps, p, numCenter)

                # Find the new state for the current bond
                phi = effH.apply (phi_old)
                if normalize:
                    phi = phi / np.linalg.norm(phi)

                # Update tensors
                toRight = (lr==0)
                if numCenter == 2:
                    fitmps[p], fitmps[p+1], err = npmps.truncate_svd2 (phi, rowrank=2, toRight=toRight, maxdim=maxdim, cutoff=cutoff)
                    LR.delete(p)
                    LR.delete(p+1)
                elif numCenter == 1:
                    fitmps[p] = phi
                    fitmps = orthogonalize_MPS_tensor (fitmps, p, toRight, maxdim=maxdim, cutoff=cutoff)
                    LR.delete(p)
                else:
                    raise Exception

    #LR.update_LR (mps, fitmps, mpo, 0, -1)
    #overlap = np.inner (LR[-1].reshape(-1), LR[0].reshape(-1))

    if returnLR:
        return fitmps, LR
    else:
        return fitmps

