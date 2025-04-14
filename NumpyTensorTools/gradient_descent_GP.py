import sys, copy
sys.path.append('/home/chiamin/NumpyTensorTools/gradient_descent/')
sys.path.append('/home/chiamin/NumpyTensorTools/')
import npmps
from ncon import ncon
import numpy_dmrg as dmrg
import numpy as np
import time
import gradient_descent as gd
import qtt_tools as qtt
import matplotlib.pyplot as plt
import cost_function_GP_new as costf_new
import MPS_square as mpssqr
import pickle

def left_environment_gradient_dag (L, A):
    W = qtt.MPS_tensor_to_MPO_tensor (A)
    #  ----
    #  |  |---------O--- -1
    #  |  |    1    |
    #  |  |        -2
    #  |  |
    #  |  |--- -3
    #  |  |
    #  |  |--- -4
    #  |  |
    #  |  |--- -5
    #  ----
    tmp = ncon((L, np.conj(A)),((1,-3,-4,-5),(1,-2,-1)))
    #  ----
    #  |  |---------O--- -1
    #  |  |         |
    #  |  |       1 |
    #  |  |---------O--- -2
    #  |  |    2    |
    #  |  |        -3
    #  |  |
    #  |  |--- -4
    #  |  |
    #  |  |--- -5
    #  ----
    tmp = ncon((tmp,np.conj(W)),((-1,1,2,-4,-5),(2,1,-3,-2)))
    #  ----
    #  |  |---------O--- -1
    #  |  |         |
    #  |  |         |
    #  |  |---------O--- -2
    #  |  |         |
    #  |  |       1 |
    #  |  |---------O--- -3
    #  |  |    2    |
    #  |  |        -4
    #  |  |
    #  |  |--- -5
    #  ----
    tmp = ncon((tmp,W),((-1,-2,1,2,-5),(2,1,-4,-3)))
    return tmp

def right_environment_gradient_dag (R, A):
    W = qtt.MPS_tensor_to_MPO_tensor (A)
    #                 ----
    #  -1 ---O--------|  |
    #        |   1    |  |
    #       -2        |  |
    #           -3 ---|  |
    #                 |  |
    #           -4 ---|  |
    #                 |  |
    #           -5 ---|  |
    #                 ----
    tmp = ncon((R,np.conj(A)),((1,-3,-4,-5),(-1,-2,1)))
    #                 ----
    #  -1 ---O--------|  |
    #        |        |  |
    #        | 1      |  |
    #  -2 ---O--------|  |
    #        |   2    |  |
    #       -3        |  |
    #           -4 ---|  |
    #                 |  |
    #           -5 ---|  |
    #                 ----
    tmp = ncon((tmp,np.conj(W)),((-1,1,2,-4,-5),(-2,1,-3,2)))
    #                 ----
    #  -1 ---O--------|  |
    #        |        |  |
    #        |        |  |
    #  -2 ---O--------|  |
    #        |        |  |
    #        | 1      |  |
    #  -3 ---O--------|  |
    #        |   2    |  |
    #       -4        |  |
    #                 |  |
    #           -5 ---|  |
    #                 ----
    tmp = ncon((tmp,W),((-1,-2,1,2,-5),(-3,1,-4,2)))
    return tmp

def left_environment_gradient (L, A):
    W = qtt.MPS_tensor_to_MPO_tensor (A)
    #  ----
    #  |  |--- -1
    #  |  |
    #  |  |--- -2
    #  |  |
    #  |  |--- -3
    #  |  |
    #  |  |        -4
    #  |  |    1    |
    #  |  |---------O--- -5
    #  ----
    tmp = ncon((L, A),((-1,-2,-3,1),(1,-4,-5)))
    #  ----
    #  |  |--- -1
    #  |  |
    #  |  |--- -2
    #  |  |
    #  |  |        -3
    #  |  |    1    |
    #  |  |---------O--- -4
    #  |  |       2 |
    #  |  |         |
    #  |  |---------O--- -5
    #  ----
    tmp = ncon((tmp,W),((-1,-2,1,2,-5),(1,-3,2,-4)))
    #  ----
    #  |  |--- -1
    #  |  |
    #  |  |        -2
    #  |  |    1    |
    #  |  |---------O--- -3
    #  |  |       2 |
    #  |  |         |
    #  |  |---------O--- -4
    #  |  |         |
    #  |  |         |
    #  |  |---------O--- -5
    #  ----
    tmp = ncon((tmp,np.conj(W)),((-1,1,2,-4,-5),(1,-2,2,-3)))
    return tmp

def right_environment_gradient (R, A):
    W = qtt.MPS_tensor_to_MPO_tensor (A)
    #                 ----
    #           -1 ---|  |
    #                 |  |
    #           -2 ---|  |
    #                 |  |
    #           -3 ---|  |
    #                 |  |
    #       -4        |  |
    #        |   1    |  |
    #  -5 ---O--------|  |
    #                 ----
    tmp = ncon((R,A),((-1,-2,-3,1),(-5,-4,1)))
    #                 ----
    #           -1 ---|  |
    #                 |  |
    #           -2 ---|  |
    #                 |  |
    #       -3        |  |
    #        |   1    |  |
    #  -4 ---O--------|  |
    #        | 2      |  |
    #        |        |  |
    #  -5 ---O--------|  |
    #                 ----
    tmp = ncon((tmp,W),((-1,-2,1,2,-5),(-4,-3,2,1)))
    #                 ----
    #           -1 ---|  |
    #                 |  |
    #       -2        |  |
    #        |   1    |  |
    #  -3 ---O--------|  |
    #        | 2      |  |
    #        |        |  |
    #  -4 ---O--------|  |
    #        |        |  |
    #        |        |  |
    #  -5 ---O--------|  |
    #                 ----
    tmp = ncon((tmp,np.conj(W)),((-1,1,2,-4,-5),(-3,-2,2,1)))
    return tmp

class LR_envir_tensors_phi4:
    def __init__ (self, N, dtype=float):
        self.dtype = dtype
        self.centerL = 0
        self.centerR = N-1
        self.LR = dict()
        for i in range(-1,N+1):
            self.LR[i] = None
        self.LR[-1] = np.ones((1,1,1,1),dtype=dtype)
        self.LR[N] = np.ones((1,1,1,1),dtype=dtype)

    def __getitem__(self, i):
        if i >= self.centerL and i <= self.centerR:
            print('environment tensor is not updated')
            print('centerL,centerR,i =',self.centerL,self.centerR,i)
            raise Exception
        return self.LR[i]

    def delete (self, i):
        self.centerL = min(self.centerL, i)
        self.centerR = max(self.centerR, i)

    def get_centers (self):
        return self.centerL, self.centerR

    # self.LR[i-1] and self.LR[i+1] are the left and right environments for the ith site
    # self.LR[i] for i=-1,...,centerL-1 are left environments;
    # self.LR[i] for i=centerR+1,...,N are right environments
    # MPS tensor indices = (l,i,r)
    # MPO tensor indices = (l,ip,i,r)
    # Left or right environment tensor = (up, mid, down)
    def update_LR (self, mps, centerL, centerR=None):
        # Set dtype
        dtype = mps[0].dtype
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
            A = mps[p]
            #  ----
            #  |  |--- -1
            #  |  |
            #  |  |        -2
            #  |  |         |
            #  |  |---------O--- -3
            #  |  |         |
            #  |  |         |
            #  |  |---------O--- -4
            #  |  |         |
            #  |  |         |
            #  |  |---------O--- -5
            #  ----
            tmp = left_environment_gradient (self.LR[p-1], A)
            #  ----     1
            #  |  |---------O--- -1
            #  |  |         | 2
            #  |  |         |
            #  |  |---------O--- -2
            #  |  |         |
            #  |  |         |
            #  |  |---------O--- -3
            #  |  |         |
            #  |  |         |
            #  |  |---------O--- -4
            #  ----
            self.LR[p] = ncon((tmp,np.conj(A)),((1,2,-2,-3,-4),(1,2,-1)))
        # Update the right environments
        for p in range(self.centerR, centerR, -1):
            A = mps[p]
            #                 ----
            #           -1 ---|  |
            #                 |  |
            #       -2        |  |
            #        |        |  |
            #  -3 ---O--------|  |
            #        |        |  |
            #        |        |  |
            #  -4 ---O--------|  |
            #        |        |  |
            #        |        |  |
            #  -5 ---O--------|  |
            #                 ----
            tmp = right_environment_gradient (self.LR[p+1], A)
            #             1   ----
            #  -1 ---O--------|  |
            #        | 2      |  |
            #        |        |  |
            #  -2 ---O--------|  |
            #        |        |  |
            #        |        |  |
            #  -3 ---O--------|  |
            #        |        |  |
            #        |        |  |
            #  -4 ---O--------|  |
            #                 ----
            self.LR[p] = ncon((tmp,np.conj(A)),((1,2,-2,-3,-4),(-1,2,1)))

        self.centerL = centerL
        self.centerR = centerR
'''
def gradient_descent_GP (func, x, step_size, linesearch=False, normalize=True):
    if normalize:
        assert abs(np.linalg.norm (x) - 1) < 1e-12

    grad = func.env0

    global t_setd
    t1 = time.time()

    direction = -grad
    func.set_direction (direction)


    _, slope = func.val_slope (0)
    if abs(slope) < 1e-12:
        print('small slope',slope)
        return x, grad

    if linesearch:
        if step_size < 1e-2:
            step_size = 1e-2
        step_size = gd.line_search (func, step_size=step_size, c1=1e-4, c2=0.9)
    print("x=",x)
    print("step_size=",step_size)
    print("direction=",direction)
    x_next = x + step_size * direction
    if normalize:
        x_next = x_next / np.linalg.norm (x_next)

    return x_next, grad, step_size
'''



def gradient_descent_GP (func, x, step_size, linesearch=False, normalize=True):
    if normalize:
        assert abs(np.linalg.norm (x) - 1) < 1e-12

    grad = func.env0

    global t_setd
    t1 = time.time()

    direction = -grad
    func.set_direction (direction)


    '''_, slope = func.val_slope (0)
    if abs(slope) < 1e-12:
        print('small slope',slope)
        return x, grad'''

    if linesearch:
        if step_size < 1e-4:
            step_size = 1e-4
        step_size, f, df = gd.line_search (func, step_size=step_size, c1=1e-4, c2=0.9)

    x_next = x + step_size * direction
    if normalize:
        x_next = x_next / np.linalg.norm (x_next)

    return x_next, grad, step_size, df, f

# mpo is the non-interacting Hamiltonian
def gradient_descent_GP_MPS (nsweep, mps, mpo, g, step_size, niter=1, maxdim=100000000, cutoff=1e-16, linesearch=False):
    assert len(mps) == len(mpo)
    npmps.check_MPO_links (mpo)
    npmps.check_MPS_links (mps)
    #npmps.check_canonical (mps, 0)

    mps = copy.copy(mps)


    N = len(mps)
    LR = dmrg.LR_envir_tensors_mpo (N)

    #LR4 = LR_envir_tensors_phi4 (N)
    LR4 = mpssqr.MPSSquare (N, cutoff=cutoff)       # cutoff = 0, no truncation
    LR4.update_LR (mps, N-1)                    # put center to the last site


    sites = [range(N), range(N-1,-1,-1)]
    ens = []
    ts = []
    t1 = time.time()
    psi2 = None
    #LR4dim = []
    for s in range(nsweep):
        ti = time.time()
        for lr in [0,1]:
            for p in sites[lr]:

                LR.update_LR (mps, mps, mpo, p)
                LR4.update_LR (mps, p)

                for n in range(niter):
                    func2 = costf_new.cost_function_GP (LR[p-1], mpo[p], LR[p+1], LR4[p-1], mps[p], LR4[p+1], g)

                    A = mps[p]
                    mps[p], grad, step_size, df, f = gradient_descent_GP (func2, mps[p], step_size=step_size, linesearch=linesearch)
                    en = np.inner (grad.conj().flatten(), A.flatten())

                mps = dmrg.orthogonalize_MPS_tensor (mps, p, toRight=(lr==0), maxdim=maxdim, cutoff=cutoff)

        ens.append(en)
        t2 = time.time()
        ts.append(t2-t1)
        print('sweep',s,'en =',en,'alpha =',step_size,'t =',round(t2-ti,2))
        #LR4dim.append(LR4.dim(i))
        #max_dim_psi2 = np.max(LR4dim)
        #psi2_dim.append(max_dim_psi2)
    #np.savetxt('GD2_psi2_dim.txt', psi2_dim, fmt='%d')
    return mps, ens, ts

def gradient_descent_GP_MPS_new(save_iter, nsweep, mps, mpo, g, step_size, niter=1, maxdim=100000000, cutoff=1e-12, maxdim_psi2=100000000, cutoff_psi2=1e-12, linesearch=True, psi2_update_length=-1):
    assert len(mps) == len(mpo)
    npmps.check_MPO_links(mpo)
    npmps.check_MPS_links(mps)

    mps = copy.copy(mps)

    N = len(mps)
    LR = dmrg.LR_envir_tensors_mpo(N)

    sites = [range(N), range(N-1, -1, -1)]
    ens = []
    ts = []
    t1 = time.time()
    psi2 = None
    psi2_dim = []
    df_arr = []
    f_arr = []
    
    func2 = costf_new.cost_function_GP_new(g, mps, psi2, maxdim_psi2=maxdim_psi2, cutoff_psi2=cutoff_psi2, psi2_update_length=psi2_update_length)
    
    for s in range(nsweep):
        ti = time.time()
        for lr in [0, 1]:
            for p in sites[lr]:
                LR.update_LR(mps, mps, mpo, p)

                for n in range(niter):
                    func2.update(LR[p-1], mpo[p], LR[p+1], mps[p], p)

                    A = mps[p]
                    mps[p], grad, step_size, df, f = gradient_descent_GP(func2, mps[p], step_size=step_size, linesearch=linesearch)

                    func2.func4.update_psi(p, mps[p])

                toRight = (lr == 0)
                mps = dmrg.orthogonalize_MPS_tensor(mps, p, toRight=toRight, maxdim=maxdim, cutoff=cutoff)

                if toRight and p != len(mps)-1:
                    func2.func4.update_psi(p, mps[p])
                    func2.func4.update_psi(p+1, mps[p+1])
                elif not toRight and p != 0:
                    func2.func4.update_psi(p, mps[p])
                    func2.func4.update_psi(p-1, mps[p-1])

        tf = time.time()
        ts.append(tf - ti)
        ds2 = func2.func4.psi2_dims()
        en = np.inner(grad.conj().flatten(), A.flatten())
        ens.append(en)
        psi2_dim.append(max(ds2))
        df_arr.append(df)
        f_arr.append(f)

        # 
        if (s + 1) % save_iter == 0:
            with open(f'GD2_mps_step_{s+1}.pkl', 'wb') as f_mps:
                pickle.dump(mps, f_mps)

        # save data
            np.savetxt(f'NewGD_energy_slope_step{s+1}.txt', np.column_stack((f_arr, df_arr)), fmt='%.16f')
            np.savetxt(f'NewGD2_psi2_dim_steps{s+1}.txt', psi2_dim, fmt='%d')
            np.savetxt(f'GD2_CPUTIME_step_{s+1}.txt', np.column_stack((ts, ens)), fmt='%.16f')
            ts.clear()
            ens.clear()
            psi2_dim.clear()
            df_arr.clear()
            f_arr.clear()
    return mps, ens, ts

