import os, sys 
#sys.path.append('/home/jerrychen/Desktop/GP_project/xfac/build/python/')
#sys.path.append('/home/chiamin/project/2023/qtt/JhengWei/INSTALL/xfac/build/python/')
sys.path.append('/home/jerrychen/Desktop/My_Work/TN_Numerical/qtt_jerry/xfac_cytnx/build/python/')
import xfacpy
import numpy as np
import cmath
import time
import matplotlib.pyplot as plt
from matplotlib import cm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../NumpyTensorTools')))
import plot_utility as pltut

def fitfun_xy (x, y, seed):
    np.random.seed(seed)
    f = np.sqrt(1/np.pi)*np.exp(-(x**2+y**2)/2)*(x+1j*y)*np.exp(2*np.pi*1j*np.random.rand())
    return f

def inds_to_num (inds, dx, shift):
    bstr = ''
    for i in inds:
        bstr += str(i)
    return pltut.bin_to_dec (bstr, dx, shift)

def fitfun (inds, seed = 15):
    N = len(inds)//2
    Ndx = 2**N
    dx = (x2-x1)/Ndx
    shift = x1

    xinds, yinds = inds[:N], inds[N:]
    x = inds_to_num (xinds, dx, shift)
    y = inds_to_num (reversed(yinds), dx, shift)
    return fitfun_xy (x, y, seed)

def xfac_to_npmps (mpsX, nsite):
  #mps = [None for i in range(nsite)]
  #for it in range(nsite):
  #  mps[it] = mpsX.get(it)
  mps = mpsX.core
  return mps 

def get_init_state (N, x1, x2, maxdim):
    mps = tci (fitfun, 2*N, 2, maxdim=maxdim)
    #mps = npmps.normalize_MPS (mps)
    return mps  

def tci (fun, length, phys_dim, maxdim, tol=1e-30, cplx=True, chk=False):
  fun.__name__
  incD = 2
  maxdim = maxdim + incD
  pm = xfacpy.TensorCI2Param()
  x_pivot1 = np.ones(length//2, dtype=int)
  x_pivot1[-1] = 0
  xy_pivot = np.concatenate((x_pivot1, x_pivot1), axis=None)
  #print(xy_pivot)
  pm.pivot1 = xy_pivot

  pm.reltol = 1e-40
  pm.bondDim = 2 

  if (cplx):
    tci = xfacpy.TensorCI2_complex(fun, [phys_dim]*length, pm) 
  else:
    tci = xfacpy.TensorCI2(fun, [phys_dim]*length, pm) 

  it = 0 
  while (tci.param.bondDim < maxdim):
    t0 = time.time()
    tci.iterate(2,2)
    err0 = tci.pivotError[0]
    err1 = tci.pivotError[-1]
    print("tci: {0:10}| {1:5d} {2:20.3e} {3:20.3e} {4:20.3e} {5:20.2e}".
         format(fun.__name__, tci.param.bondDim, err0, err1, err1/err0, time.time()-t0), flush=True)
    if (err1/err0 < tol):
      break
    tci.param.bondDim = tci.param.bondDim + incD

  if (chk):
    print("tci.trueError = ", tci.trueError())

  return xfac_to_npmps(tci.tt, length) 

if __name__ == '__main__':
    N = 8
    x1,x2 = -6,6
    Ndx = 2**N
    dx = (x2-x1)/Ndx
    print('dx',dx)
    maxdim = 20
    cutoff_mps = 1e-12
    def absSqr (a):
        return abs(a)**2
    absSqr = np.vectorize(absSqr)

    # Initial MPS
    psi = get_init_state (N, x1, x2, maxdim=maxdim)
    pltut.plot_2D_proj(psi,x1,x2,ax=None,func=absSqr,label="tci")

