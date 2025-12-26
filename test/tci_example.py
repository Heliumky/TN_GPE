import os, sys 
sys.path.append('/home/chiamin/project/2023/qtt/JhengWei/INSTALL/xfac/build/python/')
import xfacpy

import numpy as np
import cmath

import time


def funX(x0):
  f0 = cmath.sqrt(x0)
  return f0

def funQ(inds):
  s0 = sum([b<<i for i, b in enumerate(inds[0::+1])])
  x0 = X0[0] + dX[0]*s0
  return funX(x0)

def xfac_to_npmps (mpsX, nsite):
  mps = [None for i in range(nsite)]
  for it in range(nsite):
    mps[it] = mpsX.get(it)
  return mps 

def tci(fun, length, dim, cplx=True, chk=False):
  fun.__name__
  minD = 2
  incD = 2 
  maxD = 200 
  

  pm = xfacpy.TensorCI2Param()
  pm.pivot1 = [0]*int(length)

  pm.reltol = 1e-20
  pm.bondDim = 2 

  if (cplx):
    tci = xfacpy.TensorCI2_complex(fun, [dim]*length, pm) 
  else:
    tci = xfacpy.TensorCI2(fun, [dim]*length, pm) 

  it = 0 
  while (tci.param.bondDim < maxD):
    t0 = time.time()
    tci.iterate(2,2)
    err0 = tci.pivotError[0]
    err1 = tci.pivotError[-1]
    print("tci: {0:10}| {1:5d} {2:20.3e} {3:20.3e} {4:20.3e} {5:20.2e}".
         format(fun.__name__, tci.param.bondDim, err0, err1, err1/err0, time.time()-t0), flush=True)
    if (err1/err0 < 1e-10):
      break
    tci.param.bondDim = tci.param.bondDim + incD

  if (chk):
    print("tci.trueError = ", tci.trueError())

  return xfac_to_npmps(tci.tt, length) 

if __name__ == '__main__':
  NS = 20

  X0 = np.array([-20.0])
  X1 = np.array([+20.0])
  dX = (X1-X0)/(2**NS)

  tci(funQ, NS, 2, True)
