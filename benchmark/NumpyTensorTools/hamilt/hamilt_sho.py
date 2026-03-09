import polynomial as poly
import differential as diff
import npmps
import numpy as np
import matplotlib.pyplot as plt
import copy

def make_H (N, x1, x2, hbar=1, m=1, omega=1):
    Hk = diff.diff2_MPO (N, x1, x2)
    HV = poly.make_xsqr_mpo (N, x1, x2)

    #check_hermitian (Hk)
    #check_hermitian (HV)

    coef_k = -1*hbar**2/m
    coef_V = m*omega**2
    Hk[0] *= coef_k
    HV[0] *= coef_V
    H = npmps.sum_2MPO (Hk, HV)

    #check_hermitian (H)
    #Hbk = copy.copy(H)

    H = npmps.svd_compress_MPO (H, cutoff=1e-12)

    #check_the_same (Hbk, H)
    #check_hermitian (H)

    return H

def plot_GS_exact (x1, x2, ax, hbar=1, m=1, omega=1, **args):
    def gs (x):
        return (m*omega/(np.pi*hbar))**0.25 * np.exp(-m*omega*x*x*0.5/hbar)

    xs = np.linspace(x1,x2,200)
    ys = [gs(i) for i in xs]
    ax.plot(xs,ys, **args)

def plot_1ES_exact (x1, x2, ax, hbar=1, m=1, omega=1, **args):
    def es (x):
        return (m*omega/(np.pi*hbar*4))**0.25 * np.exp(-m*omega*x*x*0.5/hbar) * (2*(m*omega/hbar)**0.5*x)

    xs = np.linspace(x1,x2,200)
    ys = [es(i) for i in xs]
    ax.plot(xs,ys, **args)

def exact_energy (n, hbar=1, omega=1):
    return (n+0.5)*hbar*omega

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
    print(m1-m2)
    assert d < 1e-10
