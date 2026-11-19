import sys, copy
import npmps
from ncon import ncon
import numpy_dmrg as dmrg
import numpy as np
import qtt_tools as qtt

def inner (A1, A2, conj=True):
    if conj:
        A1 = A1.conj()
    return np.inner(A1.flatten(), A2.flatten())

def psi2_contraction (L, R, A1, A2):
    #        ----
    #        |  |--- -2
    #        |  |
    #  -1 ---|  |       -3
    #        |  |   1    |
    #        |  |--------O--- -4
    #        ----
    tmp = ncon((L,A1),((-1,-2,1),(1,-3,-4)))
    #                   -2
    #        ----        |
    #        |  |--------O--- -3
    #        |  |   1    |
    #  -1 ---|  |        | 2
    #        |  |        |
    #        |  |--------O--- -4
    #        ----
    W = qtt.MPS_tensor_to_MPO_tensor (A2)
    tmp = ncon((tmp,W),((-1,1,2,-4),(1,-2,2,-3)))
    #
    #                   -2        
    #        ----        |    1   ----
    #        |  |--------O--------|  |
    #        |  |        |        |  |
    #  -1 ---|  |        |        |  |--- -3
    #        |  |        |    2   |  |
    #        |  |--------O--------|  |
    #        ----                 ----
    tmp = ncon((tmp,R),((-1,-2,1,2),(1,2,-3)))
    return tmp

def psi4_contraction (L, R, A1, A2, A3, A4):
    AA1 = psi2_contraction (L, R, A1, A2)
    AA2 = psi2_contraction (L, R, A2, A3)
    return inner(AA2,AA1)

#            ----                           ----
#            |  |--- -2               -1 ---|  |
#            |  |                           |  |
#  L = -1 ---|  |                 R =       |  |--- -3
#            |  |                           |  |
#            |  |--- -3               -2 ---|  |
#            ----                           ----
#
#
#               -2                         -2                         -2
#                |                          |                          |
#  Apsi2 = -1 ---O--- -3          A = -1 ---O--- -3    ==>   W = -1 ---O--- -4
#                                                                      |
#                                                                     -3
def psi4_environment (L, R, Apsi2, A):
    W = qtt.MPS_tensor_to_MPO_tensor (A)
    #        ----
    #        |  |--- -2
    #        |  |
    #  -1 ---|  |
    #        |  |        -3
    #        |  |    1    |
    #        |  |---------O--- -5
    #        ----         |
    #                    -4
    tmp = ncon((L,W),((-1,-2,1),(1,-3,-4,-5)))
    #        ----
    #        |  |--- -1
    #        |  |
    #   -----|  |
    #   |    |  |        -2
    #   |    |  |         |
    #   |    |  |---------O--- -3
    #   |    ----         |
    #   |                 | 2
    #   |            1    |
    #   ------------------O--- -4
    tmp = ncon((tmp.conj(),Apsi2),((1,-1,-2,2,-3),(1,2,-4)))
    #        ----                   ----
    #        |  |--- -1       -3 ---|  |
    #        |  |                   |  |
    #   -----|  |                   |  |-----
    #   |    |  |        -2         |  |    |
    #   |    |  |         |         |  |    |
    #   |    |  |---------O---------|  |    |
    #   |    ----         |    1    ----    |
    #   |                 |                 |
    #   |                 |    2            |
    #   ------------------O------------------
    tmp = ncon((tmp,R.conj()),((-1,-2,1,2),(-3,1,2)))
    return tmp

# Quartic term
class cost_function_phi4:
    def __init__ (self, L, R, x, normalize=True):
        self.L = L
        self.R = R
        self.x = x
        self.normalize = normalize
        if normalize:
            assert abs(1-np.linalg.norm(x)) < 1e-12
        self.psi2xx0 = psi2_contraction (L, R, x, x)
        self.val0 = inner (self.psi2xx0, self.psi2xx0).real

        # df/dA
        self.env0 = psi4_environment (L, R, self.psi2xx0, x)

    def set_direction (self, d):
        self.d = d
        d_new = d
        if self.normalize:
            xd = inner (self.x, d)               # complex number
            d_new = d - xd * self.x                  # complex vector

        psi2xd0 = psi2_contraction (self.L, self.R, self.x, d_new)
        self.slope0 = 4 * inner (psi2xd0, self.psi2xx0).real

    def val_slope (self, a):
        if a == 0:
            return self.val0, self.slope0

        x = self.x + a * self.d
        if self.normalize:
            norm = np.linalg.norm(x)
            x = x / norm

        phi2xx = psi2_contraction(self.L, self.R, x, x)
        val = inner(phi2xx,phi2xx).real

        d = self.d
        # rescale d
        if self.normalize:
            d = d / norm
            xd = inner (x, d)               # complex number
            d = d - xd * x                  # complex vector

        phi2xd = psi2_contraction(self.L, self.R, x, d)
        slope = 4 * inner(phi2xd, phi2xx).real             # real number
        return val, slope

    def move (self, a):
        self.x += a * self.d

        if self.normalize:
            self.norm0 = np.linalg.norm(self.x)
            self.x = self.x / self.norm0
        self.psi2xx0 = psi2_contraction (self.L, self.R, self.x, self.x)
        self.val0 = inner (self.psi2xx0, self.psi2xx0)
        self.env0 = psi4_environment (self.L, self.R, self.psi2xx0, self.x)
        self.d = None

class cost_function_xHx:
    def __init__ (self, L, M, R, x, normalize=True):
        self.x = x
        self.effH = dmrg.eff_Hamilt_1site (L, M, R)

        self.normalize = normalize
        if normalize:
            assert abs(np.linalg.norm(x) - 1) < 1e-12

        self.Hx = self.effH.apply(x)

    def set_direction (self, d):
        self.d = d
        self.Hd = self.effH.apply(d)

    def move (self, a):
        self.x += a * self.d
        self.Hx += a * self.Hd

        if self.normalize:
            norm = np.linalg.norm(self.x)
            self.x /= norm
            self.Hx /= norm

        self.d = self.Hd = None

    def env (self, a):
        return self.Hx + a * self.Hd

    def val_slope (self, a):
        x = self.x + a * self.d
        env = self.env(a)
        val = inner(env, x).real

        df = 2 * env
        d = self.d

        if self.normalize:
            norm = np.linalg.norm(x)
            val = val / norm**2
            d = d / norm**2

            x = x / norm
            xd = inner (x, d)               # complex number
            d = d - xd * x                  # complex vector
        g = inner(df, d).real             # real number

        return val, g

class cost_function_phi4_new:
    def __init__ (self, psi, maxdim_psi2, cutoff_psi2, psi2=None, normalize=True, psi2_update_length=-1):
        self.psi = copy.copy(psi)
        self.normalize = normalize
        self.maxdim_psi2 = maxdim_psi2
        self.cutoff_psi2 = cutoff_psi2
        self.LR = None
        self.psi2_update_length = psi2_update_length

        # Initialize psi2
        self.psi_op = qtt.MPS_to_MPO (psi)
        if psi2 == None:
            #self.psi2 = npmps.exact_apply_MPO (self.psi_op, self.psi)
            #self.psi2 = npmps.svd_compress_MPS (self.psi2, maxdim=maxdim_psi2, cutoff=cutoff_psi2)

            self.psi2, LR = dmrg.fit_apply_MPO (self.psi_op, self.psi, self.psi, 2, nsweep=2, maxdim=maxdim_psi2, cutoff=1e-14, returnLR=True)
            print("Initial psi2 dim =",npmps.MPS_dims(self.psi2))

            #gg = npmps.normalize_MPS(self.psi2)
            #dd = npmps.normalize_MPS(psi2)
            #print('***',npmps.inner_MPS(gg,dd))
            #exit()
        else:
            self.psi2 = psi2

    def update_psi (self, site, A):
        self.psi[site] = A
        self.psi_op[site] = qtt.MPS_tensor_to_MPO_tensor(A)


    def set_site (self, site):
        self.x = self.psi[site]
        self.site = site
        if self.normalize:
            assert abs(1-np.linalg.norm(self.x)) < 1e-12

        self.env0 = self.__get_gradient__(self.x)
        self.val0 = inner (self.env0, self.x).real

    def psi2_dims (self):
        return npmps.MPS_dims(self.psi2)

    def __get_gradient__ (self, A):
        site = self.site

        psi = copy.copy(self.psi)
        psi[site] = A

        psi_op = copy.copy(self.psi_op)
        psi_op[site] = qtt.MPS_tensor_to_MPO_tensor(A)

        self.psi2, self.LR = dmrg.fit_apply_MPO_new (psi_op, psi, self.psi2, numCenter=2, nsweep=1, maxdim=self.maxdim_psi2, cutoff=self.cutoff_psi2, returnLR=True, LR=self.LR, site=self.site, psi2_update_length=self.psi2_update_length)
        self.LR.update_LR (psi, self.psi2, psi_op, site)

        #ds2 = self.psi2_dims()
        #print(max(ds2),ds2)

        #  ---- 1              4 ----
        #  |  |---------O--------|  |
        #  |  |         | 3      |  |
        #  |  | 2       |      5 |  |
        #  |  |---------O--------|  |
        #  |  |         |        |  |
        #  |  |        -2        |  |
        #  |  |---- -1    -3 ----|  |
        #  ----                  ----
        tmp = ncon((self.LR[site-1], npmps.conj(self.psi2[site]), psi_op[site], self.LR[site+1]), ((1,2,-1),(1,3,4),(2,3,-2,5),(4,5,-3)))

        return tmp.conj()

    def set_direction (self, d):
        self.d = d
        d_new = d
        if self.normalize:
            xd = inner (self.x, d)               # complex number
            d_new = d - xd * self.x                  # complex vector

        self.slope0 = 4 * inner(self.env0, d_new).real

    def val_slope (self, a):
        if a == 0:
            return self.val0, self.slope0

        x = self.x + a * self.d
        if self.normalize:
            norm = np.linalg.norm(x)
            x = x / norm

        env = self.__get_gradient__(x)

        val = inner(env, x).real

        d = self.d
        # rescale d
        if self.normalize:
            d = d / norm
            xd = inner (x, d)               # complex number
            d = d - xd * x                  # complex vector

        slope = 4 * inner(env, d).real             # real number
        return val, slope


class cost_function_GP:
    def __init__ (self, L, M, R, L4, x, R4, g, normalize=True):
        self.g = g
        self.func2 = cost_function_xHx (L, M, R, x, normalize)
        self.func4 = cost_function_phi4 (L4, R4, x, normalize)
        self.env0 = self.func2.Hx + g * self.func4.env0

    def set_direction (self, d):
        self.func2.set_direction(d)
        self.func4.set_direction(d)

    def val_slope (self, a):
        val2, g2 = self.func2.val_slope(a)
        val4, g4 = self.func4.val_slope(a)
        return 2*val2+self.g*val4, 2*g2+self.g*g4

class cost_function_GP_new:
    def __init__ (self, g, psi, psi2, maxdim_psi2, cutoff_psi2, normalize=True, psi2_update_length=-1):
        self.g = g
        self.normalize = normalize
        self.func4 = cost_function_phi4_new (psi, maxdim_psi2, cutoff_psi2, psi2, normalize, psi2_update_length)

    def update (self, L, M, R, x, site):
        self.func2 = cost_function_xHx (L, M, R, x, self.normalize)
        self.func4.set_site (site)
        self.env0 = self.func2.Hx + self.g * self.func4.env0

    def set_direction (self, d):
        self.func2.set_direction(d)
        self.func4.set_direction(d)

    def val_slope (self, a):
        val2, g2 = self.func2.val_slope(a)
        val4, g4 = self.func4.val_slope(a)

        return 2*val2+self.g*val4, 2*g2+self.g*g4
