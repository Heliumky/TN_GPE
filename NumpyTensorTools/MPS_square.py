import numpy as np
import qtt_tools as qtt
import npmps
from ncon import ncon

def compress_toRight (ACL, A, p, maxdim=100000000, cutoff=0.):
    W = qtt.MPS_tensor_to_MPO_tensor (A)
    #
    #        ----        -2
    #        |  |    3    |
    #        |  |---------O--- -3
    #  -1 ---|  |         |
    #        |  |    1    | 2
    #        |  |---------O--- -4
    #        ----
    tmp = ncon((ACL,A,W), ((-1,3,1),(1,2,-4),(3,-2,2,-3)))
    #        -2                           -2
    #         |                            |
    #        ----                         ----              ----
    #        |  |                         |  |              |  |
    #        |  |--- -3     ==>           |  |   -3    -1   |  |--- -2
    #  -1 ---|  |                   -1 ---|  |--------------|  |
    #        |  |                         |  |              |  |
    #        |  |--- -4                   |  |              |  |--- -3
    #        ----                         ----              ---
    L, R, err = npmps.truncate_svd2 (tmp, rowrank=2, toRight=True, maxdim=maxdim, cutoff=cutoff)
    return L, R

def compress_toLeft (ACR, A, p, maxdim=100000000, cutoff=0.):
    W = qtt.MPS_tensor_to_MPO_tensor (A)
    #
    #          -3         ----
    #           |     3   |  |
    #     -1 ---O---------|  |
    #           |         |  |--- -4
    #           | 2   1   |  |
    #     -2 ---O---------|  |
    #                     ----
    tmp = ncon((ACR,A,W), ((3,1,-4),(-2,2,1),(-1,-3,2,3)))
    #        -3                                             -2
    #         |                                              |
    #        ----                         ----              ----
    #        |  |                   -1 ---|  |              |  |
    #  -1 ---|  |           ==>           |  |   -3    -1   |  |
    #        |  |--- -4                   |  |--------------|  |--- -3
    #        |  |                         |  |              |  |
    #  -2 ---|  |                   -2 ---|  |              |  |
    #        ----                         ----              ----
    #print(tmp)
    #print(np.isnan(tmp).any(), np.isinf(tmp).any(), tmp.shape, np.max(tmp), np.min(tmp))
    L, R, err = npmps.truncate_svd2 (tmp, rowrank=2, toRight=False, maxdim=maxdim, cutoff=cutoff)
    return L, R

class MPSSquare:
    def __init__ (self, N, maxdim=100000000, cutoff=0., dtype=float):
        self.dtype = dtype
        self.centerL = 0
        self.centerR = N-1
        self.ALR = dict()
        self.AC = dict()
        for i in range(-1,N+1):
            self.ALR[i] = None
            self.AC[i] = None
        self.AC[-1] = np.ones((1,1,1),dtype=dtype)
        self.AC[N] = np.ones((1,1,1),dtype=dtype)
        self.ALR[-1] = np.ones((1,1,1),dtype=dtype)
        self.ALR[N] = np.ones((1,1,1),dtype=dtype)
        self.maxdim = maxdim
        self.cutoff = cutoff

    def __getitem__ (self, i):
        if i >= self.centerL and i <= self.centerR:
            print('environment tensor is not updated')
            print('centerL,centerR,i =',self.centerL,self.centerR,i)
            raise Exception
        return self.AC[i]

    def center (self):
        if self.centerL != self.centerR:
            print('center site is not well defined')
            print('centerL,centerR =',self.centerL,self.centerR)
            raise Exception
        return self.centerL

    def center_to_right (self, A, maxdim=100000000, cutoff=0.):
        ci = self.center()
        W = qtt.MPS_tensor_to_MPO_tensor (A)
        #
        #        ----        -2
        #        |  |    3    |
        #        |  |---------O--- -3
        #  -1 ---|  |         |
        #        |  |    1    | 2
        #        |  |---------O--- -4
        #        ----
        LC = ncon((self.AC[ci-1],A,W), ((-1,3,1),(1,2,-4),(3,-2,2,-3)))
        #
        #                       -2        
        #            ----        |    1   ----
        #            |  |--------O--------|  |
        #            |  |        |        |  |
        #  C = -1 ---|  |        |        |  |--- -3
        #            |  |        |    2   |  |
        #            |  |--------O--------|  |
        #            ----                 ----
        C = ncon((LC,self.AC[ci+1]),((-1,-2,1,2),(1,2,-3)))
        #
        #        -2                                     -2
        #         |                           -2     -1  |
        #   -1 ---O--- -3    ==>    -1 ---|>-------------O--- -3
        #
        # Get the isometry tensor
        V, _, err = npmps.truncate_svd2 (C, rowrank=1, toRight=True, maxdim=maxdim, cutoff=cutoff)
        #
        #        -2
        #         |    1
        #   -1 ---|>------|>--- -3
        #
        self.ALR[ci-1] = ncon((self.ALR[ci-1],V), ((-1,-2,1),(1,-3)))
        #                 -2
        #                  |
        #                 ----
        #                 |  |--- -3
        #             1   |  |
        #  -1 ---<|-------|  |
        #                 |  |
        #                 |  |--- -4
        #                 ----
        LC = ncon((V,LC), ((1,-1),(1,-2,-3,-4)))
        #        -2                           -2
        #         |                            |
        #        ----                         ----              ----
        #        |  |                         |  |              |  |
        #        |  |--- -3     ==>           |  |   -3    -1   |  |--- -2
        #  -1 ---|  |                   -1 ---|  |--------------|  |
        #        |  |                         |  |              |  |
        #        |  |--- -4                   |  |              |  |--- -3
        #        ----                         ----              ---
        self.ALR[ci], self.AC[ci], err = npmps.truncate_svd2 (LC, rowrank=2, toRight=True, maxdim=maxdim, cutoff=cutoff)
        self.centerL = self.centerR = ci+1

    def center_to_left (self, A, maxdim=100000000, cutoff=0.):
        ci = self.center()
        W = qtt.MPS_tensor_to_MPO_tensor (A)
        #
        #          -3         ----
        #           |     3   |  |
        #     -1 ---O---------|  |
        #           |         |  |--- -4
        #           | 2   1   |  |
        #     -2 ---O---------|  |
        #                     ----
        RC = ncon((self.AC[ci+1],A,W), ((3,1,-4),(-2,2,1),(-1,-3,2,3)))
        #
        #                       -2        
        #            ----   1    |        ----
        #            |  |--------O--------|  |
        #            |  |        |        |  |
        #  C = -1 ---|  |        |        |  |--- -3
        #            |  |   2    |        |  |
        #            |  |--------O--------|  |
        #            ----                 ----
        C = ncon((self.AC[ci-1],RC),((-1,1,2),(1,2,-2,-3)))
        #
        #        -2                      -2
        #         |                       |   -3     -1
        #   -1 ---O--- -3    ==>    -1 ---O-------------<|--- -2
        #
        # Get the isometry tensor
        _, V, err = npmps.truncate_svd2 (C, rowrank=2, toRight=False, maxdim=maxdim, cutoff=cutoff)
        #
        #                 -2
        #              1   |
        #   -1 ---<|------<|--- -3
        #
        self.ALR[ci+1] = ncon((V,self.ALR[ci+1]), ((-1,1),(1,-2,-3)))
        #        -3
        #         |
        #        ----
        #  -1 ---|  |
        #        |  |    1
        #        |  |--------|>--- -4
        #        |  |
        #  -2 ---|  |
        #        ----
        RC = ncon((RC,V), ((-1,-2,-3,1),(-4,1)))
        #        -3                                             -2
        #         |                                              |
        #        ----                         ----              ----
        #  -1 ---|  |                   -1 ---|  |              |  |
        #        |  |           ==>           |  |   -3    -1   |  |
        #        |  |--- -4                   |  |--------------|  |--- -3
        #        |  |                         |  |              |  |
        #  -2 ---|  |                   -2 ---|  |              |  |
        #        ----                         ----              ---
        self.AC[ci], self.ALR[ci], err = npmps.truncate_svd2 (RC, rowrank=2, toRight=False, maxdim=maxdim, cutoff=cutoff)
        self.centerL = self.centerR = ci-1
        

    def dim (self, i):
        if i < self.centerL:
            return self.AC[i].shape[0]
        elif i > self.centerR:
            return self.AC[i].shape[-1]
        else:
            return 0

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
            for i in self.AC:
                if self.AC[i] != None:
                    self.AC[i] = self.AC[i].astype(dtype)
                if self.ALR[i] != None:
                    self.ALR[i] = self.ALR[i].astype(dtype)
            self.dtype = dtype

        if centerR == None:
            centerR = centerL
        if centerL > centerR+1:
            print('centerL cannot be larger than centerR+1')
            print('centerL, centerR =',centerL, centerR)
            raise Exception
        # Update the left environments
        for p in range(self.centerL, centerL):
            self.ALR[p], self.AC[p] = compress_toRight (self.AC[p-1], mps[p], p, maxdim=self.maxdim, cutoff=self.cutoff)
        # Update the right environments
        for p in range(self.centerR, centerR, -1):
            self.AC[p], self.ALR[p] = compress_toLeft (self.AC[p+1], mps[p], p, maxdim=self.maxdim, cutoff=self.cutoff)

        self.centerL = centerL
        self.centerR = centerR

    def move_center (self, mps, c, maxdim=100000000, cutoff=0.):
        while self.center() < c:
            self.center_to_right (mps[self.center()], maxdim=maxdim, cutoff=cutoff)
        while self.center() > c:
            self.center_to_left (mps[self.center()], maxdim=maxdim, cutoff=cutoff)
