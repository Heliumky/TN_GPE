import copy
import numpy as np
import qtt_tools as qtt
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib
from matplotlib.patches import Patch

def dec_to_bin (dec, N):
    bstr = ("{:0>"+str(N)+"b}").format(dec)
    return bstr
# dec_to_bin(dec=3,N=5) >> 00011


# bstr is a binary string
def bin_to_dec (bstr, rescale=1., shift=0.):
    assert type(bstr) == str
    return int(bstr[::-1],2) * rescale + shift

# bin_to_dec (00011, rescale=1., shift=0.) >>1*2**4+1*2**3 = 16+8=24 invsere the oder bstr[::-1]


def bin_to_dec_list (bstrs, rescale=1., shift=0.):
    return [bin_to_dec (bstr, rescale, shift) for bstr in bstrs]

# save the bin to dec and save the list of decimal.

# An iterator for binary numbers
class BinaryNumbers:
    # N is the number of "site", or the number of binary numbers
    def __init__(self, N):
        self.N_num = N
        self.N_dec = 2**N     # The largest decimal number + 1

    def __iter__(self):
        self.dec = 0
        return self

    def __next__(self):
        if self.dec < self.N_dec:
            dec = self.dec
            self.dec += 1
            return dec_to_bin (dec, self.N_num)[::-1]
        else:
            raise StopIteration

def get_ele_mps (mps, bstr):
    assert type(bstr) == str
    assert len(mps) == len(bstr)
    # Reduce to matrix multiplication
    res = [[1.]]
    for i in range(len(mps)):
        A = mps[i]
        bi = bstr[i]

        M = A[:,int(bi),:]
        res = res @ M
    res = res.reshape(-1)
    assert len(res) == 1
    return res[0]

def get_ele_mpo (mpo, bstr):
    assert type(bstr) == str
    assert len(mpo) == len(bstr)

    # Reduce to matrix multiplication
    res = [[1.]]
    for i in range(len(mpo)):
        A = mpo[i]
        bi = int(bstr[i])

        M = A[:,bi,bi,:]
        res = res @ M
    res = res.reshape(-1)
    assert len(res) == 1
    return res[0]

# Return a ufunc that returns the element of a 2D function in the QTT format
# A ufunc can automatically apply on all the input elements
# bx and by are binary-number strings for x and y
#
# The returned function can be used as follows:
# uf = ufunc_2D_eles (mps)
# res = uf (bxs, bys)
# where bxs and bys can be lists of binary-number strings
#
def ufunc_2D_eles_mps (mps):
    # Define a function that only x and y are input parameters
    def _get_ele (bx, by):
        # bx + by combine the binary strings to a single string
        return get_ele_mps (mps, bx+by[::-1])

    # Return a ufunc
    return np.frompyfunc (_get_ele, 2, 1)

def get_2D_mesh_eles_mps (mps, bxs, bys):
    bX, bY = np.meshgrid (bxs, bys)
    get_2D_ele = ufunc_2D_eles_mps (mps)  # ufunc
    fs = get_2D_ele (bX, bY)
    fs = fs.astype(mps[0].dtype)
    return fs

def get_2D_mesh_eles_mpo (mpo, bxs, bys):

    # Define a function that only x and y are input parameters
    def _get_ele (bx, by):
        # bx + by combine the binary strings to a single string
        return get_ele_mpo (mpo, bx+by[::-1])

    # Make a ufunc
    get_2D_ele = np.frompyfunc (_get_ele, 2, 1)

    bX, bY = np.meshgrid (bxs, bys)
    fs = get_2D_ele (bX, bY)
    fs = fs.astype(np.float64)
    return fs

def plot_1D(mps, x1, x2, ax=None, func=None, label=None, **args):
    N = len(mps)
    Ndx = 2**N
    rescale = (x2 - x1) / Ndx
    shift = x1

    bxs = list(BinaryNumbers(N))
    xs = bin_to_dec_list(bxs, rescale, shift)
    ys = [get_ele_mps(mps, bx) for bx in bxs]

    if func is not None:
        func = np.vectorize(func)
        ys = func(ys)

    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(xs, ys, **args, label=label)
    ax.legend()

    if label is not None:
        plt.savefig(f"{label}.pdf", bbox_inches='tight')

    return ax

def plot_1D_mpo (mpo, x1, x2, ax=None, func=None, **args):
    N = len(mpo)
    Ndx = 2**N
    rescale = (x2-x1)/Ndx
    shift = x1

    bxs = list(BinaryNumbers(N))
    xs = bin_to_dec_list (bxs, rescale, shift)
    ys = [get_ele_mpo (mpo, bx) for bx in bxs]
    if func != None:
        func = np.vectorize(func)
        ys = func(ys)
    if ax == None:
        fig,ax = plt.subplots()
    ax.plot(xs,ys, **args)
    ax.legend()
    return ax

def plot_2D (mps, x1, x2, ax=None, func=None, label=None, **args):
    N = len(mps)//2
    Ndx = 2**N
    rescale = (x2-x1)/Ndx
    shift = x1

    bxs = list(BinaryNumbers(N))
    bys = list(BinaryNumbers(N))

    xs = bin_to_dec_list (bxs, rescale, shift)
    ys = bin_to_dec_list (bys, rescale, shift)
    X, Y = np.meshgrid (xs, ys)

    Z = get_2D_mesh_eles_mps (mps, bxs, bys)
    if func != None:
        func = np.vectorize(func)
        Z = func(Z)
    if ax == None:
        fig,ax = plt.subplots(subplot_kw={"projection": "3d"})
    surfxy = ax.plot_surface(X, Y, Z, edgecolor='royalblue', lw=0.5, rstride=8, cstride=8,
                    alpha=0.3)
    ax.contour(X, Y, Z, zdir='z', offset=-1, cmap='coolwarm', alpha=0.7)
    ax.contour(X, Y, Z, zdir='x', offset=-x1, cmap='coolwarm', alpha=0.7)
    ax.contour(X, Y, Z, zdir='y', offset=x2, cmap='coolwarm', alpha=0.7)
    ax.set(xlim=(-x1, x1), ylim=(-x2, x2), zlim=(-1, 1),
          xlabel='X', ylabel='Y', zlabel='Z')
    #ax.set_xlabel(r'$x$', rotation=0)
    #ax.set_ylabel(r'$y$', rotation=0)
    fake2Dline = [Patch(facecolor='royalblue', edgecolor='r', alpha=0.3, label=label)]
    ax.legend(handles=fake2Dline)
    fig.colorbar(surfxy, shrink=0.5, aspect=5)
    ax.view_init(15, -150)
    ax.legend(handles=fake2Dline)
    if label is not None:
        plt.savefig(f"{label}.pdf", bbox_inches='tight')
    return X, Y, Z

def plot_2D_proj(mps,x1,x2,ax=None,func=None, label=None, **args):
    N = len(mps)//2
    Ndx = 2**N
    rescale = (x2-x1)/Ndx
    shift = x1
    bxs = list(BinaryNumbers(N))
    bys = list(BinaryNumbers(N))

    xs = bin_to_dec_list (bxs, rescale, shift)
    ys = bin_to_dec_list (bys, rescale, shift)
    X, Y = np.meshgrid (xs, ys)

    Z = get_2D_mesh_eles_mps (mps, bxs, bys)
    if func != None:
        func = np.vectorize(func)
        Z = func(Z)
    if ax == None:
        fig,ax = plt.subplots(figsize=(10, 8))
    surfxy = ax.contourf(X, Y, Z, cmap=cm.coolwarm, antialiased=False)
    ax.set_xlabel(r'$x$', rotation=0)
    ax.set_ylabel(r'$y$', rotation=0)
    fake2Dline = matplotlib.lines.Line2D([0], [0], linestyle="none", c='y', marker='o')
    ax.legend([fake2Dline], [r'$|\psi(x,y)| ^2$'], numpoints=1)
    fig.colorbar(surfxy, shrink=0.5, aspect=5)
    ax.contourf(X, Y, Z, cmap=cm.coolwarm, antialiased=False)
    if label is not None:
        plt.savefig(f"{label}.pdf", bbox_inches='tight')
    return X, Y, Z

def plot_2D_mpo (mpo, x1, x2, ax=None, func=None, **args):
    N = len(mpo)//2
    bxs = list(BinaryNumbers(N))
    bys = list(BinaryNumbers(N))

    xs = bin_to_dec_list (bxs)
    ys = bin_to_dec_list (bys)
    X, Y = np.meshgrid (xs, ys)

    Z = get_2D_mesh_eles_mpo (mpo, bxs, bys)
    if func != None:
        func = np.vectorize(func)
        Z = func(Z)
    if ax == None:
        fig,ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface (X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    return X, Y, Z

if __name__ == '__main__':
    print(list(BinaryNumbers(4)))
    exit()

    a = [[1,2],[3,4]]
    b = [[10,20],[30,40]]
    def myf (x,y): return x+y
    myff = np.frompyfunc(myf,2,1)
    print (myff (a,b))
    exit()

    # An example of using BinaryNumbers
    for x in BinaryNumbers(4):
        print(x)
