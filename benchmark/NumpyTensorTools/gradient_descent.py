import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('/home/jerrychen/Desktop/GP_project/modify_CM_code/NumpyTensorTools/')

def gradient_descent (func, x, step_size, linesearch=False, normalize=True):
    if normalize:
        assert abs(np.linalg.norm (x) - 1) < 1e-12

    val, grad = func.val_slope (x)
    #val = func.grad_x (grad, x)

    direction = -grad

    if linesearch:
        step_size = line_search (func, step_size=step_size, c1=1e-4, c2=0.9)

    x_next = x + step_size * direction
    if normalize:
        x_next = x_next / np.linalg.norm (x_next)

    return x_next, grad, val

# ------------------------ line search -------------------------
# Following papers:
# https://link.springer.com/book/10.1007/978-0-387-40065-5, Algo.3.5

def sufficient_decrease_condition (f, f0, df0, c1, a):
    return f <= f0 + c1*a*df0

def curvature_condition (df, df0, c2):
    return abs(df) <= abs(c2*df0)

def strong_Wolfe_condition (f, df, f0, df0, c1, c2, a):
    cond1 = sufficient_decrease_condition (f, f0, df0, c1, a)
    cond2 = curvature_condition (df, df0, c2)
    return cond1 and cond2

def plot_func_vs_step (func, a1, a2, c1=0, c2=0, Na=200):
    f0, df0 = func.val_slope(0)

    fs,dfs,dfs2,fs_Wolfe = [],[],[],[]
    fpre = 0
    a_scan = np.linspace(a1,a2,Na)
    da = a_scan[1] - a_scan[0]
    for a in a_scan:
        f,df = func.val_slope (a)
        fs.append(f)
        dfs.append(df)
        dfs2.append((f-fpre)/da)
        fpre = f
        if c1 != 0 and c2 != 0:
            if strong_Wolfe_condition (f, df, f0, df0, c1, c2, a):
                fs_Wolfe.append(f)
            else:
                fs_Wolfe.append(None)
    dfs2[0] = float('Nan')
    plt.plot(a_scan,fs,marker='.',c='tab:blue',label='f')
    plt.plot(a_scan,dfs,marker='x',c='tab:orange',label='df')
    plt.plot(a_scan-0.5*da,dfs2,marker='+',c='tab:green',label='df2')
    if c2 != 0:
        plt.axhline(c2*df0,ls='--',c='gray')
        plt.axhline(-c2*df0,ls='--',c='gray')
    if len(fs_Wolfe) != 0:
        plt.plot(a_scan,fs_Wolfe,marker='o',c='darkblue',label='Wolfe')
    plt.legend()

def search_interval (stepf, a_lo, a_hi, f_lo, f_hi, df_lo, df_hi, c1=1e-4, c2=0.9):
    def assert_condition (f_lo, f_hi, df_lo, a_hi, a_lo):
        assert f_lo <= f_hi
        assert df_lo*(a_hi-a_lo) < 0
    assert_condition (f_lo, f_hi, df_lo, a_hi, a_lo)

    if True:#__debug__:
        a_lo0, a_hi0 = a_lo, a_hi
        def plot (a_lo, a_hi, f_lo, f_hi, a):
            plot_func_vs_step (stepf, a_lo0, a_hi0, c1=c1, c2=c2)
            ax = plt.gca()
            ax.plot ([a_lo], [f_lo], c='r', marker='v', ms=10, label = "high f(x)")
            ax.plot ([a_hi], [f_hi], c='r', marker='^', ms=10, label = "low f(x)")
            ax.plot ([a], [f], c='r', marker='o', ms=10, label = "f(x)")
            ax.plot ([a], [df], c='r', marker='x', ms=10, label = "gradient f(x)")
            ax.legend()
            return ax

    f0, df0 = stepf.val_slope(0)
    for c in range(10):
        #print('c',c)

        assert_condition (f_lo, f_hi, df_lo, a_hi, a_lo)
        if abs(a_lo-a_hi) < 1e-12:
            print('Cannot find a solution')
            return a_lo

        # bisection search
        a = 0.5 * (a_lo + a_hi)
        f, df = stepf.val_slope (a)

        #if abs(df) < 1e-12:
        #    print('Small slope')
        #    return a


        # sufficient_decrease_condition is not satisfied
        if not sufficient_decrease_condition (f, f0, df0, c1, a) or f >= f_lo:
            # move to the a_lo window
            a_hi = a
            f_hi = f
            df_hi = df
            assert_condition (f_lo, f_hi, df_lo, a_hi, a_lo)

            if __debug__:
                ax = plot (a_lo, a_hi, f_lo, f_hi, a)
                ax.axvspan(a_lo, a_hi, alpha=0.2, color='gray')
                print('11')
                plt.show()
        else:
            # sufficient_decrease_condition is satisfied
            # curvature_condition is satisfied
            if curvature_condition (df, df0, c2):
                if __debug__:
                    ax = plot (a_lo, a_hi, f_lo, f_hi, a)
                    ax.plot ([a], [f], c='r', marker='*', ms=20)
                    print('22')
                    plt.show()
                return a
            # sufficient_decrease_condition is satisfied
            # curvature_condition is not satisfied
            # df * (a_hi - a_lo) >= 0
            elif df * (a_hi - a_lo) >= 0:
                if __debug__:
                    print('33')

                # switch a_lo and a_hi
                a_hi = a_lo
                f_hi = f_lo
                df_hi = df_lo
            a_lo = a
            f_lo = f
            df_lo = df

            if __debug__:
                ax = plot (a_lo, a_hi, f_lo, f_hi, a)
                ax.axvspan(a_lo, a_hi, alpha=0.2, color='gray')
                plt.show()
    return a

def line_search (stepf, step_size=1, c1=1e-4, c2=0.9):
    f0, df0 = stepf.val_slope(0)
    f_pre, df_pre = f0, df0

    a_pre = 0
    a = step_size
    first_iter = True

    for c in range(10):
        assert a_pre < a

        f, df = stepf.val_slope (a)

        if __debug__:
            plot_func_vs_step (stepf, a_pre, a, c1=c1, c2=c2)

        # sufficient_decrease_condition is not satisfied
        if not sufficient_decrease_condition (f, f0, df0, c1, a) or (f >= f_pre and not first_iter):
            if __debug__:
                print('line_search 1:',a_pre, a)
                plt.show()


            # search in the window
            a = search_interval (stepf, a_pre, a, f_pre, f, df_pre, df)
            return a, f, df

        # Both sufficient_decrease_condition and curvature_condition are satisfied
        elif curvature_condition (df, df0, c2):


            if __debug__:
                print('line_search 2')
                plt.show()

            return a, f, df

        # sufficient_decrease_condition is satisfied
        # curvature_condition is not satisfied
        # curvature is positive
        elif df >= 0:
            if __debug__:
                print('line_search 3',a, a_pre)
                plt.show()


            # search in the window
            a = search_interval (stepf, a, a_pre, f, f_pre, df, df_pre)
            return a, f, df

        # sufficient_decrease_condition is satisfied
        # curvature_condition is not satisfied
        # curvature is negative
        # --> the whole window does not satisfy Wolfe condition
        else:
            # enlarge the windown
            step_size *= 2
            a_pre, f_pre, df_pre = a, f, df
            a = a + step_size
            if __debug__:
                print('line_search 4')
                plt.show()

        first_iter = False
    return a, f, df
