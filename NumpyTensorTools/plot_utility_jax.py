# plot_utility_jax_2d.py
import jax
import jax.numpy as jnp
import numpy as np
import time
from jax import jit
jax.config.update("jax_enable_x64", True)

def convert_mps_to_jax_arrays(mps_list):
    """将MPS列表转换为JAX复数数组元组"""
    return tuple(jnp.array(tensor, dtype=jnp.complex128) for tensor in mps_list)

def convert_mpo_to_jax_arrays(mpo_list):
    """将MPO列表转换为JAX复数数组元组"""
    return tuple(jnp.array(tensor, dtype=jnp.complex128) for tensor in mpo_list)

def convert_binary_to_jax_arrays(bxs, bys):
    """将二进制数组转换为JAX数组（二维版本）"""
    bxs_jax = jnp.array([jnp.array([int(bit) for bit in bx], dtype=jnp.int32) for bx in bxs])
    bys_jax = jnp.array([jnp.array([int(bit) for bit in by], dtype=jnp.int32) for by in bys])
    return bxs_jax, bys_jax

@jit
def compute_mps_element_single(mps_tuple, bstr):
    """计算单个二进制字符串对应的MPS元素（复数版本）"""
    result = jnp.ones((1, 1), dtype=jnp.complex128)
    for i in range(len(mps_tuple)):
        A = mps_tuple[i]
        bi = bstr[i]
        M = A[:, bi, :]
        result = jnp.dot(result, M)
    return result[0, 0]

@jit
def compute_mpo_element_single(mpo_tuple, bstr):
    """计算单个二进制字符串对应的MPO元素（复数版本）"""
    result = jnp.ones((1, 1), dtype=jnp.complex128)
    for i in range(len(mpo_tuple)):
        A = mpo_tuple[i]
        bi = bstr[i]
        M = A[:, bi, bi, :]
        result = jnp.dot(result, M)
    return result[0, 0]

@jit
def compute_batch_mps(mps_tuple, bstr_batch):
    """批量计算MPS元素"""
    return jax.vmap(lambda b: compute_mps_element_single(mps_tuple, b))(bstr_batch)

@jit
def compute_batch_mpo(mpo_tuple, bstr_batch):
    """批量计算MPO元素"""
    return jax.vmap(lambda b: compute_mpo_element_single(mpo_tuple, b))(bstr_batch)

def get_2D_mesh_eles_mps(mps, bxs, bys, batch_size=1024, return_complex=True):
    """2D网格MPS元素计算 - 使用批次并行加速（复数版本）"""
    print("Converting MPS to JAX format...")
    start_time = time.time()
    mps_jax = convert_mps_to_jax_arrays(mps)
    bxs_jax, bys_jax = convert_binary_to_jax_arrays(bxs, bys)
    print(f"Time for conversion: {time.time() - start_time:.4f} seconds")

    nx, ny = len(bxs), len(bys)
    
    # 根据返回类型选择数据类型
    if return_complex:
        result = np.zeros((ny, nx), dtype=np.complex128)
    else:
        result = np.zeros((ny, nx), dtype=np.float64)

    print(f"Computing 2D MPS mesh ({nx} x {ny} = {nx*ny} points)...")
    print(f"Binary string order: bx + by[::-1]")
    print(f"Example: bx='{bxs[0]}', by='{bys[0]}' -> bstr='{bxs[0] + bys[0][::-1]}'")

    # 预编译函数
    print("Pre-compiling JAX function...")
    start_time = time.time()
    # 确认顺序：bx + by[::-1]
    test_bstr = jnp.concatenate([bxs_jax[0], bys_jax[0][::-1]])
    compute_mps_element_single(mps_jax, test_bstr).block_until_ready()
    print(f"Time for pre-compilation: {time.time() - start_time:.4f} seconds")

    total = nx * ny
    if return_complex:
        flat_results = np.zeros(total, dtype=np.complex128)
    else:
        flat_results = np.zeros(total, dtype=np.float64)

    # 批量处理
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_size_now = end - start

        bstr_batch = []
        for idx in range(start, end):
            i = idx % nx  # x 索引
            j = idx // nx  # y 索引
            # 关键：使用 bx + by[::-1] 的顺序
            bx_array = bxs_jax[i]  # 例如: [0, 0, 0, 0]
            by_array = bys_jax[j][::-1]  # 例如: [1, 0, 0, 0] 反转后
            bstr = jnp.concatenate([bx_array, by_array])
            bstr_batch.append(bstr)
        
        bstr_batch = jnp.stack(bstr_batch)

        start_time = time.time()
        batch_result = compute_batch_mps(mps_jax, bstr_batch)
        
        if return_complex:
            flat_results[start:end] = np.array(batch_result)
        else:
            # 返回绝对值
            flat_results[start:end] = np.abs(np.array(batch_result))
        
        if start % (batch_size * 10) == 0 or end == total:
            print(f"Processed {end}/{total} points ({end/total*100:.1f}%) - Batch time: {time.time() - start_time:.4f}s")

    result = flat_results.reshape((ny, nx))
    return result

def get_2D_mesh_eles_mpo(mpo, bxs, bys, batch_size=1024, return_complex=False):
    """2D网格MPO元素计算 - 使用批次并行加速（复数版本）"""
    print("Converting MPO to JAX format...")
    start_time = time.time()
    mpo_jax = convert_mpo_to_jax_arrays(mpo)
    bxs_jax, bys_jax = convert_binary_to_jax_arrays(bxs, bys)
    print(f"Time for conversion: {time.time() - start_time:.4f} seconds")

    nx, ny = len(bxs), len(bys)
    
    # 根据返回类型选择数据类型
    if return_complex:
        result = np.zeros((ny, nx), dtype=np.complex128)
    else:
        result = np.zeros((ny, nx), dtype=np.float64)

    print(f"Computing 2D MPO mesh ({nx} x {ny} = {nx*ny} points)...")
    print(f"Binary string order: bx + by[::-1]")
    print(f"Example: bx='{bxs[0]}', by='{bys[0]}' -> bstr='{bxs[0] + bys[0][::-1]}'")

    # 预编译函数
    print("Pre-compiling JAX function...")
    start_time = time.time()
    # 确认顺序：bx + by[::-1]
    test_bstr = jnp.concatenate([bxs_jax[0], bys_jax[0][::-1]])
    compute_mpo_element_single(mpo_jax, test_bstr).block_until_ready()
    print(f"Time for pre-compilation: {time.time() - start_time:.4f} seconds")

    total = nx * ny
    if return_complex:
        flat_results = np.zeros(total, dtype=np.complex128)
    else:
        flat_results = np.zeros(total, dtype=np.float64)

    # 批量处理
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_size_now = end - start

        bstr_batch = []
        for idx in range(start, end):
            i = idx % nx  # x 索引
            j = idx // nx  # y 索引
            # 关键：使用 bx + by[::-1] 的顺序
            bx_array = bxs_jax[i]  # 例如: [0, 0, 0, 0]
            by_array = bys_jax[j][::-1]  # 例如: [1, 0, 0, 0] 反转后
            bstr = jnp.concatenate([bx_array, by_array])
            bstr_batch.append(bstr)
        
        bstr_batch = jnp.stack(bstr_batch)

        start_time = time.time()
        batch_result = compute_batch_mpo(mpo_jax, bstr_batch)
        
        if return_complex:
            flat_results[start:end] = np.array(batch_result)
        else:
            # 返回绝对值
            flat_results[start:end] = np.abs(np.array(batch_result))
        
        if start % (batch_size * 10) == 0 or end == total:
            print(f"Processed {end}/{total} points ({end/total*100:.1f}%) - Batch time: {time.time() - start_time:.4f}s")

    result = flat_results.reshape((ny, nx))
    return result

# 保持原有的辅助函数
def dec_to_bin(dec, N):
    bstr = ("{:0>"+str(N)+"b}").format(dec)
    return bstr

def bin_to_dec(bstr, rescale=1., shift=0.):
    assert type(bstr) == str
    return int(bstr[::-1], 2) * rescale + shift

def bin_to_dec_list(bstrs, rescale=1., shift=0.):
    return [bin_to_dec(bstr, rescale, shift) for bstr in bstrs]

class BinaryNumbers:
    def __init__(self, N):
        self.N_num = N
        self.N_dec = 2**N

    def __iter__(self):
        self.dec = 0
        return self

    def __next__(self):
        if self.dec < self.N_dec:
            dec = self.dec
            self.dec += 1
            return dec_to_bin(dec, self.N_num)[::-1]
        else:
            raise StopIteration

# 原有的单点计算函数（用于兼容性）
def get_ele_mps(mps, bstr):
    """计算单个二进制字符串对应的MPS元素（兼容原函数）"""
    mps_jax = convert_mps_to_jax_arrays(mps)
    bstr_jax = jnp.array([int(bit) for bit in bstr], dtype=jnp.int32)
    return np.array(compute_mps_element_single(mps_jax, bstr_jax))

def get_ele_mpo(mpo, bstr):
    """计算单个二进制字符串对应的MPO元素（兼容原函数）"""
    mpo_jax = convert_mpo_to_jax_arrays(mpo)
    bstr_jax = jnp.array([int(bit) for bit in bstr], dtype=jnp.int32)
    return np.array(compute_mpo_element_single(mpo_jax, bstr_jax))

def ufunc_2D_eles_mps(mps):
    """返回2D MPS元素的ufunc（复数版本）"""
    def _get_ele(bx, by):
        return get_ele_mps(mps, bx + by[::-1])
    return np.frompyfunc(_get_ele, 2, 1)

def ufunc_2D_eles_mpo(mpo):
    """返回2D MPO元素的ufunc（复数版本）"""
    def _get_ele(bx, by):
        return get_ele_mpo(mpo, bx + by[::-1])
    return np.frompyfunc(_get_ele, 2, 1)

# 绘图函数
try:
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib
    from matplotlib.patches import Patch

    def plot_2D(mps, x1, x2, ax=None, func=None, label=None, batch_size=1024, return_complex=False, **args):
        """2D绘图函数 - 使用JAX加速版本"""
        N = len(mps) // 2
        Ndx = 2**N
        rescale = (x2 - x1) / Ndx
        shift = x1

        bxs = list(BinaryNumbers(N))
        bys = list(BinaryNumbers(N))

        xs = bin_to_dec_list(bxs, rescale, shift)
        ys = bin_to_dec_list(bys, rescale, shift)
        X, Y = np.meshgrid(xs, ys)

        # 使用 JAX 加速的版本
        Z = get_2D_mesh_eles_mps(mps, bxs, bys, batch_size=batch_size, return_complex=return_complex)
        
        if func is not None:
            func = np.vectorize(func)
            Z = func(Z)
            
        if ax is None:
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
            
        if return_complex:
            # 对于复数，默认绘制绝对值
            Z_plot = np.abs(Z)
        else:
            Z_plot = Z
            
        surfxy = ax.plot_surface(X, Y, Z_plot, edgecolor='royalblue', lw=0.5, rstride=8, cstride=8,
                        alpha=0.3)
        ax.contour(X, Y, Z_plot, zdir='z', offset=np.min(Z_plot)-0.1, cmap='coolwarm', alpha=0.7)
        ax.contour(X, Y, Z_plot, zdir='x', offset=np.min(xs), cmap='coolwarm', alpha=0.7)
        ax.contour(X, Y, Z_plot, zdir='y', offset=np.max(ys), cmap='coolwarm', alpha=0.7)
        ax.set(xlim=(np.min(xs), np.max(xs)), ylim=(np.min(ys), np.max(ys)), 
               zlim=(np.min(Z_plot)-0.1, np.max(Z_plot)+0.1),
               xlabel='X', ylabel='Y', zlabel='Z')
        
        fake2Dline = [Patch(facecolor='royalblue', edgecolor='r', alpha=0.3, label=label)]
        ax.legend(handles=fake2Dline)
        fig.colorbar(surfxy, shrink=0.5, aspect=5)
        ax.view_init(15, -150)
        ax.legend(handles=fake2Dline)
        
        if label is not None:
            plt.savefig(f"{label}.pdf", bbox_inches='tight')
            
        return X, Y, Z

except ImportError:
    print("Matplotlib not available, plotting functions disabled")

if __name__ == '__main__':
    # 测试代码 - 验证顺序
    N = 4
    bxs = list(BinaryNumbers(N))
    bys = list(BinaryNumbers(N))
    
    print("=== 验证二进制字符串顺序 ===")
    print("Binary strings for x:", bxs[:3])
    print("Binary strings for y:", bys[:3])
    print("Reversed binary strings for y:", [by[::-1] for by in bys[:3]])
    print("Combined bstr (bx + by[::-1]):")
    for i in range(3):
        for j in range(3):
            bstr = bxs[i] + bys[j][::-1]
            print(f"  bx='{bxs[i]}', by='{bys[j]}' -> bstr='{bstr}'")
