import sys, copy, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../NumpyTensorTools')))
import numpy_dmrg as dmrg
import numpy as np
import matplotlib.pyplot as plt
import npmps
import numpy as np
import pickle

def binary_tuple(n, bits):
    """将整数 n 转换为 bits 位的二进制元组 (q1, q2, ...)"""
    return tuple(int(b) for b in format(n, f'0{bits}b'))

def reversed_binary_tuple(n, bits):
    """将整数 n 转换为反转的二进制元组"""
    normal_bin = format(n, f'0{bits}b')  # 先得到标准二进制字符串
    reversed_bin = normal_bin[::-1]  # 反转字符串
    return tuple(int(b) for b in reversed_bin)  # 转换为元组

def discretize_function(N,x1, seed):
    """构造 2N 维度的 tensor, 存储 f(x, y) = x - y"""
    Ndx = 2**N
    x2 = -x1
    dx = (x2-x1)/Ndx
    shape = (2,) * (2 * N)  # 例如 N=2, shape=(2,2,2,2)
    np.random.seed(seed)
    F = np.zeros(shape, dtype=np.complex128)  # 创建张量

    num_points = 2**N

    x_vals = np.arange(x1, x1 + Ndx * dx, dx) 
    print(x_vals)
    y_vals = x_vals # y 采样点

    for i in range(num_points):
        for j in range(num_points):
            x_bin = reversed_binary_tuple(i, N)  # x 方向标准二进制编码
            y_bin = binary_tuple(j, N)  # y 方向反转二进制编码

            x = x_vals[i]
            y = y_vals[j]

            # 计算 f(x, y) 并存入张量
            F[x_bin][y_bin] = np.sqrt(1/np.pi)*np.exp(-(x**2+y**2)/2)*(x+1j*y)*np.exp(2*np.pi*1j*np.random.rand())
    return F

# 运行示例：N=2 时，张量形状为 (2,2,2,2)


def tensor_to_mps(F, max_bond_dim=20, cutoff=0.0):

    dims = F.shape  # 例如 (2,2,2,2,2,2,2,2)
    n_sites = len(dims)  # 物理维度的数量

    mps = []  # 存储 MPS 张量
    bond_dims = [1] * (n_sites + 1)  # 初始虚拟维度
    current_tensor = F.copy()  # 复制 F 避免修改原始数据

    for i in range(n_sites - 1):
        # 先 reshape 成 (bond_L * physical_dim, 剩余部分)
        D_L = bond_dims[i]
        d = dims[i]  # 每个物理维度都是 2
        current_tensor = current_tensor.reshape(D_L * d, -1)

        # SVD 分解
        U, S, Vh = np.linalg.svd(current_tensor, full_matrices=False)

        # 计算有效 bond 维度
        valid_bond_dim = np.sum(S > cutoff)
        new_bond_dim = min(valid_bond_dim, max_bond_dim)  # 限制 bond 维度

        # 进行截断
        U = U[:, :new_bond_dim]
        S = S[:new_bond_dim]
        Vh = Vh[:new_bond_dim, :]

        # 存储 MPS 张量，形状 (D_L, 2, D_R)
        mps.append(U.reshape(D_L, d, new_bond_dim))

        # 计算新的 bond 维度
        bond_dims[i + 1] = new_bond_dim

        # 更新 current_tensor，吸收 S
        current_tensor = (S[:, np.newaxis] * Vh)

    # 处理最后一个 MPS 张量
    D_L, d, D_R = bond_dims[-2], dims[-1], bond_dims[-1]
    mps.append(current_tensor.reshape(D_L, d, D_R))  # 形状 (D_L, 2, 1)

    return mps

F = discretize_function(N=8,x1=-6, seed=15)
# 转换为 MPS 形式
mps = tensor_to_mps(F, max_bond_dim=200)
print('Initial psi dim:',npmps.MPS_dims(mps))
mps = npmps.svd_compress_MPS (mps, cutoff=1e-18)


with open(f'test_one_v.pkl', 'wb') as f:
    pickle.dump(mps, f)
# 输出 MPS 形状
for i, A in enumerate(mps):
    print(f"MPS[{i}] shape = {A.shape}")

