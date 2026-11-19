import numpy as np
T = np.load('svd_matrix.npy')
U, S, Vh = np.linalg.svd (T, full_matrices=False)
print(np.linalg.norm(T),np.min(T),np.max(T))
print(S)
