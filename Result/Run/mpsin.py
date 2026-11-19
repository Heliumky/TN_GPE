import sys, copy, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../NumpyTensorTools')))
import numpy_dmrg as dmrg
import numpy as np
import matplotlib.pyplot as plt
import npmps
import plot_utility as pltut
import hamilt.hamilt_sho as sho
import qtt_tools as qtt
import hamilt.hamilt_angular_momentum as ang
import time
import pickle
import gradient_descent_GP as gdGP
import ncon


#N = 9
input_mps_path = "7_vortex_gd_input.pkl"
with open(input_mps_path, 'rb') as file:
    data = pickle.load(file)
mps = data

for i in range(len(mps)):
    print(f"mps[{i}]",mps[i].shape)

ins_x_pos = 0
vir_bond_x = mps[ins_x_pos].shape[0]
ins_x_tens = np.zeros((vir_bond_x,2,vir_bond_x))
ins_x_tens[0,:,0] = np.array([1,1])
mps.insert(ins_x_pos,ins_x_tens)
#mps.insert(ins_x_pos+2,ins_x_tens)

ins_y_pos = int(len(mps))
#ins_y_pos = len(mps)-1
vir_bond_y = mps[ins_y_pos-1].shape[2]
ins_y_tens = np.zeros((vir_bond_y,2,vir_bond_y))
ins_y_tens[0,:,0] = np.array([1,1])
mps.insert(ins_y_pos,ins_y_tens)

#for i in range(len(mps)):
#    print(f"mps[{i}]",mps[i].shape)

for i in range(len(mps)):
    print(f"mps[{i}]",mps[i].shape)

with open('19_vortex_ext.pkl', 'wb') as f:
    pickle.dump(mps, f)

#for i in range(8,11):
#   print(mps[i])

#extend_mps = npmps.vir_random_MPS (2, 2, vdim=mps[int(N-2)].shape[2], seed=15)


#for i in range(len(extend_mps)):
#    print(f"extend_mps[{i}]",extend_mps[i].shape)
#for i in range(len(extend_mps)):
#	mps.insert(int(N-2+i), extend_mps[i])
#for i in range(len(mps)):
#    print(f"mps[{i}]",mps[i].shape)
    #mps = npmps.normalize_MPS (mps)
    #mps = npmps.normalize_MPS (mps)

