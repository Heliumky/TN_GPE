import numpy as np
import itertools 
from scipy import integrate
import os, sys 
from matplotlib import cm
import matplotlib
import copy
os.environ["OMP_NUM_THREADS"] = "6"
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator
import plotsetting as ps


site = 9
data2 = np.loadtxt(f'u2data_site{site}.txt')
#data8 = np.loadtxt('11.txt')


# In[11]:



CP2 = data2[:, 2]


fig,ax = plt.subplots(figsize=(10, 8))
#X, Y, Z = axes3d.get_test_data(0.05)
x = data2[:,0].reshape(2**site,2**site).T
y = data2[:,1].reshape(2**site,2**site).T
z = data2[:,2].reshape(2**site,2**site).T
# Plot the 3D surface
surfxy = ax.contourf(x, y, z, cmap=cm.coolwarm, antialiased=False)

# Plot projections of the contours for each dimension.  By choosing offsets
# that match the appropriate axes limits, the projected contours will sit on
#the 'walls' of the graph.
ax.set_xlabel(r'$x$', rotation=0)
ax.set_ylabel(r'$y$', rotation=0)
fake2Dline = matplotlib.lines.Line2D([0], [0], linestyle="none", c='y', marker='o')
ax.legend([fake2Dline], [r'$|\psi(x,y)| ^2$'], numpoints=1)
fig.colorbar(surfxy, shrink=0.5, aspect=5)
ax.contourf(x, y, z, cmap=cm.coolwarm, antialiased=False)
#ax.set(xlim=(-8, 8), ylim=(-8, 8), zlim=(-1, 1),xlabel='X', ylabel='Y', zlabel='Z')
plt.savefig("exact_sol"+ '.pdf',transparent=False)
plt.show()
