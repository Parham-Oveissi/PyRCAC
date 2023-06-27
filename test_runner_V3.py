"""
Created on Mon Jun  5 12:29:24 2023

@author: Parham Oveissi
"""

import PyRCAC_V3
import numpy as np
import matplotlib.pyplot as plt

A = np.array([[0.3, 0.2],[0.1, 0.6]])
B = np.array([[0],[1]])
C = np.array([[1,0]])
D = 0


lu = B.shape[1] 
lx = A.shape[0]
lz = C.shape[0]
ly = C.shape[0]
# ------------------------------------------------------------ Simulation Setup ------------------------------------------------------------------

Step = 500

# ref_sig = np.ones((ly, Step))
ref_sig = np.hstack((np.ones((ly,int(Step/2))),2*np.ones((ly,int(Step/2)))))


my_controller = PyRCAC_V3.RCAC(Nc = 4, Rz = 1, Ru = 0e+1, RegZ = 1, FF = 0, Integrator = 1, FIR = 0, ContType = "dense", R0 = 1e5, Lambda = 0.9995)


x, u, y0, z, theta = my_controller.Simulation_Initializer(lx, lu, ly, lz, Step)


my_filter = PyRCAC_V3.Filter_initializer(lu, Type = "TF", Nu = np.array([[1]]), nf_end = 5, GfRD = 0)


# ------------------------------------------------------------ Simulation ------------------------------------------------------------------------

for kk in range(Step):
    ref = ref_sig[:,kk]
    if kk == 0:
        x[:,kk] = np.random.randn()*np.ones((lx,))
        x[:,kk] = np.array([-7.230404077719068e-01, -7.230404077719068e-01])
        y0[:,kk] = np.matmul(C, x[:,kk].reshape(lx,1)).reshape(ly,)
        z[:,kk] = y0[:,kk] - ref
        u[:,kk], theta[:,kk] = my_controller.RCAC_Control(kk, np.zeros((lu,1)), 0*z[:,kk], 0*z[:,kk], ref, my_filter) 
    else:
        x[:,kk] = (np.matmul(A, x[:,kk-1].reshape(lx,1)) + np.matmul(B, u[:,kk-1].reshape(lu,1))).reshape(lx,)
        y0[:,kk] = np.matmul(C, x[:,kk].reshape(lx,1)).reshape(ly,)
        z[:,kk] = y0[:,kk] - ref
        u[:,kk], theta[:,kk] = my_controller.RCAC_Control(kk, u[:,kk-1], z[:,kk-1], 0*z[:,kk-1], ref, my_filter)
        
    if abs(u[:,kk]) > 1e5:
        break


# ------------------------------------------------------------ Plots ---------------------------------------------------------------------------
plt.plot(x[0,:])
plt.plot(x[1,:])
plt.ylabel('x')
plt.figure()
plt.plot(y0[0,:])
plt.plot(ref_sig[0,:])
plt.ylabel('y')
plt.figure()
plt.plot(u[0,:])
plt.ylabel('u')
