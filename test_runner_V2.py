"""
Created on Mon Jun  5 12:29:24 2023

@author: Parham Oveissi
"""

import PyRCAC_V2
import numpy as np
import matplotlib.pyplot as plt

A = np.array([[0.3, 0.2],[0.1, 0.6]])
B = np.array([[0],[1]])
C = np.array([[1,0]])
# C = np.eye(2)
D = 0

# ------------------------------------------------------------ Simulation ------------------------------------------------------------------

my_system = PyRCAC_V2.System_definition(A, B, C, D)                                                     # First Step: Define your system (A, B, C and D matrices)

# my_system_tf = my_system.get_TF_from_ABCD()


my_controller = PyRCAC_V2.RCAC(Nc = 4, Rz = 1, Ru = 0e+1, RegZ = 1, FF = 0, Integrator = 1, FIR = 0, ContType = "dense", R0 = 1e5, Lambda = 0.9995)          # Second Step: Setup your RCAC control structure



my_filter = PyRCAC_V2.Filter_initializer(my_system)                                                  # Third Step: Initialize your filter


my_simulation = PyRCAC_V2.Dynamic_Simulation(Step = 500)
x, u, y0, z, theta, ref_sig = my_simulation.simulate(my_system, my_filter, my_controller, 'Double_Step_Input')   # Fourth Step: Run the simulation(Step_Input, Double_Step_Input, ...)



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