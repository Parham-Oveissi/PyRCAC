"""
Created on Fri Jun  2 14:27:47 2023

Code Title: Classes
    
@author: Parham Oveissi
"""
import numpy as np
import control as ct
# import matplotlib.pyplot as plt
np.random.seed(0)

class RCAC:
    """
    Nc          Order \n
    Rz:         Performance weight \n
    Ru:         Control weight \n
    RegZ:       Regressoin on performance (if 0 then y goes in the controller elif 1 then z goes in the controller) \n
    FF:         Feedforward structure \n
    Integrator: Integrated performance in the regressor \n
    FIR:        FIR structure \n
    ContType:   Control Type \n
    R0:         R_theta = controller gain weight \n
    Lambda:     Forgetting factor \n
    \n
    Other Attributes: \n
        ltheta \n
        Step \n
        u_h \n
        r_h \n
        yp_h \n
        z_h \n
        intg \n
        PHI \n
        pn \n
        PHI_window \n
        u_window \n
        z_window \n
        P_k \n
        theta_k \n
    """
    def __init__(self, Nc = 4, Rz = 1, Ru = 0e+1, RegZ = 1, FF = 0, Integrator = 1, FIR = 0, ContType = "dense", R0 = 1e5, Lambda = 0.9995):
        
        self.Nc = Nc
        self.Rz = Rz
        self.Ru = Ru
        self.RegZ = RegZ                            # Either 0 or 1: 0->y goes in the controller.
        self.FF = FF                                # Either 0 or 1
        self.Integrator = Integrator                # Either 0 or 1
        self.FIR = FIR                              # Either 0 or 1
        self.ContType = ContType                    # "dense", "PI", "PID"
        self.R0 = R0
        self.Lambda = Lambda
   
    
    
    def CalculateRegSize(self, System, Step):
        self.Step = Step
        if self.Nc == 0:
            self.ltheta = System.lu * System.lz
        else:
             if self.ContType ==  "dense":
                 self.ltheta  = System.lu * (System.lu*self.Nc + System.lz*self.Nc + System.ly*self.Nc*self.FF + System.lz*self.Integrator)
             elif self.ContType == "PID":
                 self.ltheta  = 3
             elif self.ContType == "PI":
                 self.ltheta  = 2
        if self.FIR:
            self.ltheta = System.lu*System.lz*self.Nc
        # print('ltheta is: ', self.ltheta) 
        return self.ltheta
        
        
    def RCAC_Control(self, current_step, u_in, z_in, yp_in, r_in, Filter, System):
        """
        current_step      An integer denoting discrete time k.
        u_in              u(k-1)
        z_in              z(k-1)
        yp_in             y(k-1)
        r_in              r(k-1)
        Filter            Filter structure
        System            System Definition
        """
        if current_step == 0:
            self.Buffer_Initializer(System, intg = 0)
        
        self.u_h[:,1:] = self.u_h[:,0:-1]
        self.z_h[:,1:] = self.z_h[:,0:-1]
        self.r_h[:,1:] = self.r_h[:,0:-1]
        self.yp_h[:,1:] = self.yp_h[:,0:-1]
        
        self.u_h[:,0] = u_in
        self.z_h[:,0] = z_in
        self.r_h[:,0] = r_in
        if self.RegZ == 1:
            self.yp_h[:,0] = z_in
        else:
            self.yp_h[:,0] = yp_in
        
        self.intg = self.intg + self.z_h[:,0]
        
        
        
        self.Build_Regressor(System)
        
        if current_step == 0:
            self.Gf_Optimizer_Initializer(Filter, System)
            # print(self.PHI_window.shape)
            # print(self.u_window.shape)
            # print(self.z_window.shape)
            # print(self.P_k.shape)
            # print(self.theta_k.shape)
            # print(self.PHI.shape)
           
        self.u_window[:,1:Filter.Nf + Filter.nf_end] = self.u_window[:,0:Filter.Nf + Filter.nf_end - 1] 
        self.z_window[:,1:Filter.Nf + Filter.nf_end] = self.z_window[:,0:Filter.Nf + Filter.nf_end - 1] 
        self.PHI_window[1:Filter.Nf + Filter.nf_end, :, :] = self.PHI_window[0:Filter.Nf + Filter.nf_end - 1, :, :] 
        
        self.u_window[:, 0] = u_in
        self.z_window[:, 0] = z_in
        self.PHI_window[0, :, :] = self.PHI
        
       
       
        # print(self.z_window.shape)
        # print(self.PHI_window.shape)
        # print(self.u_h.shape, self.z_h.shape, self.r_h.shape, self.intg, self.yp_h.shape) 
        # print(self.intg, self.z_h[:,0]) 

        # print(self.u_h, self.z_h, self.r_h, self.intg, self.yp_h) 
        # print(Filter.nf_end)
         
        PHI_filt, u_filt, z_filt = self.FilterSignals(Filter)
        
        # print('start')
        # print(PHI_filt.shape)
        # print(u_filt.shape)
        # print(z_filt.shape)
        
        if current_step > max(self.ltheta, Filter.Nf * System.lz * System.lu):
            self.control_on = 1
            self.RLS_update(PHI_filt, z_filt, u_filt)
            
        else:
            self.control_on = 0
            # print(self.control_on)
            
        self.u_out = self.control_on * np.matmul(self.PHI, self.theta_k)
        # print('start')
        # print(self.theta_k.shape)
        # print(self.u_out.shape)
        
        return self.u_out, self.theta_k.flatten()
      
        
    
    
    
    def Buffer_Initializer(self, System, intg = 0):
        self.u_h = np.zeros((System.lu, self.Nc))
        self.z_h = np.zeros((System.lz, self.Nc + 1))
        self.r_h = np.zeros((System.lz, self.Nc + 1))
        self.intg = intg
        if self.RegZ == 1:
            self.yp_h = np.zeros((System.lz, self.Nc + 1))
        else:
            self.yp_h = np.zeros((System.ly, self.Nc + 1))
            
    def Build_Regressor(self, System):
        
        U = self.u_h[:, 0:self.Nc]
        V = self.yp_h[:, 0:self.Nc]
        R = self.r_h[:, 0:self.Nc]
        
        if self.FIR == 1:
            U = np.array([])
            V = self.yp_h[:, 0:max(0,self.Nc)]
        
        U_flatten = U.flatten().reshape(U.size,1)
        V_flatten = V.flatten().reshape(V.size,1)
        R_flatten = R.flatten().reshape(R.size,1)
        if self.ContType == "dense":
            if self.FF == 1:        
                phi = np.vstack((U_flatten,V_flatten,R_flatten))
            elif self.Integrator == 1:
                phi = np.vstack((U_flatten,V_flatten,self.intg))
            else:
                phi = np.vstack((U_flatten,V_flatten))
            
            self.PHI = np.kron(np.eye(System.lu), phi.T)
            
            
        elif self.ContType == "PID":
            self.PHI = np.hstack((self.yp_h[:,0].reshape(self.yp_h[:,0].size,1), self.intg.reshape(self.intg.size,1) , (self.yp_h[:,0]-self.yp_h[:,1]).reshape(self.yp_h[:,0].size,1)))
                
        elif self.ContType == "PI":
            self.PHI = np.hstack((self.yp_h[:,0].reshape(self.yp_h[:,0].size,1), self.intg.reshape(self.intg.size,1)))
        
        # print(self.PHI.shape)
        
        
    def Gf_Optimizer_Initializer(self, Filter, System):
        self.pn = round(self.Step + Filter.Nf + self.Nc)       # According to line 130 in RCAC_v6.m
        self.PHI_window = np.zeros((self.pn, System.lu, self.ltheta))
        self.u_window = np.zeros((System.lu, self.pn))
        self.z_window = np.zeros((System.lz, self.Step))
        
        self.P_k = np.eye(self.ltheta)/self.R0
        self.theta_k = 0e-2 + np.zeros((self.ltheta,1))
        
        
    def FilterSignals(self, Filter):
        Phi_b_rr = Filter.filt_collapse(self.PHI_window[1 + Filter.GfRD : 1 + Filter.Nf + Filter.GfRD, :,:])
        U_b_rr = self.u_window[:, Filter.GfRD : 1 + Filter.Nf + Filter.GfRD - 1]
        U_b_rr_flatten = U_b_rr.flatten().reshape(U_b_rr.size,1)
        
        
        
        PHI_filt = np.matmul(Filter.Nu, Phi_b_rr)
        u_filt   = np.matmul(Filter.Nu, U_b_rr_flatten)
        z_filt   = self.z_window[:, 0].reshape(self.z_window.shape[0],1)
        
        # print('start')
        # print(PHI_filt.shape)
        # print(u_filt.shape)
        # print(self.z_window.shape)
        # print(z_filt.shape)
        
        return PHI_filt, u_filt, z_filt
    
    def RLS_update(self, PHI_filt, z_filt, u_filt):
        if self.Ru == 0:
            self.P_k = self.P_k - np.matmul((np.matmul(PHI_filt, self.P_k)).T, (np.matmul(PHI_filt, self.P_k))) / (1 + np.matmul(np.matmul(PHI_filt, self.P_k), PHI_filt.T))
            self.theta_k = self.theta_k - np.matmul(np.matmul(self.P_k, PHI_filt.T), (np.matmul(PHI_filt, self.theta_k) + z_filt - u_filt))
            
            # print('start')
            # print(self.P_k.shape)
            # print(self.theta_k.shape)
            
        else:
            print('no')
    
   
        
        

class System_definition:
    def __init__(self, A, B, C, D):
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.lu = self.B.shape[1] 
        self.lx = self.A.shape[0]
        self.lz = self.C.shape[0]
        self.ly = self.C.shape[0]  
    def get_TF_from_ABCD(self):
        for i in range(self.lu):
            for j in range(self.ly):
                cont_tf = ct.ss2tf(self.A, self.B[:,i], self.C[j,:], 0)
                num, den = ct.tfdata(cont_tf)
                return ct.tf(num,den,1)


        


class Filter_initializer:
    def __init__(self, System, Type = "TF", Nu = np.array([[1]]), nf_end = 5, GfRD = 0):
        self.Type = Type                            # Either "TF" or "SS" 
        self.Nu = Nu
        self.Nf = int(self.Nu.shape[1]/System.lu)      # Accorning to lines 47 and 80 in RCAC_linear_MP_runner Matlab File
        self.nf_end = nf_end
        self.GfRD = GfRD
    def filt_collapse(self, tensor):
        k = tensor.shape[0]
        m = tensor.shape[1]
        n = tensor.shape[2]
        tensor_collapsed = np.zeros((m*k, n))
        for ii in range(k):
            tensor_collapsed[m*ii:m*(ii+1), :] = tensor[ii, :, :]
        return tensor_collapsed     
            

class Dynamic_Simulation:
    def __init__(self, Step = 500):
        self.Step = Step
        
    
    def simulate(self, System, Filter, Controller, Refrence_Signal = 'Step_Input'):
        """1st Step: Define the refrence signal"""
        if Refrence_Signal == 'Step_Input':
            ref_sig = np.ones((System.ly, self.Step))
        elif Refrence_Signal == 'Double_Step_Input':
            ref_sig = np.hstack((np.ones((System.ly,250)),2*np.ones((System.ly,250))))
        else:
            ref_sig = Refrence_Signal
            
        """2nd Step: Calculate ltheta"""
        ltheta = Controller.CalculateRegSize(System, self.Step)
        
        """3rd Step: Initialize variables for the simulation"""
        x, u, y0, z, theta = self.Simulation_Initializer(System , ltheta)
       
        """4th Step: Iterate the simulation"""
        for kk in range(self.Step):
            ref = ref_sig[:,kk]
            if kk == 0:
                x[:,kk] = np.random.randn()*np.ones((System.lx,))
                x[:,kk] = np.array([-7.230404077719068e-01, -7.230404077719068e-01])
                y0[:,kk] = np.matmul(System.C, x[:,kk].reshape(System.lx,1)).reshape(System.ly,)
                z[:,kk] = y0[:,kk] - ref
                u[:,kk], theta[:,kk] = Controller.RCAC_Control(kk, np.zeros((System.lu,1)), 0*z[:,kk], 0*z[:,kk], ref, Filter, System) 
            else:
                x[:,kk] = (np.matmul(System.A, x[:,kk-1].reshape(System.lx,1)) + np.matmul(System.B, u[:,kk-1].reshape(System.lu,1))).reshape(System.lx,)
                y0[:,kk] = np.matmul(System.C, x[:,kk].reshape(System.lx,1)).reshape(System.ly,)
                z[:,kk] = y0[:,kk] - ref
                u[:,kk], theta[:,kk] = Controller.RCAC_Control(kk, u[:,kk-1], z[:,kk-1], 0*z[:,kk-1], ref, Filter, System)
                
            if abs(u[:,kk]) > 1e5:
                break
            
        return x, u, y0, z, theta, ref_sig
    
    
    def Simulation_Initializer(self, System, ltheta):
        x = np.zeros((System.lx, self.Step))
        u = np.zeros((System.lu, self.Step))
        y0 = np.zeros((System.ly, self.Step)) 
        z = np.zeros((System.lz, self.Step))
        theta = np.zeros((ltheta, self.Step))
        return x, u, y0, z, theta
        
