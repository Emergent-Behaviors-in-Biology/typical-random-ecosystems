import numpy as np
from scipy.integrate import odeint
import pdb
import time
#pdb.set_trace()
##########################################################################
class LVModel(object):
    def __init__(self, par):
        '''initilize  parameters'''
        self.S = par['S']  # system size S speces
        self.A = par['A']    #  species interactions
        self.k = par['k']    #  carrying capacities
        self.g = par['g']    #  intrinsic growth rates

        self.t0, self.t1, self.Nt = par['ODE_Time'] # ODE simulation time
        self.N_ini = par['N_ini']   # initialize species abundance

        '''Assert initilization error'''
        assert (self.S  == len(self.N_ini)),"initialed species abundance size error"
        assert (self.S  == len(self.k)), "carrying capacities k size error"
        assert (self.S  == len(self.g)), "intrinsic growth rates g size error"
        assert (self.A.ndim == 2), "The interaction matrix A must be a square matrix"
        assert (np.shape(self.A)[0] == np.shape(self.A)[1] and np.shape(self.A)[0]==self.S), "The interaction matrix A must be a square matrix"

        self.N_f = np.zeros(self.S)  # finial species abundance

    def _simulation(self,):
        N_ini=self.N_ini
        T = np.linspace(self.t0, self.t1, num=self.Nt)
        par = [self.A, self.k, self.g] 
        N = odeint(self._LV_dynamics, N_ini, T, args=(par,),mxstep=50000, atol=10 ** -7, hmin=1e-20)
        N[np.where(N < 10 ** -4)] = 0

        self.N_f = N[-1, :]
        self.survive = np.count_nonzero(self.N_f)   

        return T,N

    def _LV_dynamics(self, N, t, par):
        [A , k, g] = par;
        N[np.where(N < 0)] = 0;
        output_vector = N * g * (k - A.dot(N)) 
        return output_vector
   


   






  
