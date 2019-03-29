import time
import pandas as pd
import matplotlib
from matplotlib import cm
from matplotlib import colors
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import numpy as np
import itertools
from Eco_function.eco_lib import *
from Eco_function.eco_plot import *
from Eco_function.eco_func import *
from Eco_function.Model_cavity import *
import pdb
import os.path
import pickle
from scipy.integrate import odeint
from multiprocessing import Pool
import multiprocessing 
import argparse
parser = argparse.ArgumentParser(description='Process types and dynamics')
parser.add_argument('--B', default='identity')
parser.add_argument('--C', default='null')
parser.add_argument('--d', default='quadratic')

args = parser.parse_args()
dynamics=  args.d #'quadratic', 'linear'
B_type = args.B  #'identity', 'null', 'circulant' and 'block'
C_type = args.C  #'gaussian', 'gamma'，‘binomial’binomial


start_time = time.time()
Pool_num=28 # num of thread in simualtions
file_name='Community_'+C_type+'_'+B_type +'_'+dynamics+'_log_v2.csv'

parameters = {}
parameters['sample_size']=10;
parameters['S'] =100;
parameters['M']=100;

parameters['K']=10.0;
parameters['sigma_K']=1.0;

parameters['mu']=1.0; 
parameters['sigma_c']=2.0; 

parameters['m']=1.;
parameters['sigma_m']=0.1;
parameters['loop_size']=50;


parameters['t0']=0;
parameters['t1']=10000;
parameters['Nt']=100000;



def func_parallel(para):
	parameters['sample_size']=para[0];
	parameters['S'] =para[1];
	parameters['M']=para[2];

	parameters['K']=para[3];
	parameters['sigma_K']=para[4];

	parameters['mu']=para[5];
	parameters['sigma_c']=para[6]; 

	parameters['m']=para[7];
	parameters['sigma_m']=para[8];
	parameters['loop_size']=para[9];


	parameters['t0']=para[10];
	parameters['t1']=para[11];
	parameters['Nt']=para[12];
	epsilon=para[13]
	mu=para[14]
	Model=Cavity_simulation(parameters)
	Model.Bnormal=False
	Model.gamma_flag='S/M'
	if B_type=='identity': #'diag', 'null', 'circulant' and 'block'
		Model.B_type='identity'
		Model.mu=mu
		Model.epsilon=epsilon
	elif B_type=='null':
		Model.B_type='null'
		Model.mu=mu
		Model.sigma_c=epsilon
	elif B_type=='circulant':
		Model.B_type='circulant'
		Model.mu=mu
		Model.epsilon=epsilon
	elif B_type=='block':
		Model.B_type='block'
		Model.mu=mu
		Model.epsilon=epsilon
	if C_type=='binomial':
		Model.C_type='binomial'
		Model.p_c=epsilon
		Model.mu = para[14]
		Model.epsilon = para[13]
	elif C_type=='gamma':
		Model.C_type='gamma'
		Model.mu=mu
		Model.epsilon=epsilon
	elif C_type=='gaussian':
		Model.C_type='gaussian'
		Model.mu=mu
		Model.epsilon=epsilon
	elif C_type=='uniform':
		Model.C_type='uniform'
		Model.mu=mu
		Model.epsilon=epsilon
	if dynamics=='linear': #'quadratic' 'linear'
		mean_var=Model.ode_simulation(Dynamics=dynamics,Simulation_type='ODE')
	if dynamics=='quadratic': #'quadratic' 'linear'
		mean_var=Model.ode_simulation(Dynamics=dynamics,Simulation_type='QP')
	mean_var['dynamics']=dynamics
	mean_var['size']=parameters['S']
	mean_var['mu']=mu
	mean_var['epsilon']=epsilon
	index = [0]
	para_df = pd.DataFrame(mean_var, index=index)
	return para_df

jobs=[];
for S in [100]:
	parameters['S'] =S;
	parameters['M'] =S;
	parameters['sample_size']=int(100*100/S);
	#for mu in np.append(0,np.logspace(-3.0, 2., num=10)):
	#for mu in [0, 0.6, 1.0, 3.0, 5.0, 8.0, 10.0]:
	mu=0
	for epsilon in np.logspace(-1.0,3, num=28):  
		#for epsilon in np.linspace(0.0, 2.0, num=201): 
			jobs.append([parameters['sample_size'],parameters['S'],parameters['M'],parameters['K'],parameters['sigma_K'], parameters['mu'], parameters['sigma_c'],parameters['m'],parameters['sigma_m'],parameters['loop_size'],parameters['t0'],parameters['t1'],parameters['Nt']  ,epsilon, mu])
pool = Pool(processes=Pool_num)
results = pool.map(func_parallel, jobs)
pool.close()
pool.join()
results_df = pd.concat(results)
with open(file_name, 'a') as f:
		results_df.to_csv(f, index=False,encoding='utf-8')




