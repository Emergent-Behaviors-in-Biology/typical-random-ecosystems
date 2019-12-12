# -*- coding: utf-8 -*-
"""
Created on Thu 03/31/2019

@author: Wenping Cui
"""
import time
import pandas as pd
import pdb
import matplotlib
from matplotlib import cm
from matplotlib import colors
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import numpy as np
import itertools
from Eco_function.eco_lib import *
from Eco_function.Model_cavity import *
import pdb
import os.path
import pickle
from scipy.integrate import odeint
from multiprocessing import Pool
import multiprocessing 
import argparse
#pdb.set_trace()

parser = argparse.ArgumentParser(description='Process types and dynamics')
parser.add_argument('--A', default='gaussian')
parser.add_argument('--B', default='block')
save_pkl=True


args = parser.parse_args()
A_type = args.A   #'gaussian',‘binomial'
B_type = args.B   #'identity', 'null', 'circulant' and 'block'，‘block-identity’,'foodweb'


start_time = time.time()
Pool_num=10

parameters = {}
parameters['sample_size']=10
parameters['S'] =100 
parameters['M'] =100 
parameters['A_type']=A_type
parameters['B_type']=B_type

parameters['k']=1.0;
parameters['sigma_k']=0.1;

parameters['mu']=1.0;
parameters['epsilon'] =0.1

parameters['g']=1.;
parameters['sigma_g']=0.


parameters['B']=0

parameters['mapping_CR']=True;

parameters['m']=0.1
parameters['sigma_m']=0.01

parameters['ODE_Time']=[0,1000, 20000]


def func_parallel(para,save_pkl=save_pkl):
	index=para[0]
	paras={}
	paras={**para[1],**paras}
	S=paras['S']
	assert paras['A_type'] in ['binomial','gamma', 'gaussian','uniform'], \
	"A type must be 'binomial','gamma' ,'gaussian' or 'uniform'"

	assert paras['B_type'] in ['identity','null', 'circulant','block','block-identity','foodweb'], \
	"B type must be 'identity','null', 'circulant','block','block-identity','foodweb'"

	if paras['B_type']=='identity': #'diag', 'null', 'circulant' and 'block'
		B=np.identity(S)
	elif paras['B_type']=='null':
		B=0
	elif paras['B_type']=='circulant':
		D = [7, 1]  # generalist, specialist
		B=circ(S, D[1])
	elif paras['B_type']=='block':
		B=block(int(S/10), 10)
	elif paras['B_type']=='block-identity':
		B=block(int(S/10), 10)+np.identity(S)
	elif paras['B_type']=='foodweb':
		B=foodweb(int(S/10), 10)
	paras['B']=B

	Model=LV_Cavity_simulation(paras)

	mean_var=Model._simulation(dynamics='CVXOPT')
	if save_pkl:
		filename='CR_LV_'+A_type +'_'+B_type+'_sigc_'+str(round(paras['epsilon'],2)) +'.pkl'
		with open(filename, 'wb') as f:  # Python 3: open(..., 'wb')
			pickle.dump((Model.lams, Model.N_org, Model.phin_list), f)
	if paras['A_type']=='gaussian':
		mu=paras['mu']
		epsilon=paras['epsilon']
	elif paras['A_type']=='uniform':
		mu=paras['epsilon']*paras['S']/2
		epsilon=paras['epsilon']*np.sqrt(paras['S']/12)
	elif paras['A_type']=='binomial':
		mu=paras['epsilon']*paras['S']
		epsilon=np.sqrt(paras['epsilon']*paras['S']*(1-paras['epsilon']))
	if paras['B_type']=='identity':
		phi_R=1.
		paras['r_rev']=(paras['k']+mu*paras['k']*phi_R-paras['m'])/np.sqrt(phi_R*epsilon**2*paras['k']**2+paras['sigma_m']**2)
	elif paras['B_type']=='null':
		phi_R=1.
		paras['r_rev']=(mu*paras['k']*phi_R-paras['m'])/np.sqrt(phi_R*epsilon**2*paras['k']**2+paras['sigma_m']**2)
	paras.pop("B", None)
	paras.pop("ODE_Time", None)

	data= {**paras,**mean_var}
	para_df = pd.DataFrame(data, index=[index])
	return para_df


jobs=[];
index=0
#'identity', 'null', 'circulant' and 'block'，‘block-identity’,'foodweb'
for B_type in ['foodweb']:
	for S in [100]:
		for k in [1.0]:
			for mu in [0]:
				#for epsilon in np.logspace(-6,3., num=200):
				if A_type == 'gaussian':
					ranges=np.logspace(-6.0,3., num=150)
				elif A_type == 'binomial':
					ranges=np.logspace(-6.0,-0.3, num=160)
				elif A_type == 'uniform':
					ranges=np.logspace(-6.0, 2, num=160)
				ranges=[0, 0.01, 0.1, 0.3, 0.5, 1.,5., 10., 20., 50.,100.]
				for epsilon in ranges:
					parameters['k']=k
					parameters['S'] =S
					parameters['M']=S
					parameters['sample_size']=int(10000)
					parameters['B_type']=B_type
					parameters['mu']=mu  
					parameters['epsilon'] =epsilon  
					var=parameters.copy()
					jobs.append([index, var])
					index=index+1
pool = Pool(processes=Pool_num)
results = pool.map(func_parallel, jobs)
pool.close()
pool.join()
results_df = pd.concat(results)
file_name='CM_LV_'+parameters['A_type']+'_'+parameters['B_type']+'.csv'
with open(file_name, 'a') as f:
		results_df.to_csv(f, index=False,encoding='utf-8')

print ('finish time',time.time() - start_time)



