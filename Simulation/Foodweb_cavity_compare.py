# -*- coding: utf-8 -*-
"""
Created on Thu 03/31/2019

@author: Wenping Cui
"""
import os
from numpy import linalg as LA
import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join
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

start_time = time.time()
A_type='binomial'
dynamics='linear'
save_pkl=1
#Foodweb_type="Plant-Herbivore"
Foodweb_type="Pollination"
mypath=os.path.join(os.getcwd(), "Interactions/"+Foodweb_type)

onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
for item in ["references.csv", "README"]:
	onlyfiles.remove(item)
cols = ['name', 'S','M','phi_N_r', 'phi_N','phi_N_r_bar', 'phi_N_bar', 'N_r', 'N', 'N_r_bar', 'N_bar', 'qN_r' ,'qN', 'qN_r_bar', 'qN_bar','pc_eff1',"pc_eff2"]
data=[]
for f in onlyfiles:
	if f.endswith(".csv"):
		df=pd.read_csv(os.path.join(mypath, f), index_col=0)
		df=df.drop(['Numbers of flowers'], axis=1, errors='ignore')
		C=np.asarray(df).transpose()
		if len(np.where(C.flatten()>1)[0])>0: 
			continue
		S=C.shape[0]
		M=C.shape[1]
		parameters = {}
		parameters['sample_size']=100
		parameters['S'] =S
		parameters['M'] =M
		parameters['K']=1.0;
		parameters['sigma_K']=0.1;
		parameters['mu']=0.0;
		parameters['epsilon'] =0.1
		parameters['sigma_c'] =0.1
		parameters['g']=1.;
		parameters['sigma_g']=0.
		parameters['m']=0.1
		parameters['sigma_m']=0.01
		parameters['t0']=0
		parameters['t1']=500;
		parameters['Nt']=1000;
		parameters['loop_size']=50;
		parameters['sample_size']=200
		Model=Cavity_simulation(parameters)
		Model.B_type='null'
		Model.C_type='binomial'
		Model.C_det=C
		mean_var_real=Model.ode_simulation(Dynamics=dynamics,Simulation_type="CVXOPT",Initial='Manually')
		if save_pkl:
			filename=dynamics+'_Real_'+f+'.pkl'
			with open(filename, 'wb') as file:  # Python 3: open(..., 'wb')
				pickle.dump((Model.col_N, Model.N_org, Model.phin_list,Model.lams), file)
		pc_eff1,pc_eff2=np.mean(C.flatten())*np.sqrt(C.shape[0]),np.mean(C.flatten())*np.sqrt(C.shape[1])
		Model=Cavity_simulation(parameters)
		Model.B_type='null'
		Model.C_type='binomial'
		Model.mu=0
		Model.p_c =np.mean(C.flatten())
		Model.epsilon=np.mean(C.flatten())

		mean_var_the=Model.ode_simulation(Dynamics=dynamics,Simulation_type="CVXOPT")
		if save_pkl:
			filename=dynamics+'_Cavity_'+f+'.pkl'
			with open(filename, 'wb') as file:  # Python 3: open(..., 'wb')
				pickle.dump((Model.col_N, Model.N_org, Model.phin_list,Model.lams), file)
		data.append([f, S, M, mean_var_real['phi_N'], mean_var_the['phi_N'],mean_var_real['phi_N_bar'],mean_var_the['phi_N_bar']\
			, mean_var_real['mean_N'], mean_var_the['mean_N'],mean_var_real['mean_N_bar'],mean_var_the['mean_N_bar'],\
			 mean_var_real['q_N'],  mean_var_the['q_N'],mean_var_real['q_N_bar'],mean_var_the['q_N_bar'],pc_eff1,pc_eff2])
df = pd.DataFrame(data, columns=cols)
file_name=Foodweb_type+"_"+dynamics+'_'+A_type+'.csv'
with open(file_name, 'a') as f:
		df.to_csv(f, index=False,encoding='utf-8')
print ('finish time',time.time() - start_time)
		
