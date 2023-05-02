# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#
# Purpose: Generate (nxQ) ensemble of HEC-RAS configurations accounting for n & Q
#          and output the top model members with the smallest SSE
#
# Created by Tao Huang, June 2022
#
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

import os
import numpy as np  
from scipy.stats import norm
import copy
#import matplotlib.pyplot as plt

# function to create a folder to store the results if it does not exist

def ResultsFolder(Folder):
    if os.path.exists(Folder) == False:
        os.mkdir(Folder)

### a function that yields a number of samples (ratios) drawn from a prior distribution of the channel Manning's n and upstream streamflow
def GLUE_ensemble(n,Q):

    MN_uniform = np.random.uniform(low=0.8, high=1.2, size=n)
    MN_normal = norm.rvs(loc=1,scale=0.1,size=n)

    Q_uniform = np.random.uniform(low=0.8, high=1.2, size=Q)
    Q_normal = norm.rvs(loc=1,scale=0.1,size=Q)

    np.savetxt('./Results_GLUE/GLUE_n_uniform.csv', MN_uniform, fmt='%f',delimiter = ',')
    np.savetxt('./Results_GLUE/GLUE_n_normal.csv', MN_normal, fmt='%f',delimiter = ',')

    np.savetxt('./Results_GLUE/GLUE_Q_uniform.csv', Q_uniform, fmt='%f',delimiter = ',')
    np.savetxt('./Results_GLUE/GLUE_Q_normal.csv', Q_normal, fmt='%f',delimiter = ',')


### a function that calculates sum of squared errors (inverse of which is the likelihood) given Obs (D) and Sim(S)
### and output the top_number of model members with the smallest SSE
def GLUE_SSE(D,S,top_number):

    # number of models
    K = S.shape[1]

    # a list for the sum of squared errors
    SSE = []

    for i in range(K):
        SSE.append(np.sum(np.square(S[:,i]-D)))

    temp_SSE = copy.deepcopy(SSE)
        
    # find the index list of top number of the smallest SSEs
    index_min = []
    Large_Number = 1e6
    
    for i in range(top_number):
        index_min.append(temp_SSE.index(min(temp_SSE)))
        temp_SSE[temp_SSE.index(min(temp_SSE))]=Large_Number

    index_min.sort()

    GLUE_top = S[:,index_min]
    
    np.savetxt('./Results_GLUE/SSE.csv', SSE, fmt='%f',delimiter = ',')
    np.savetxt('./Results_GLUE/GLUE_top_'+str(top_number)+'.csv', GLUE_top, fmt='%f',delimiter = ',')
        

### Main Program

Folder1 = './Results_GLUE/'
ResultsFolder(Folder1)

## Generateng GLUE ensemble with multiple model configuration
GLUE_ensemble(n=20,Q=20)

# Original observed hydrologic data
D = np.genfromtxt('Obs.csv', delimiter=',',skip_header=True)

# Model predictions of GLUE ensemble based on the uniform or normal prior
S = np.genfromtxt('GLUE_Uniform.csv', delimiter=',',skip_header=True)
#S = np.genfromtxt('GLUE_Normal.csv', delimiter=',',skip_header=True)

top_number = 300

GLUE_SSE(D,S,top_number)


### Output the process and final results
# number of obervations
GLUE_sample = np.genfromtxt('./Results_GLUE/GLUE_top_'+str(top_number)+'.csv', delimiter=',')
T = GLUE_sample.shape[0]

header_T = np.array(list(range(1,T+1))).T

GLUE_mean = np.full([T,1],np.nan)
Upper_GLUE_mean = np.full([T,1],np.nan)
Lower_GLUE_mean = np.full([T,1],np.nan)

# Obtain (1-alpha)*100% confidence interval by taking the corresponding quantiles
alpha = 0.10

for i in range(T):
    GLUE_mean[i] = np.mean(GLUE_sample[i,:])
    Upper_GLUE_mean[i] = np.quantile(GLUE_sample[i,:],1-alpha/2)
    Lower_GLUE_mean[i] = np.quantile(GLUE_sample[i,:],alpha/2)

GLUE_result = np.column_stack((header_T,GLUE_mean,Upper_GLUE_mean,Lower_GLUE_mean))

np.savetxt('./Results_GLUE/GLUE_top_mean.csv', GLUE_result, fmt='%f',header="Time,GLUE_mean,Upper_bound,Lower_bound",delimiter = ',',comments='')

print("DONE!")
