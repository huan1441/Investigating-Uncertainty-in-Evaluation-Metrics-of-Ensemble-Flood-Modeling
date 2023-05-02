# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#
# Purpose: Calculate the means and confidence intervals of UCs obtained based on different Priors
#
# Created by Tao Huang, June 2022
#
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

import os
import numpy as np
#from scipy.stats import ttest_ind


## function to create a folder to store the results if it does not exist

def ResultsFolder(Folder):
    if os.path.exists(Folder) == False:
        os.mkdir(Folder)

Folder1 = './Results_UC/'
ResultsFolder(Folder1)


## Obtain the mean and (1-alpha)*100% confidence interval of UCs   
# Original observed hydrologic data
U = np.genfromtxt('UCs_Uniform.csv', delimiter=',',skip_header=True)

# Model predictions (GLUE_top_mean)
N = np.genfromtxt('UCs_Normal.csv', delimiter=',',skip_header=True)

# number of comparisons
K = U.shape[1]

header_K = np.array(list(range(1,K+1))).T

U_mean = np.full([K,1],np.nan)
Upper_U = np.full([K,1],np.nan)
Lower_U = np.full([K,1],np.nan)

N_mean = np.full([K,1],np.nan)
Upper_N = np.full([K,1],np.nan)
Lower_N = np.full([K,1],np.nan)

# Obtain (1-alpha)*100% confidence interval of UCs by taking the corresponding quantiles
alpha = 0.10

for i in range(K):
    U_mean[i] = np.mean(U[:,i])
    Upper_U[i] = np.quantile(U[:,i],1-alpha/2)
    Lower_U[i] = np.quantile(U[:,i],alpha/2)

    N_mean[i] = np.mean(N[:,i])
    Upper_N[i] = np.quantile(N[:,i],1-alpha/2)
    Lower_N[i] = np.quantile(N[:,i],alpha/2)

U_result = np.column_stack((header_K,U_mean,Upper_U,Lower_U))
N_result = np.column_stack((header_K,N_mean,Upper_N,Lower_N))

np.savetxt('./Results_UC/UC_uniform.csv', U_result, fmt='%f',header="No.,UC_mean,Upper_bound,Lower_bound",delimiter = ',',comments='')
np.savetxt('./Results_UC/UC_normal.csv', N_result, fmt='%f',header="No.,UC_mean,Upper_bound,Lower_bound",delimiter = ',',comments='')

print("Done!")


