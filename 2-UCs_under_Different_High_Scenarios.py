# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#
# Purpose: Investigating the impact of different high-flow scenarios
#         on the evaluation metrics (UC1-UC4,and IUC) using Bootstrapping
#
# Created by Tao Huang, June 2022
#
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

import os
import numpy as np
#from scipy.stats import norm
import statsmodels.api as lr
import matplotlib.pyplot as plt

## function to calculate 1-NSE (%) given Obs (D) and Sim(S)
def UC_NSE(D,S):
    obs_mean = np.mean(D)

    numerator = np.sum(np.square(D-S))
    RMSE = np.sqrt(numerator/len(D))
    denominator = np.sum(np.square(D-obs_mean))

    UC_NSE = numerator/denominator*100
        
    return UC_NSE,RMSE


## function to calculate 1-KGE (%) given Obs (D) and Sim(S)
def UC_KGE(D,S):

    r = np.corrcoef(D,S)[0][1]
    ratio_std = np.std(S)/np.std(D)
    ratio_mean = np.mean(S)/np.mean(D)

    UC_KGE = np.sqrt((r-1)**2+(ratio_std-1)**2+(ratio_mean-1)**2)*100
        
    return UC_KGE,r,np.std(D),ratio_std,np.mean(D),ratio_mean


## function to calculate 1-R^2 (%) given Obs (D) and Sim(S)
def UC_R2(D,S):

    linear_model = lr.OLS(D,S,hasconst=True).fit()

    Slope = linear_model.params[0]

    UC_R2 = (1-linear_model.rsquared)*100

    UC_R2_adj = UC_R2 + np.abs(1-Slope)*100
        
    return UC_R2,UC_R2_adj,Slope


## function to calculate N_out/N_total (%) given Obs (D) and 90% bounds of Sim(US and LS)
def UC_dist(D,US,LS):
    N_out = 0
    N_totoal = len(D)

    for i in range(N_totoal):
        if D[i]>US[i] or D[i]<LS[i]:
            N_out+=1

    UC_dist = N_out/N_totoal*100

    width_90 = np.mean(US-LS)
        
    return UC_dist,width_90

## function to calculate IUC given UCs and 90% bounds of Sim(US and LS)
def IUC_alpha(UC_1,UC_2,UC_3,UC_4,US,LS):
    
    width_90 = np.mean(US-LS)

    if width_90<=1:
        alpha = 0.1

    elif width_90>1 and width_90<=3:
        alpha = 0.25

    elif width_90>3 and width_90<=4:
        alpha = 0.5

    elif width_90>4 and width_90<=6:
        alpha = 0.25

    else:
        alpha = 0.1

    IUC = alpha*UC_1+(1-alpha)*(UC_2+UC_3+UC_4)/3
        
    return IUC,alpha


## function to create a folder to store the results if it does not exist

def ResultsFolder(Folder):
    if os.path.exists(Folder) == False:
        os.mkdir(Folder)

Folder1 = './Results_UC/'
ResultsFolder(Folder1)


### Main Program    
# Original observed hydrologic data
D = np.genfromtxt('Obs.csv', delimiter=',',skip_header=True)

# Model predictions (GLUE_top_mean)
S = np.genfromtxt('GLUE_top_mean.csv', delimiter=',',skip_header=True)

# Upper bound predictions
US = S[:,2]

# Lower bound predictions
LS = S[:,3]

# Mean predictions
S = S[:,1]

# number of study models
#K = D.shape[1]
K = 1

# Options of percentiles (same as quantiles*100)
percentiles = np.array([0,10,20,30,40,50,60,70,80,90])

for k in range(K):

    results_UC1_dist = []
    results_UC2_NSE = []
    results_UC3_KGE = []
    #results_UC4_R2 = []
    results_UC4_R2_adj = []
    results_IUC = []
    results_UC_basics = []


    for pt in percentiles:
        ## identify the index where a value is larger than the q-th quantile of the data
        #threshold = np.percentile(D[:,k],pt)
        #index = np.where(D[:,k] >= threshold)
        threshold = np.percentile(D,pt)
        index = np.where(D >= threshold)

        #D_temp = D[:,k][index]
        #S_temp = S[:,k][index]
        D_temp = D[index]
        S_temp = S[index]
        US_temp = US[index]
        LS_temp = LS[index]

        results_UC1_dist.append(UC_dist(D_temp,US_temp,LS_temp)[0])
        results_UC2_NSE.append(UC_NSE(D_temp,S_temp)[0])
        results_UC3_KGE.append(UC_KGE(D_temp,S_temp)[0])
        #results_UC4_R2.append(UC_R2(D_temp,S_temp)[0])
        results_UC4_R2_adj.append(UC_R2(D_temp,S_temp)[1])
        IUC=IUC_alpha(results_UC1_dist[-1],results_UC2_NSE[-1],results_UC3_KGE[-1],results_UC4_R2_adj[-1],US_temp,LS_temp)
        results_IUC.append(IUC[0])
        results_UC_basics.append([IUC[1],UC_dist(D_temp,US_temp,LS_temp)[1],UC_NSE(D_temp,S_temp)[1],UC_KGE(D_temp,S_temp)[2],UC_KGE(D_temp,S_temp)[3],UC_KGE(D_temp,S_temp)[4],UC_KGE(D_temp,S_temp)[5],UC_KGE(D_temp,S_temp)[1],UC_R2(D_temp,S_temp)[2]])     


    # Output the process and final results
    header = "Percentiles_%,UC1,UC2,UC3,UC4,IUC,alpha,90%width_ft,RMSE_ft,sd_obs_ft,ratio_sd,mean_obs_ft,ratio_mean,r,slope"

    UC_final = np.column_stack((percentiles.transpose(),results_UC1_dist,results_UC2_NSE,results_UC3_KGE,results_UC4_R2_adj,results_IUC,results_UC_basics))

    np.savetxt('./Results_UC/UC_final_'+str(k+1)+'.csv', UC_final, fmt='%f',header=header,delimiter = ',',comments='')


    ### plot the obs & sim water stage

    plt.figure(figsize = (16,9))
    lines = ['-k','--r']
    label = ['Observation','GLUE mean']

    #plt.plot(range(1,len(D)+1),D[:,k],lines[0],label=label[0],linewidth=5)
    #plt.plot(range(1,len(D)+1),S[:,k],lines[1],label=label[1],linewidth=5)

    plt.plot(range(1,len(D)+1),D,lines[0],label=label[0],linewidth=5)
    plt.plot(range(1,len(D)+1),S,lines[1],label=label[1],linewidth=5)

    plt.legend(fontsize=20)
    plt.xlabel('Days', fontsize=20)
    plt.ylabel('Water stage (ft)', fontsize=20)
    plt.xticks(range(0,len(D)+1,15),fontsize=20)
    plt.yticks(fontsize=20)

    plt.savefig('./Results_UC/WaterStage_'+str(k+1)+'.jpg',dpi=300)
    plt.close()    


    ### plot the UC values
    plt.figure(figsize = (16,9))

    #markers = ['-ro','--gs','b:^']
    markers = ['-o','--s','-.^',':*']
    labels = ['UC1','UC2','UC3','UC4']

    for i in range(4):
        plt.plot(UC_final[:,0],UC_final[:,i+1],markers[i],label=labels[i],linewidth=5,markersize=15)

    plt.legend(fontsize=20)
    plt.xlabel('≥ Percentiles (%)', fontsize=20)
    plt.ylabel('Uncertainty (%)', fontsize=20)
    plt.xticks(percentiles,fontsize=20)
    plt.yticks(fontsize=20)

    plt.savefig('./Results_UC/UC_final_'+str(k+1)+'.jpg',dpi=300)
    plt.close()



## Estimate the uncertainty of each UC using the Bootstrapping procedure
# number of Bootstrapping sampling
Boot = 1000

# Options of percentiles (same as quantiles*100)
percentiles_new = np.array([0,25,50,75])

for pt in percentiles_new:

    Boot_UC1_dist = []
    Boot_UC2_NSE = []
    Boot_UC3_KGE = []
    Boot_UC4_R2_adj = []
    Boot_IUC = []
    
    ## identify the index where a value is larger than the q-th quantile of the data
    #threshold = np.percentile(D[:,k],pt)
    #index = np.where(D[:,k] >= threshold)
    threshold = np.percentile(D,pt)
    index = np.where(D >= threshold)

    D_temp_0 = D[index]
    S_temp_0 = S[index]
    US_temp_0 = US[index]
    LS_temp_0 = LS[index]

    for b in range(Boot):

        #D_temp = D[:,k][index]
        #S_temp = S[:,k][index]
        Boot_index = np.random.choice(len(D[index]),len(D[index]),replace=True)
        D_temp = D_temp_0[Boot_index]
        S_temp = S_temp_0[Boot_index]
        US_temp = US_temp_0[Boot_index]
        LS_temp = LS_temp_0[Boot_index]

        Boot_UC1_dist.append(UC_dist(D_temp,US_temp,LS_temp)[0])
        Boot_UC2_NSE.append(UC_NSE(D_temp,S_temp)[0])
        Boot_UC3_KGE.append(UC_KGE(D_temp,S_temp)[0])
        Boot_UC4_R2_adj.append(UC_R2(D_temp,S_temp)[1])
        Boot_IUC.append(IUC_alpha(Boot_UC1_dist[-1],Boot_UC2_NSE[-1],Boot_UC3_KGE[-1],Boot_UC4_R2_adj[-1],US_temp,LS_temp)[0])


    # Output the Bootstrapping results
    header = "UC1,UC2,UC3,UC4,IUC"

    Boot_UC = np.column_stack((Boot_UC1_dist,Boot_UC2_NSE,Boot_UC3_KGE,Boot_UC4_R2_adj,Boot_IUC))

    np.savetxt('./Results_UC/Boot_UCs_greater than '+str(pt)+'%.csv', Boot_UC, fmt='%f',header=header,delimiter = ',',comments='')

    
print("Done!")
