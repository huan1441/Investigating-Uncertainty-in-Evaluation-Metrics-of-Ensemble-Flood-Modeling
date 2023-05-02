# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#
# Purpose: Investigating the impact of the measurement errors of observed
#          hydrologic data on the evaluation metrics
#
# Created by Tao Huang, December, 2022
#
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

import os
import numpy as np
#from scipy.stats import norm
#import statsmodels.api as lr
import matplotlib.pyplot as plt

## function to calculate 1-NSE (%) given Obs (D) and Sim(S)
def UC_NSE(D,S):
    obs_mean = np.mean(D)

    numerator = np.sum(np.square(D-S))
    denominator = np.sum(np.square(D-obs_mean))

    UC_NSE = numerator/denominator*100
        
    return UC_NSE


## function to calculate 1-KGE (%) given Obs (D) and Sim(S)
def UC_KGE(D,S):

    r = np.corrcoef(D,S)[0][1]
    ratio_std = np.std(S)/np.std(D)
    ratio_mean = np.mean(S)/np.mean(D)

    UC_KGE = np.sqrt((r-1)**2+(ratio_std-1)**2+(ratio_mean-1)**2)*100
        
    return UC_KGE


## function to calculate 1-R^2 (%) given Obs (D) and Sim(S)
def UC_R2(D,S):

    linear_model = lr.OLS(D,S,hasconst=True).fit()

    Slope = linear_model.params[0]

    UC_R2 = (1-linear_model.rsquared)*100

    UC_R2_adj = UC_R2 + np.abs(1-Slope)*100
        
    return UC_R2,UC_R2_adj


## function to calculate N_out/N_total (%) given Obs (D) and 90% bounds of Sim(US and LS)
def UC_dist(D,US,LS):
    N_out = 0
    N_totoal = len(D)

    for i in range(N_totoal):
        if D[i]>US[i] or D[i]<LS[i]:
            N_out+=1

    UC_dist = N_out/N_totoal*100
        
    return UC_dist

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
# Read data
UC_dist_final = np.genfromtxt('UC_dist.csv', delimiter=',',skip_header=True)

UC_NSE_final = np.genfromtxt('UC_NSE.csv', delimiter=',',skip_header=True)

UC_KGE_final = np.genfromtxt('UC_KGE.csv', delimiter=',',skip_header=True)

UC_R2_adj_final = np.genfromtxt('UC_R2_adj.csv', delimiter=',',skip_header=True)

#IUC_final = np.genfromtxt('IUC.csv', delimiter=',',skip_header=True)


# plot the UCs
labels = ["White noise","Positive bias","Negative bias"]

fig, ((ax0),(ax1),(ax2),(ax3)) = plt.subplots(4, 1, figsize=(3.5, 6))

# plot UC1 (N_out/N_total)
ax0.boxplot(UC_dist_final,labels=labels,showfliers=False,patch_artist=True,
            boxprops = {'color':'black','linewidth':'1.0'})
ax0.set_ylabel("$\it{UC1}$ (%)")
ax0.set_xticklabels(labels,rotation=10)


# plot UC2(1-NSE)
ax1.boxplot(UC_NSE_final,labels=labels,showfliers=False,patch_artist=True,
                boxprops = {'color':'black','linewidth':'1.0'})
#[bx0['boxes'][i].set(facecolor="red") for i in range(3)]
#ax0.set_ylabel("UC1 (1-NSE)%",fontsize=12)
ax1.set_ylabel("$\it{UC2}$ (%)")
ax1.set_xticklabels(labels,rotation=10)

# plot UC3(1-KGE)
ax2.boxplot(UC_KGE_final,labels=labels,showfliers=False,patch_artist=True,
            boxprops = {'color':'black','linewidth':'1.0'})
ax2.set_ylabel("$\it{UC3}$ (%)")
ax2.set_xticklabels(labels,rotation=10)

# plot UC4(1-R^2_adj)
ax3.boxplot(UC_R2_adj_final,labels=labels,showfliers=False,patch_artist=True,
            boxprops = {'color':'black','linewidth':'1.0'})
#ax3.set_ylabel("$\it{UC4 (1-R^{2}_{adj})}$%")
ax3.set_ylabel("$\it{UC4}$ (%)")
ax3.set_xticklabels(labels,rotation=10)

# plot IUC
#ax4.boxplot(IUC_final,labels=labels,showfliers=False,patch_artist=True,
            #boxprops = {'color':'black','linewidth':'1.0'})
#ax4.set_ylabel("$\it{IUC}$ (%)")
#ax4.set_xticklabels(labels,rotation=10)

plt.tight_layout(pad=0.5)
plt.savefig('./Results_UC/UC_obs_error.jpg',dpi=300)
plt.close()
#plt.show()

print("Done!")
