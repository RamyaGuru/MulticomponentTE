"""
Created on Wed Nov 10 23:14:09 2021

@author: ramyagurunathan

Quaternary Chalcogenide Test Script
"""

import numpy as np
import sys
sys.path.append('../')
import klemens_thermal as kt
import matplotlib.pyplot as plt
import math


'''
endmember properties
'''

PbTe = {
      'vs' : 1850,
      'atmMass' :[207.2, 127.60],
      'atmRadius': [0.775, 0.97],
      'atmV': 35.379e-30,
      'natoms': 2,
      'stoich': [1,1],
      'k0' : 1.73,
      }

PbSe = {
      'vs' : 1960,
      'atmMass' :[207.2, 78.971],
      'atmRadius': [0.775, 0.5],
      'atmV': 30.127e-30,
      'natoms': 2,
      'stoich': [1,1],
      'k0' : 1.43, 
      }

SnTe = {
      'vs' : 1800,
      'atmMass' :[118.71, 127.60], 
      'atmRadius': [.69, 0.97],
      'atmV': 33.0335e-30,
      'natoms': 2,
      'stoich': [1,1],
      'k0' : 2.02,
      }


SnSe = {
      'vs' : 1420,
      'atmMass' :[118.71, 78.971], 
      'atmRadius': [.69, 0.5],
      'atmV': 28.13e-30,
      'natoms': 2,
      'stoich': [1,1],
      'k0' : 0.99,
      }

'''
Get binaries and fit epsilon parameters to each one 
'''
exp_kL = np.loadtxt('../datafiles/quat_tcond_chalc.csv',delimiter = ',')

PbTe_PbSe = np.array([np.linspace(1e-10,1 - 1e-10, 11), exp_kL[0,:]])

PbTe_SnTe = np.array([np.linspace(1e-10,1 - 1e-10, 11),exp_kL[:,0]])

SnTe_SnSe = np.array([np.linspace(1e-10,1 - 1e-10, 11), exp_kL[-1, :]])

PbSe_SnSe = np.array([np.linspace(1e-10,1 - 1e-10, 11), exp_kL[:,-1]])

PbTe_SnSe = np.array([np.linspace(1e-10,1 - 1e-10, 11), [exp_kL[j,j] for j in range(len(exp_kL))]])

SnTe_PbSe = np.array([np.linspace(1e-10,1 - 1e-10, 11), [exp_kL[j,len(exp_kL)-1-j] for j in range(len(exp_kL)-1, -1, -1)]])

bin_list = [PbSe_SnSe, SnTe_SnSe, PbTe_SnSe, SnTe_PbSe, PbTe_PbSe, PbTe_SnTe]
host_list = [PbSe, SnTe, PbTe, SnTe, PbTe, PbTe]
defect_list = [SnSe, SnSe, SnSe, PbSe, PbSe, SnTe]

eps_list = []
for bin_data, host, defect in zip(bin_list, host_list, defect_list):
    eps_list.append(kt.fit_eps_poly(kt.gammaM_vac, kt.gammaV, bin_data, host, defect, 3))
    
    
'''
Plot Experimental Thermal conductivity Heatmap
'''

for r in range(4): #Exclude lower right because samples are in the Pnma phase
    exp_kL[7+r][-(1+r):] = math.nan
plt.matshow(exp_kL, cmap = plt.cm.get_cmap('rainbow'), vmin = 0.55, vmax = 2) 
plt.title('Experimental', size = 14)  
ax = plt.gca()
plt.tick_params(left=False,
                top=False, bottom = False)
ax.set_xticks([])
ax.set_yticks([])
ax.text(0, -0.05, 'SnTe', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
ax.text(1, -0.05, 'SnSe', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
ax.text(0, 1.05, 'PbTe', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
ax.text(1, 1.05, 'PbSe', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
cbar = plt.colorbar(shrink = 0.8,  pad = 0.01)    
cbar.set_label(r'Lattice Thermal Conductivity (W/m/K)', size = 12)


'''
Plot Model Themral Conductivity
'''

quat_kL = kt.run_kL_gamma_reciprocal_quat(kt.kL_tot_muggianu, kt.gammaM_vac, kt.gammaV, eps_list, 10, [SnSe, PbSe, SnTe, PbTe])

for r in range(4):
    quat_kL[7+r][-(1+r):] = math.nan
plt.matshow(quat_kL, cmap = plt.cm.get_cmap('rainbow'), vmin = 0.6, vmax = 2)
plt.title('Alloy Model', size = 24)
ax = plt.gca()
plt.tick_params(left=False,
                top=False, bottom = False)
ax.set_xticks([])
ax.set_yticks([])
ax.text(0, -0.05, 'SnTe', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
ax.text(1, -0.05, 'SnSe', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
ax.text(0, 1.05, 'PbTe', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
ax.text(1, 1.05, 'PbSe', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
cbar = plt.colorbar(shrink = 0.8,  pad = 0.01)    
cbar.set_label(r'Lattice Thermal Conductivity (W/m/K)', size = 14) 



