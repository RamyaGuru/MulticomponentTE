#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 09:31:33 2020

@author: ramyagurunathan

Thermal Conductivity: General Methods for Multicopmponent Thermal Conductivity
"""
from math import pi
import numpy as np
from scipy.optimize import curve_fit
from itertools import combinations 

'''
Constants
'''
kB = 1.38E-23
hbar = 1.054E-34
Na = 6.02E23




'''
Methods for Property Conversions
'''

def debyeT(atmV, vs):
    return (hbar /kB) * (6*pi**2 / atmV)**(1/3) * vs

def vs_from_debyeT(atmV, debyeT):
    return (kB / hbar) * (6*pi**2 / atmV)**(-1/3) * debyeT

def average_phase_velocity(vp):
    avg_vp = (1/3) * ( vp[0]**(-3) + vp[1]**(-3) + vp[2]**(-3))**(-1/3)
    return avg_vp


'''
Binary PDScattering Model
'''


def gammaM_vac(stoich, mass, subst, c):
    """
    Function: Mass difference equation from the current paper. With Virial Theorem
    handling of vacancies.
    """
    natoms = sum(stoich)
    delM2 = 0
    denom = 0
    for n in range(len(mass)):
        msite = subst[n]*c + mass[n]*(1-c)
        #delM2 = delM2 + stoich[n]*(c*(subst[n] - msite)**2 + (1-c)*(mass[n] - msite)**2)
        if subst[n] == 0 or mass[n] == 0:
            delM2 = delM2 + stoich[n]*c*(1-c)*(3*(subst[n] - mass[n]))**2
            denom = denom + stoich[n]*msite
        else:
            delM2 = delM2 + stoich[n]*c*(1-c)*(subst[n] - mass[n])**2
            denom = denom + stoich[n]*msite               
    gamma = (delM2/natoms)/((denom/natoms)**2)
    return gamma


def gammaV(stoich, rad, subst, c):
    '''
    Function for radius difference scattering
    '''
    natoms = sum(stoich)
    delR2 = 0
    denom = 0
    for n in range(len(rad)):
        rsite = subst[n]*c + rad[n]*(1-c)        
        delR2 = delR2 + stoich[n]*c*(1-c)*(subst[n] - rad[n])**2
        denom = denom + stoich[n]*rsite               
    gamma = (delR2/natoms)/((denom/natoms)**2)
    return gamma    

'''
Binary Thermal Conductivity
'''
def kL_from_gamma(gamma, propA, propB, c):
    '''
    Returns lattice thermal conductivity given a value for the scattering
    strength, Gamma.
    '''
    atmV = (1-c) * propA['atmV'] + c * propB['atmV']
    vs = (1-c) * propA['vs'] + c * propB['vs']
    k0 = (1-c) * propA['k0'] + c * propB['k0']
    prefix = (6**(1/3)/2)*(pi**(5/3)/kB)*(atmV**(2/3)/vs)
    u = (prefix * gamma * k0)**(1/2)
    kL = k0*np.arctan(u)/u
    return kL

def kL_tot(c, eps, Mfunc, Rfunc, propA, propB, return_gamma = False):
    gamma = Mfunc(propA['stoich'], propA['atmMass'], propB['atmMass'], c) +\
    eps * Rfunc(propA['stoich'], propA['atmRadius'], propB['atmRadius'], c)
    kL = kL_from_gamma(gamma, propA, propB, c)
    if return_gamma:
        return kL, gamma
    else:
        return kL


def fit_eps_kL(Mfunc, Rfunc, data, propA, propB):
    data = data[data[:,0].argsort()]
    eps, cov = curve_fit(lambda c, eps:\
                         kL_tot(c, eps, Mfunc, Rfunc, propA, propB), data[0,:],\
                                data[1,:], bounds = (0,np.inf))
    kL = np.zeros(100)
    i = 0
    for c in np.linspace(data[0,0], data[0, -1], 100):
        kL[i] = kL_tot(c, eps, Mfunc, Rfunc, propA, propB)
        i = i+1
    j = 0
    kL_full = np.zeros(100)
    for d in np.linspace(1e-10,9.9999999e-1,100):
        kL_full[j] = kL_tot(d, eps, Mfunc, Rfunc, propA, propB)
        j = j+1
    return eps, kL, kL_full

def run_kL(Mfunc, Rfunc, eps, propA, propB):
    kL_full = np.zeros(100)
    j = 0
    for d in np.linspace(1e-10,9.9999999e-1,100):
        kL_full[j] = kL_tot(d, eps, Mfunc, Rfunc, propA, propB)
        j = j+1
    return kL_full


'''
Multicomponent Thermal Conductivity: Arbitrary Number of Components in Alloy

Set of lattice themral conductivity methods generalized to an arbitrary number of components
using the Berman, Foster, Ziman formulation for the Gamma scattering parameter.

Paper: Berman, Foster, and Ziman, Royal Society 1956

eps: list of epsilon values fit to each binary

(need to redo so it's the sum of binary terms)
'''

def gamma_multi(stoich, props : list, c : list):
    '''
    c : compositions for each component in the alloy
    props : lattice property (i.e. atomic mass/radius) for each component in the alloy
    
    Should add a check that c adds up to approximately 1
    '''
    num = 0
    denom = 0
    natoms = sum(stoich)
    for n in range(stoich):
        psite = sum([f * prop[n] for f, prop in zip(c, props)])
        denom = denom + stoich[n] * psite
        for i,j, in list(combinations(range(len(props)), 2)):
            num = num + stoich[n] * c[i] * c[j] * (props[i][n] - props[j][n])**2                       
    gamma = (num/natoms)/((denom/natoms)**2)
    return gamma  
      
def kL_from_gamma_multi(gamma, props : list, c : list):
    atmV = sum([c[i] * props[i]['atmV'] for i in range(len(c))])
    vs = sum([c[i] * props[i]['vs'] for i in range(len(c))])
    k0 = sum([c[i] * props[i]['k0'] for i in range(len(c))])
    prefix = (6**(1/3)/2)*(pi**(5/3)/kB)*(atmV**(2/3)/vs)
    u = (prefix * gamma * k0)**(1/2)
    kL = k0*np.arctan(u)/u
    return kL   

def kL_tot_multi(C, eps, Mfunc, Rfunc, components : list): 
    gamma = Mfunc(components[0]['stoich'], [c['atmMass'] for c in components], list(C)) +\
    eps * Rfunc(components[0]['stoich'], [c['atmRadius'] for c in components], list(C))
    kL = kL_from_gamma_multi(gamma, components, C)
    try:
        kL = kL[0]
    except:
        pass
    return kL  


#def fit_eps_kL_multi(Mfunc, Rfunc, Tt, At, propA, propB, p0 = 0):
#    eps, cov = curve_fit(lambda c, eps:\
#                         kL_tot_multi(c, eps, Mfunc, Rfunc, propA, propB), Tt,\
#                                At, p0 = p0, bounds = (0,np.inf), method = 'dogbox')
#    return eps


'''
CALPHAD-inspired thermal conductivity models:
    Redlich-Kister Polynomial for composition-dependent epsilon (strain
    scattering parameter)
    Muggianu Geometric Method to extrapolate from binary to high-order systems
'''

def kL_tot_RKpoly(C, coeffs, Mfunc, Rfunc, propA, propB, elaborate = False):
    '''
    Returns lattice thermal conductivity with a composition-depenndent epsilon
    fit to each binary alloy system (represented by a Redlich-Kister polynomial)
    '''
    n = 0
    eps = 0
    for c in coeffs:
        eps += c * (1 - 2*C) ** n
        n = n+1
    gammaM = Mfunc(propA['stoich'], propA['atmMass'], propB['atmMass'], C)
    gammaS = eps * Rfunc(propA['stoich'], propA['atmRadius'], propB['atmRadius'], C)
    gamma = gammaM + gammaS
    k0 = (1-C) * propA['k0'] + C * propB['k0']
    kL = kL_from_gamma(gamma, propA, propB, C)
    if elaborate:
        return kL, k0
    else:
        return kL

def muggianu_model_gamma(c, eps_list, Mfunc, Rfunc, prop_list : list):
    '''
    c : Quaternary composition (%SnSe, %PbSe, %SnTe, %PbTe)
    '''
    k = 0
    quat_gamma = 0
    for i, j in list(combinations(range(len(c)), 2)): #should be easy to genneralize with a permutation generator
        coeffX = (4 * c[i] * c[j]) / ((1 + c[i] - c[j]) * (1 + c[j] - c[i]))
        binX = [((1 + c[i] - c[j]) / 2) , ((1 + c[j] - c[i]) / 2)]
        gammaM = Mfunc(prop_list[j]['stoich'], prop_list[j]['atmMass'], prop_list[i]['atmMass'], binX[1])
        gammaS = 0
        n = 0
        for e in range(len(eps_list[k])):
            gammaS += eps_list[k][e] * (1 - 2 * binX[1]) ** n
            n = n+1
        gammaS = gammaS * Rfunc(prop_list[j]['stoich'], prop_list[j]['atmRadius'], prop_list[i]['atmRadius'], binX[1])
        gammaX = gammaM + gammaS
        quat_gamma = quat_gamma + coeffX * gammaX
        k = k+1
    return quat_gamma
    

def kL_tot_muggianu(c : list, eps_list, Mfunc, Rfunc, prop_list : list):
    gamma_quat = muggianu_model_gamma(c, eps_list, Mfunc, Rfunc, prop_list)
    kL = kL_from_gamma_multi(gamma_quat, prop_list, c)
    return kL



def fit_eps_poly(Mfunc, Rfunc, data, propA, propB, n_coeffs, p0 = [0,0,0]):
    coeffs, cov = curve_fit(lambda c, *eps_list:\
                         kL_tot_RKpoly(c, eps_list, Mfunc, Rfunc, propA, propB), data[0],\
                                data[1], p0 = p0, bounds = ([0]*n_coeffs,[np.inf]*n_coeffs), method = 'dogbox')
    return coeffs   

     

'''
Methods to Generate Thermal Conductivity Compositional Mapping
Run lattice thermal conductivity calculation over compositional range in order
to plot property versus composition heat maps
'''


def run_kL_tern(kLfunc, Mfunc, Rfunc, eps, n, prop_list : list):
    kL_full = dict()
    first = 1e-10
    last = 9.9999999999999e-1
    j = 0
    for c in np.arange(first,1, (last - first) / n):
        k = 0
        for d in np.arange(first, 1 - c, (last - first) / n):
            kL_full[(c*100,d*100)] = kLfunc((c,d), eps, Mfunc, Rfunc, prop_list)
            k = k+1
        j = j+1
    return kL_full


'''
QUATERNARY METHOD
'''

def run_kL_gamma_reciprocal_quat(kLfunc, Mfunc, Rfunc, eps_quat, n, prop_list):
    '''
    x : concentration of Se
    y : concentration of Sn
    '''
    kL_full = np.zeros([n+1, n+1])
    first = 1e-10
    last = 9.999999999e-01
    i = 0
    for x in np.arange(first, 1, (last-first)/n):
        j = 0
        for y in np.arange(first, 1, (last- first)/n):
            c_list = np.array([x * y, x * (1-y) , (1-x) * y, (1-x) * (1-y)]) 
            kL_full[j,i]  = kLfunc(c_list, eps_quat, Mfunc, Rfunc, prop_list)
            j = j+1
        i = i+1
    return kL_full



