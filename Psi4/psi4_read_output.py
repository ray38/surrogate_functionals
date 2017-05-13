# -*- coding: utf-8 -*-
"""
Created on Sun May 07 22:39:53 2017

@author: Ray
"""

import sys
import json
#import pprint

filename = sys.argv[1]
try:
    output_filename = sys.argv[2]
except:
    output_filename = 'data.json'


result = {}


reading_basis_set = False
with open(filename,'rb') as f:
    for line in f:
        if line.startswith('#@!#@!Molecule:'):
            temp_molecule = line[15:].strip()
            if temp_molecule not in result:
                result[temp_molecule] = {}
                
                
        if line.startswith('!@#!@#Method:'):
            temp_functional = line[13:].strip()
            if temp_functional not in result[temp_molecule]:
                result[temp_molecule][temp_functional] = {}
                
                
        if line.strip().startswith('==> Primary Basis <=='):
            reading_basis_set = True
        if reading_basis_set:
            if line.strip().startswith('Basis Set:'):
                temp_basis = line.strip()[11:].strip()
                if temp_basis not in result[temp_molecule][temp_functional]:
                    result[temp_molecule][temp_functional][temp_basis] = {}
                reading_basis_set = False
        
        if line.strip().startswith('Nuclear Repulsion Energy ='):
            result[temp_molecule][temp_functional][temp_basis]['nuclear_repulsion'] = float(line.strip().split()[4])
            
        if line.strip().startswith('One-Electron Energy ='):
            result[temp_molecule][temp_functional][temp_basis]['one_electron'] = float(line.strip().split()[3])
            
        if line.strip().startswith('Two-Electron Energy ='):
            result[temp_molecule][temp_functional][temp_basis]['two_electron'] = float(line.strip().split()[3])
            
        if line.strip().startswith('DFT Exchange-Correlation Energy ='):
            result[temp_molecule][temp_functional][temp_basis]['psi4_xc'] = float(line.strip().split()[4])
            
        if line.strip().startswith('Empirical Dispersion Energy ='):
            result[temp_molecule][temp_functional][temp_basis]['empirical_dispersion'] = float(line.strip().split()[4])
            
        if line.strip().startswith('VV10 Nonlocal Energy ='):
            result[temp_molecule][temp_functional][temp_basis]['vv10'] = float(line.strip().split()[4])
            
        if line.strip().startswith('PCM Polarization Energy ='):
            result[temp_molecule][temp_functional][temp_basis]['pcm_polarization'] = float(line.strip().split()[4])
            
        if line.strip().startswith('Total Energy ='):
            result[temp_molecule][temp_functional][temp_basis]['psi4_total_energy'] = float(line.strip().split()[3])

composition_dict = {'C4H6':{'C':4, 'H':6},
                    'C2H2':{'C':2, 'H':2},
                    'C2H4':{'C':2, 'H':4},
                    'C2H6':{'C':2, 'H':6},
                    'C3H4':{'C':3, 'H':4},
                    'C3H6':{'C':3, 'H':6},
                    'C3H8':{'C':3, 'H':8},
                    'CH2':{'C':1, 'H':2},
                    'CH2OCH2':{'C':2, 'H':4, 'O':1},
                    'CH3CH2OH':{'C':2, 'H':6, 'O':1},
                    'CH3CH2NH2':{'N':1, 'C':2, 'H':7},
                    'CH3CHO':{'C':2, 'H':4, 'O':1},
                    'CH3CN':{'N':1, 'C':2, 'H':3},
                    'CH3COOH':{'C':2, 'H':4, 'O':2},
                    'CH3NO2':{'N':1, 'C':1, 'H':3, 'O':2},
                    'CH3OCH3':{'C':2, 'H':6, 'O':1},
                    'CH3OH':{'C':1, 'H':4, 'O':1},
                    'CH4':{'C':1, 'H':4},
                    'CO2':{'C':1, 'O':2},
                    'CO':{'C':1, 'O':1},
                    'H2':{'H':2},
                    'H2CCO':{'C':2, 'H':2, 'O':1},
                    'H2CO':{'C':1, 'H':2, 'O':1},
                    'H2O2':{'H':2, 'O':2},
                    'H2O':{'H':2, 'O':1},
                    'H3CNH2':{'N':1, 'C':1, 'H':5},
                    'HCN':{'N':1, 'C':1, 'H':1},
                    'C4H8':{'C':4, 'H':8},
                    'HCOOH':{'C':1, 'H':2, 'O':2},
                    'HNC':{'N':1, 'C':1, 'H':1},
                    'N2':{'N':2},
                    'N2H4':{'N':2, 'H':4},
                    'N2O':{'N':2, 'O':1},
                    'NCCN':{'N':2, 'C':2},
                    'NH3':{'N':1, 'H':3},
                    'isobutene':{'C':4, 'H':8},
                    'glycine':{'N':1, 'C':2, 'H':5, 'O':2},
                    'C2H5CN':{'N':1, 'C':3, 'H':5},
                    'butadiene':{'C':4, 'H':6},
                    '1-butyne':{'C':4, 'H':6},
                    'CCH':{'C':2, 'H':1},
                    'propanenitrile':{'N':1, 'C':3, 'H':5},
                    'NO2':{'N':1, 'O':2},
                    'NH':{'N':1, 'H':1},
                    'pentadiene':{'C':5, 'H':8},
                    'cyclobutene':{'C':4, 'H':6},
                    'NO':{'N':1, 'O':1},
                    'OCHCHO':{'C':2, 'H':2, 'O':2},
                    'cyclobutane':{'C':4, 'H':8},
                    'propyne':{'C':3, 'H':4},
                    'CH3':{'C':1, 'H':3},
                    'NH2':{'N':1, 'H':2},
                    'CH3NHCH3':{'N':1, 'C':2, 'H':7},
                    'CH':{'C':1, 'H':1},
                    'CN':{'N':1, 'C':1},
                    'z-butene':{'C':4, 'H':8},
                    '1-butene':{'C':4, 'H':8},
                    'isobutane':{'C':4, 'H':10},
                    '2-propanamine':{'N':1, 'C':3, 'H':9},
                    'cyclopentane':{'C':5, 'H':10},                
                    'butane':{'C':4, 'H':10},
                    'HCO':{'C':1, 'H':1, 'O':1},
                    'CH3CONH2':{'N':1, 'C':2, 'H':5, 'O':1},
                    'e-butene':{'C':4, 'H':8},
                    'CH3O':{'C':1, 'H':3, 'O':1},
                    'propene':{'C':3, 'H':6},
                    'OH':{'H':1, 'O':1},
                    'methylenecyclopropane':{'C':4, 'H':6},
                    'C6H6':{'C':6, 'H':6},
                    'trimethylamine':{'N':1, 'C':3, 'H':9},
                    'cyclopropane':{'C':3, 'H':6},
                    'H2CCHCN':{'N':1, 'C':3, 'H':3},
                    '1-pentene':{'C':5, 'H':10},
                    '2-butyne':{'C':4, 'H':6},
                    'O3':{'O':3}}

for molecule in result:
    if molecule in composition_dict:
        result[molecule]['composition'] = composition_dict[molecule]

#pprint.pprint(result)
with open(output_filename, 'w') as outfile:
    json.dump(result, outfile)
        
        
        
