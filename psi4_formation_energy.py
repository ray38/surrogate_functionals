# -*- coding: utf-8 -*-
"""
Created on Thu May 11 16:17:08 2017

@author: ray
"""

import numpy as np
import json
import sys
import csv







def convert_formation_energies(energy_dict,atomic_references,composition_dict):
    """
    Convert dictionary of energies, atomic references and compositions into a dictionary of formation energies
    :param energy_dict: Dictionary of energies for all species.
                        Keys should be species names and values
                        should be energies.
                        
    :type energy_dict: dict
    :param atomic_references: Dictionary of atomic reference compositions (e.g. {H2O:{H:2,O:2}})
    :type atomic_references: dict
    :param composition_dict: Dictionary of compositions
    :type composition_dict: dict
    .. todo:: Explain the keys and values for energy_dict, atomic_references, and composition_dict
    """
    n = len(atomic_references)
    R = np.zeros((n,n))
    e = []
    ref_offsets = {}
    atoms = sorted(atomic_references)
    print atoms
    for i,a in enumerate(atoms):
        composition = composition_dict[atomic_references[a]]
        e.append(energy_dict[atomic_references[a]])
        for j,a in enumerate(atoms):
            n_a = composition.get(a,0)
            R[i,j] = n_a
    if not np.prod([R[i,i] for i in range(0,n)]):
        raise ValueError('Reference set is not valid.')
    e1 = []
    for i in range(len(e)):
        e1.append(e[i][0])
    e1 = np.array(e1)
    try:
        R_inv = np.linalg.solve(R,np.eye(n))
    except np.linalg.linalg.LinAlgError:
        raise ValueError('Reference set is not valid.')
    x = list(np.dot(R_inv,e1))
    for i,a in enumerate(atoms):
        ref_offsets[a] = x[i]

    return ref_offsets

def get_formation_energies(energy_dict,ref_dict,composition_dict):
    formation_energies = {}
    for molecule in energy_dict:
        E = energy_dict[molecule]
        for atom in composition_dict[molecule]:
            E -= float(composition_dict[molecule][atom]) *ref_dict[atom]
        formation_energies[molecule] = E
    return formation_energies


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
data_filename = sys.argv[1]

data = {}
with open(data_filename,'rb') as f:
    for line in f:
        if line.strip() != '':
            temp = line.strip().split()
            temp_name = temp[0]
            temp_original_energy = float(temp[1])
            temp_predict_energy  = float(temp[2])
            data[temp_name] = {}
            data[temp_name]['predict_exc'] = temp_predict_energy
            data[temp_name]['original_exc'] = temp_original_energy

print data
for molecule in data:
    if 'composition' not in data[molecule]:
        if molecule in composition_dict:
            data[molecule]['composition'] = composition_dict[molecule]

original_energy_dict = {}
predict_energy_dict = {}
composition_dict = {}
for molecule in data:
    if 'composition' in data[molecule] and 'predict_exc' in data[molecule] and 'original_exc' in data[molecule]:
        print molecule
        original_energy_dict[molecule] = data[molecule]['original_exc']

                                
        predict_energy_dict[molecule] = data[molecule]['predict_exc']

        composition_dict[molecule] = data[molecule]['composition']


#atomic_references = {'O':'CH3CH2OH','H':'C2H2','C':'C2H6'}
atomic_references = {'N':'NH3','O':'H2O','H':'H2','C':'CH4'}
compound_original_en_dict = {}
compound_predict_en_dict = {}

for key in original_energy_dict:

    compound_original_en_dict[key]     = [original_energy_dict[key]]
    compound_predict_en_dict[key]      = [predict_energy_dict[key]]



ref_offset_original_en       = convert_formation_energies(compound_original_en_dict.copy(),atomic_references,composition_dict)
ref_offset_predict_en        = convert_formation_energies(compound_predict_en_dict.copy(),atomic_references,composition_dict)



formation_energies_original_en       = get_formation_energies(compound_original_en_dict.copy(),ref_offset_original_en.copy(),composition_dict)
formation_energies_predict_en        = get_formation_energies(compound_predict_en_dict.copy(),ref_offset_predict_en.copy(),composition_dict)
#formation_energies_compare_en   = get_formation_energies(compare_energy_dict.copy(),ref_offset_compare_en.copy(),composition_dict)


print '{:10}\t{}\t{}'.format('name', 'form. E. 1', 'form. E. 2')
print '--------- 1: predicted xc energy  2: psi4 xc energy projected on fd-grid\n'
for key in formation_energies_original_en.keys():
    #print '{:10}\t{:8.5f}\t{:8.5f}'.format(key,formation_energies_original_en[key],formation_energies_predict_en[key])
    print '{}\t{}\t{}\t{}\t{}'.format(key,compound_original_en_dict[key][0],compound_predict_en_dict[key][0],formation_energies_original_en[key][0],formation_energies_predict_en[key][0])


with open("formation_energies.csv", "wb") as f:
    writer = csv.writer(f)
    for key in formation_energies_original_en.keys():
        temp = [key,compound_original_en_dict[key][0],compound_predict_en_dict[key][0],formation_energies_original_en[key][0],formation_energies_predict_en[key][0]]
        writer.writerow(temp)