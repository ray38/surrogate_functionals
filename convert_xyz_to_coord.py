import json
import sys


xyz_filename = sys.argv[1]
molecule_name = sys.argv[2]

with open(xyz_filename ) as f:
    content = f.readlines()


result_dict = {}
result_dict[molecule_name] = {}
result_dict[molecule_name]["symmetry"] = "c1"

coordinates = []
atoms = []
for line in content:
	temp = line.strip().split()
	if len(temp) != 0:
		atoms.append(temp[0])
		coordinates.append([float(temp[1]),float(temp[2]),float(temp[3])])

result_dict[molecule_name]["atoms"] = atoms
result_dict[molecule_name]["coordinates"] = coordinates 

with open('convert_result.json', 'w') as f:
    json.dump(result_dict, f, indent=4)