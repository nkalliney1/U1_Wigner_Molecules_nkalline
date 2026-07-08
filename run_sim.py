import subprocess
import sys
import tenpy
import yaml
from su2_model import corr_function_ss, SU2Model
import matplotlib.pyplot as plt
import numpy as np

#run like:
#python run_sim.py my_model.py simulation.yml

def get_output_filename(params):
    fn_params = params['output_filename_params']
    prefix = fn_params['prefix']
    suffix = fn_params['suffix']
    parts = fn_params['parts']
    
    # build the middle parts by looking up each key in the params dict
    part_strings = []
    for key_path, fmt in parts.items():
        # navigate nested dict using dot-separated key path e.g. 'algorithm_params.trunc_params.chi_max'
        keys = key_path.split('.')
        value = params
        for k in keys:
            value = value[k]
        part_strings.append(fmt.format(value))
    
    filename = '_'.join([prefix] + part_strings) + suffix
    return filename

model = sys.argv[1]
#if model was input as a .py file, remove suffix
if model[-3:] == ".py":
    model = model[:-3]
yaml_file = sys.argv[2]

#load yaml file to find output file name
with open(yaml_file, 'r') as f:
    data = yaml.full_load(f)
filename = get_output_filename(data)

#terminal command
#equivalent to python -m tenpy -i my_model simulation.yml
#run the simulation
cmd = ['python','-m', 'tenpy', '-i', model, yaml_file]
res = subprocess.run(cmd, stdout=subprocess.PIPE).stdout.decode('utf-8')
#print any warnings/outputs
print(res)

#if filename has changed, find new filename
#find where this phrase starts
start = res.find("changed output filename to ")
if start != -1: #file name has changed
    name_idx = start + 27 #length of prefix string
    end = res.find("\n", name_idx)
    filename = res[name_idx:end]

#get results of the simulation
results = tenpy.tools.hdf5_io.load('results/' + filename)
L = results['psi'].L
corr = corr_function_ss(results['psi'], list(range(L)), list(range(L)))

#create the sites again so we can plot them
#idk a better way to do this rn
model = SU2Model(data['model_params'])
sites = model.lat.mps_sites() # create sites

print("FILE NAME: " + filename)

# plot nearest neighbor correlations
fig, ax = plt.subplots()
model.lat.plot_sites(ax)
model.lat.plot_coupling_correlations(ax, corr, cmap='RdBu_r', value_func=np.real, vmin=-3/4, vmax=1/4)
ax.set_aspect('equal')
plt.savefig(f'results/{filename}_corr.png')

# plot energy per iteration
fig2, ax2 = plt.subplots()
energy_arr = results['sweep_stats']['E']
ax2.plot(np.arange(len(energy_arr)), energy_arr)
plt.savefig(f'results/{filename}_energy.png')

#plot maximum truncation error per iteration
fig3, ax3 = plt.subplots()
max_trunc_err_arr = results['sweep_stats']['max_trunc_err']
ax3.plot(np.arange(len(max_trunc_err_arr)), max_trunc_err_arr)
plt.savefig(f'results/{filename}_trunc_err.png')