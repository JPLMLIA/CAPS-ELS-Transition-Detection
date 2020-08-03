#!/usr/bin/env python
#PBS -q shortq
#PBS -l select=2:ncpus=4
#PBS -l walltime=3:00:00
#PBS -J 0-9998:1
#PBS -j oe
#PBS -o /scratch_lg/image-content/ameyasd/logs/combined
#PBS -M ameya.s.daigavane@jpl.nasa.gov
#PBS -m abe

# To evaluate entire directories on the HPC cluster.

# NOTE
# '#PBS' directives must immediately follow your shell initialization line '#!/bin/<shell>'
# '#PBS' directives must be consecutively listed without any empty lines in-between directives
# Reference the PBS Pro User Guide for #PBS directive options.
# To determine the appropriate queue (#PBS -q) and walltime (#PBS -l walltime) to use,
#  run (qmgr -c 'print server') on the login node.

# This is an example "Job Array" job script
# PBS Job Arrays are typically used for embarrassingly parallel codes
# The "#PBS -J" directive specifies an index range, which correlates to the amount of jobs spawned
# Input files can be named according to the index (input.1, input.2, ..etc)
# Reference the PBS Pro User Guide to learn about the power of Job Arrays.

import os
import yaml
import sys
from pathlib import Path

# Load paths from config.
CONFIG_FILE = os.environ['CONFIG_FILE']
with open(CONFIG_FILE, 'r') as config_file_object:
    config = yaml.safe_load(config_file_object) 
    SCRIPTS_DIR = config['SCRIPTS_DIR']
    DATA_DIR = config['DATA_DIR']
    LABELS_DIR = config['LABELS_DIR']
    RESULTS_DIR = config['RESULTS_DIR']

# Run executable
# Job Arrays step through the index range specified by '#PBS -J'
# The value of $PBS_ARRAY_INDEX changes as each consecutive job is launched

# Unpack the parameters dictionary as a string.
def unpack_as_string(params_dict, sep=' '):
    return ('%s' % sep).join(['-%s%s%s' % (param, sep, value) for param, value in params_dict.iteritems()])


# 'test' mode checks if we have any errors in hpc_evaluate.py and runs the script as if PBS_ARRAY_INDEX were 0.
# 'eval' mode is what you should use to actually run on the cluster as a job array.
if len(sys.argv) >= 2 and sys.argv[1] == 'test':
    mode = 'test'
else:
    mode = 'eval'

# Get the array index.
if mode == 'eval':
    PBS_ARRAY_INDEX = int(os.environ['PBS_ARRAY_INDEX'])
elif mode == 'test':
    PBS_ARRAY_INDEX = 0
else:
    raise ValueError('Invalid mode.')

# Load configuration.
with open(SCRIPTS_DIR + 'secondary_scripts/hpc_config.yaml', 'r') as config_file_object:
    config = yaml.safe_load(config_file_object)

# Load the data-files.
if config['data_files'][0] == 'all':
    file_list = [os.path.splitext(file_name)[0] for file_name in os.listdir(DATA_DIR)]
elif config['data_files'][0] == 'trainset':
    file_list = [os.path.splitext(file_name)[0] for file_name in os.listdir(DATA_DIR) if '2004' in file_name]
elif config['data_files'][0] == 'testset':
    file_list = [os.path.splitext(file_name)[0] for file_name in os.listdir(DATA_DIR) if '2004' not in file_name]
else:
    file_list = config['data_files']

# Remove duplicates.
file_list = list(sorted(set(file_list)))

num_data_files = len(file_list)
num_algorithms = len(config['algorithms'])

# Choose DAT file and the corresponding labels file.
filename_no_ext = file_list[PBS_ARRAY_INDEX // num_algorithms]
filename_data = DATA_DIR + filename_no_ext + '.DAT'
filename_labels = LABELS_DIR + filename_no_ext + '.yaml'

# Load the right algorithm and parameters.
algorithm = config['algorithms'][PBS_ARRAY_INDEX % num_algorithms]
algorithm_class = config['algorithm_class'][algorithm]
params = config['algorithm_parameters'][algorithm]
params_unpacked = unpack_as_string(params)

# Where to put the results.
if mode == 'eval':
    if not os.path.exists(RESULTS_DIR + filename_no_ext):
        Path(RESULTS_DIR + filename_no_ext).mkdir(parents=True, exist_ok=True)


results_file = RESULTS_DIR + filename_no_ext + '/' + algorithm + '.hdf5'

# If the results file already exists, exit now.
if os.path.exists(results_file):
    exit(0)

# The script we want to run.
script = SCRIPTS_DIR + 'find_scores.py'
command = '%s %s %s %s %s' % (script, algorithm_class, filename_data, results_file, params_unpacked)

if mode == 'test':
    print 'When PBS_ARRAY_INDEX is 0, the following script will be called:'
    print command
    print
    print 'Over the whole job array, %d algorithm(s) will be run, each on %d datafile(s), for a total of %d evaluations.' % (num_algorithms, num_data_files, num_algorithms * num_data_files)
    print 'Algorithm(s) to be run: %s.' % config['algorithms']
    print
    print 'Note: Please ensure the job array size is bigger than the total number of evaluations! Unfortunately, this script cannot change this value programmatically.'
    print 'Press \'y\' to continue execution of the sample script above, and anything else to quit.'
    choice = raw_input()
    if choice not in ['y', 'Y']:
        exit(0)

# Call the script.
os.system(command)
