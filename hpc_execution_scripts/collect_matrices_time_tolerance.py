#!/usr/bin/env python
#PBS -q array
#PBS -l select=1:ncpus=1
#PBS -l walltime=1:00:00
#PBS -J 0-9999:1
#PBS -j oe
#PBS -o /scratch_lg/image-content/ameyasd/logs/combined
#PBS -M ameya.s.daigavane@jpl.nasa.gov
#PBS -m abe

# This script computes all the file-wise confusion matrices on the new crossings dataset.
# Author: Ameya Daigavane

# External dependencies.
import os
import yaml
import sys
from pathlib2 import Path

# Internal dependencies.
sys.path.append('/scratch_lg/image-content/ameyasd/europa-onboard-science/src/caps_els/secondary_scripts/') # Hack to import correctly.
from compute_file_list import list_of_ELS_files
from compute_algorithm_params import list_of_algorithms

# Load paths from config.
CONFIG_FILE = os.environ['CONFIG_FILE']
with open(CONFIG_FILE, 'r') as config_file_object:
    config = yaml.safe_load(config_file_object)
    SCRIPTS_DIR = config['SCRIPTS_DIR']
    DATA_DIR = config['DATA_DIR']
    LABELS_SUBDIR = config['LABELS_SUBDIR']
    LABELS_DIR = config['LABELS_DIR']
    MODE = config['MODE']
    RESULTS_DIR = config['RESULTS_DIR']
    TIME_TOLERANCE = config['TIME_TOLERANCE']
    THRESHOLDS_DIR = config['THRESHOLDS_DIR']
    COMMANDS_FILE = config['COMMANDS_FILE']
    NUM_ALGORITHMS = config['NUM_ALGORITHMS']

# Current job array index. 
PBS_ARRAY_INDEX = int(os.environ['PBS_ARRAY_INDEX'])

# Every result (HDF5) file is in a folder, named according to the ELS file.
folders = list_of_ELS_files(DATA_DIR, LABELS_DIR, MODE, remove_extension=True)
folder = folders[PBS_ARRAY_INDEX // NUM_ALGORITHMS]

files = list_of_algorithms(COMMANDS_FILE)
result_file = files[PBS_ARRAY_INDEX % NUM_ALGORITHMS]
result_file_full = RESULTS_DIR + folder + '/' + result_file
result_file_no_ext = os.path.splitext(result_file)[0]

script = SCRIPTS_DIR + 'evaluate_methods_time_tolerance.py'
labels_file = LABELS_DIR + os.path.splitext(folder)[0] + '.yaml'
output_directory = RESULTS_DIR + folder + '/%dmin/' % TIME_TOLERANCE + LABELS_SUBDIR

if not os.path.exists(output_directory):
    Path(output_directory).mkdir(parents=True, exist_ok=True)

print('%s %s -l %s -opd %s -tf %s -t %d --no_plots' % (script, result_file_full, labels_file, output_directory, THRESHOLDS_DIR, TIME_TOLERANCE))
os.system('%s %s -l %s -opd %s -tf %s -t %d --no_plots' % (script, result_file_full, labels_file, output_directory, THRESHOLDS_DIR, TIME_TOLERANCE))
