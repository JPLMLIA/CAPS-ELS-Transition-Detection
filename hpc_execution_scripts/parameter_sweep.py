#!/usr/bin/env python
#PBS -q array
#PBS -l select=1:ncpus=1
#PBS -l walltime=48:00:00
#PBS -J 0-9999:1
#PBS -j oe
#PBS -o /scratch_lg/image-content/ameyasd/logs/combined
#PBS -M ameya.s.daigavane@jpl.nasa.gov
#PBS -m abe

# This script runs each algorithm on each datafile, as a job array.
# Author: Ameya Daigavane

# External dependencies.
from __future__ import division
import os
import sys
import yaml
import re
from string import Template
from pathlib2 import Path

# Internal dependencies.
sys.path.append('/scratch_lg/image-content/ameyasd/europa-onboard-science/src/caps_els/secondary_scripts/') # Hack to import correctly.
from compute_file_list import list_of_ELS_files

# Load paths from config.
CONFIG_FILE = os.environ['CONFIG_FILE']
with open(CONFIG_FILE, 'r') as config_file_object:
    config = yaml.safe_load(config_file_object)
    SCRIPTS_DIR = config['SCRIPTS_DIR']
    DATA_DIR = config['DATA_DIR']
    LABELS_DIR = config['LABELS_DIR']
    RESULTS_DIR = config['RESULTS_DIR']
    TRANSFORM = config['TRANSFORM']
    PCA_COMPONENTS_DIR = config['PCA_COMPONENTS_DIR']
    COMMANDS_FILE = config['COMMANDS_FILE']
    BLUR_SIGMA = config['BLUR_SIGMA']
    BIN_SELECTION = config['BIN_SELECTION']
    FILTER = config['FILTER']
    FILTER_SIZE = config['FILTER_SIZE']
    MODE = config['MODE']

# Read the contents of COMMANDS_FILE.
with open(COMMANDS_FILE, 'r') as f:
    lines = f.readlines()
    lines = filter(lambda line: line.startswith('${'), lines)

# List of data files used for evaluation.
files = list_of_ELS_files(DATA_DIR, LABELS_DIR, MODE)

# Current job array index.
PBS_ARRAY_INDEX = int(os.environ['PBS_ARRAY_INDEX'])

# Multiplex over lines (algorithms with pre-processing) and data files.
line = lines[PBS_ARRAY_INDEX % len(lines)]
data_file = files[PBS_ARRAY_INDEX // len(lines)]

# Where to keep the scores files.
results_subdir = RESULTS_DIR + os.path.splitext(data_file)[0]
if not os.path.exists(results_subdir):
    Path(results_subdir).mkdir(parents=True, exist_ok=True)

# Use the full path of the scores file.
data_file = DATA_DIR + data_file

# Add the algorithm name to the PCA components directory path.
# The parameters which are already filled in won't be changed by safe_substitute() below.
pattern = re.compile('\\w*.hdf5')
algorithm_with_ext = pattern.search(line).group()
algorithm = os.path.splitext(algorithm_with_ext)[0]

# Package all these as a dictionary.
params_dict = {
    'SCRIPTS_DIR': SCRIPTS_DIR,
    'TRANSFORM': TRANSFORM,
    'BLUR_SIGMA': BLUR_SIGMA,
    'FILTER': FILTER,
    'FILTER_SIZE': FILTER_SIZE,
    'BIN_SELECTION': BIN_SELECTION,
    'PCA_COMPONENTS_DIR': PCA_COMPONENTS_DIR + algorithm + '/',
    'DATA_FILE': data_file,
    'RESULTS_SUBDIR': results_subdir,
}

# Remove newline at the end. and convert to a template string.
line = line.rstrip()
line = Template(line)

# Fill in each line with the missing paths and parameters.
filled_line = line.safe_substitute(**params_dict)

# If the scores file doesn't exist, then valid = True. Otherwise, valid = False.
pattern = re.compile('[-_\\w\\\\/]*.hdf5')
scores_file_full_path = pattern.search(filled_line).group()

command = filled_line
if not os.path.exists(scores_file_full_path):
    print(command)
    os.system(command)
