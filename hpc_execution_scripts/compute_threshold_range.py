#!/usr/bin/env python
#PBS -q array
#PBS -l select=1:ncpus=1
#PBS -l walltime=1:00:00
#PBS -j oe
#PBS -o /scratch_lg/image-content/ameyasd/logs/combined
#PBS -M ameya.s.daigavane@jpl.nasa.gov
#PBS -m abe

# This script computes the score threshold range by computing the range of scores for each algorithm, across all result files, and dividing this into 100 parts.
# Author: Ameya Daigavane

# External dependencies.
import os
import numpy as np
import h5py
import sys
from collections import defaultdict
import yaml
from pathlib2 import Path

# Internal dependencies.
sys.path.append('/scratch_lg/image-content/ameyasd/europa-onboard-science/src/caps_els/secondary_scripts/') # Hack to import correctly.
from compute_file_list import list_of_ELS_files
from compute_algorithm_params import list_of_algorithms

# Load from config.
CONFIG_FILE = os.environ['CONFIG_FILE']
with open(CONFIG_FILE, 'r') as config_file_object:
    config = yaml.safe_load(config_file_object)
    RESULTS_DIR = config['RESULTS_DIR']
    DATA_DIR = config['DATA_DIR']
    LABELS_DIR = config['LABELS_DIR']
    THRESHOLDS_DIR = config['THRESHOLDS_DIR']
    MODE = config['MODE']
    COMMANDS_FILE = config['COMMANDS_FILE']
    NUM_DATAFILES = config['NUM_DATAFILES']

min_scores = defaultdict(lambda: np.inf)
max_scores = defaultdict(lambda: -np.inf)
num_files = defaultdict(lambda: 0)

folders = [RESULTS_DIR + dir for dir in list_of_ELS_files(DATA_DIR, LABELS_DIR, MODE, remove_extension=True)]
algorithms = list_of_algorithms(COMMANDS_FILE, remove_extension=True)

# Iterate through all folders.
for folder in folders:
    for algorithm in algorithms:
        file_full_path = folder + '/' + algorithm + '.hdf5'

        try:
            with h5py.File(file_full_path, 'r') as filedata:
                scores = filedata['scores'][()]
                min_scores[algorithm] = min(min_scores[algorithm], np.min(scores))
                max_scores[algorithm] = max(max_scores[algorithm], np.max(scores))
                num_files[algorithm] += 1
        except IOError:
            pass

# Thresholds are 100 linearly-spaced points from the min score to the max score over all files.
thresholds = {}
for algorithm in min_scores:
    algorithm_min_score = min_scores[algorithm]
    algorithm_max_score = max_scores[algorithm]
    thresholds[algorithm] = np.linspace(algorithm_min_score, algorithm_max_score, 100)

print 'Thresholds:', thresholds
print 'Number of algorithms:', len(thresholds)
print 'Total number of ELS files for this mode:', NUM_DATAFILES

# Create directory if it doesn't exist.
if not os.path.exists(THRESHOLDS_DIR):
    Path(THRESHOLDS_DIR).mkdir(parents=True, exist_ok=True)

# Save thresholds, one-by-one.
for algorithm, thresholds_range in thresholds.items():
    if num_files[algorithm] == NUM_DATAFILES:
        np.save(THRESHOLDS_DIR + algorithm + '_thresholds.npy', thresholds_range)
        print 'Algorithm %s has had thresholds saved.' % (algorithm)
    else:
        print 'Algorithm %s has %d files out of %d, and has not had thresholds saved.' % (algorithm, num_files[algorithm], NUM_DATAFILES)
