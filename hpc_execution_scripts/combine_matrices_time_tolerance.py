#!/usr/bin/env python
#PBS -q array
#PBS -l select=1:ncpus=1
#PBS -l walltime=1:00:00
#PBS -j oe
#PBS -o /scratch_lg/image-content/ameyasd/logs/combined
#PBS -M ameya.s.daigavane@jpl.nasa.gov
#PBS -m abe

# This script combines all the file-wise confusion matrices on the new crossings dataset.
# Author: Ameya Daigavane

# External dependencies.
import os
import numpy as np
import yaml
import sys
from pathlib2 import Path
from collections import defaultdict

# Internal dependencies.
sys.path.append('/scratch_lg/image-content/ameyasd/europa-onboard-science/src/caps_els/secondary_scripts/') # Hack to import correctly.
from compute_file_list import list_of_ELS_files


# Makes all None values as NaN.
def fix(arr):
    arr[arr == None] = np.nan
    return arr  

# Load paths from config.
CONFIG_FILE = os.environ['CONFIG_FILE']
with open(CONFIG_FILE, 'r') as config_file_object:
    config = yaml.safe_load(config_file_object)
    RESULTS_DIR = config['RESULTS_DIR']
    DATA_DIR = config['DATA_DIR']
    LABELS_DIR = config['LABELS_DIR']
    LABELS_SUBDIR = config['LABELS_SUBDIR']
    MODE = config['MODE']
    COMBINED_MATRICES_DIR = config['COMBINED_MATRICES_DIR']
    COMBINED_MATRICES_FILE_BASENAME = config['COMBINED_MATRICES_FILE_BASENAME']
    TIME_TOLERANCE = config['TIME_TOLERANCE']
    NUM_DATAFILES = config['NUM_DATAFILES']

# For each year, what are the folders we need to look at?
folders_by_year = {}
folders_by_year['all'] = [RESULTS_DIR + dir + ('/%dmin/%s/confusion_matrices_time_tolerance/' % (TIME_TOLERANCE, LABELS_SUBDIR)) for dir in list_of_ELS_files(DATA_DIR, LABELS_DIR, MODE, remove_extension=True)]
for year in range(2004, 2013):
    folders_by_year[year] = [RESULTS_DIR + dir + ('/%dmin/%s/confusion_matrices_time_tolerance/' % (TIME_TOLERANCE, LABELS_SUBDIR)) for dir in list_of_ELS_files(DATA_DIR, LABELS_DIR, MODE, remove_extension=True) if 'ELS_%d' % year in dir]

# Number of files.
num_files = defaultdict(lambda: 0)

# Collect results for each year.
for year, folders in folders_by_year.items():
    results = {}

    for folder in folders:
        if os.path.exists(folder):
            for file in os.listdir(folder):
                algorithm = os.path.splitext(file)[0]
                file_full_path = folder + '/' + file

                # Accumulate confusion matrices at different thresholds, by summing.
                try:
                    if algorithm not in results:
                        results[algorithm] =  fix(np.load(file_full_path, allow_pickle=True)['confusion_matrices'])
                    else:
                        results[algorithm] += fix(np.load(file_full_path, allow_pickle=True)['confusion_matrices'])
                except KeyError:
                    print 'Ignoring old results file %s.' % file_full_path
                
                # Add one to the count for this algorithm.
                if year == 'all':
                    num_files[algorithm] += 1
        else:
            print '%s does not exist!' % folder

    # Check if we have the right number of files for each algorithm.
    if year == 'all':
        print 'Config file: %s' % CONFIG_FILE
        for algorithm in num_files:
            if num_files[algorithm] != NUM_DATAFILES:
                print 'Algorithm %s has only %d confusion matrix files, out of %d.' % (algorithm, num_files[algorithm], NUM_DATAFILES)

    print 'Year \'%s\' Results: %d algorithms.' % (year, len(results))

    # Create the directory if it doesn't exist.
    if not os.path.exists(COMBINED_MATRICES_DIR):
        Path(COMBINED_MATRICES_DIR).mkdir(parents=True, exist_ok=True)

    # Save to a .npz file.
    for algorithm in results:
        np.savez(COMBINED_MATRICES_FILE_BASENAME % str(year), **results)
    
