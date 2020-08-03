#!/usr/bin/env python
#PBS -q shortq
#PBS -l select=1:ncpus=4
#PBS -l walltime=3:00:00
#PBS -J 0-1262:1
#PBS -o /halo_nobackup/image-content/ameyasd/crossings_updated_results/stdout
#PBS -e /halo_nobackup/image-content/ameyasd/crossings_updated_results/stderr

# This script collects all the results on the new crossings dataset.
# Author: Ameya Daigavane

# External dependencies.
import os

# Where the labels and the results are, on Halo HPC.
LABELS_DIR = '/halo_nobackup/image-content/ameyasd/crossings_updated/labels/'
RESULTS_DIR = '/halo_nobackup/image-content/ameyasd/crossings_updated_results/'
SCRIPTS = '/halo_nobackup/image-content/ameyasd/europa-onboard-science/src/caps_els/'

# Where the labels are locally. For testing purposes only!
# LABELS_DIR = 'temp/crossings_updated/labels/'
# RESULTS_DIR = 'temp/crossings_updated_results/'

# Current index. 
PBS_ARRAY_INDEX = int(os.environ['PBS_ARRAY_INDEX'])

# Every result (HDF5) file is in a folder.
folder = os.listdir(RESULTS_DIR)[PBS_ARRAY_INDEX]

if folder == 'stderr' or folder == 'stdout':
   exit(0)

# Each folder contains these files.
algorithms = ['hmm', 'hotsax', 'matrix_profile', 'rulsif']
result_files = [algorithm + '.hdf5' for algorithm in algorithms]

#output_directory = RESULTS_DIR + folder + '/confusion_matrices/'
output_directory = RESULTS_DIR + folder
#if not os.path.exists(output_directory):
#   os.mkdir(output_directory)

script = SCRIPTS + 'evaluate_methods.py'
labels_file = LABELS_DIR + os.path.splitext(folder)[0] + '.yaml'
error_window = 100

# Save confusion matrices.
for result_file in os.listdir(RESULTS_DIR + folder):
    result_file_full = RESULTS_DIR + folder + '/' + result_file

    # Call evaluation script.
    if not os.path.isdir(result_file_full):
        os.system('%s %s -err %s -l %s -opd %s --no_plots' % (script, result_file_full, error_window, labels_file, output_directory))
