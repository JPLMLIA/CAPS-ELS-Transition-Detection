#!/usr/bin/env python
#PBS -q array
#PBS -l select=1:ncpus=1
#PBS -l walltime=1:00:00
#PBS -j oe
#PBS -o /scratch_lg/image-content/ameyasd/logs/combined
#PBS -M ameya.s.daigavane@jpl.nasa.gov
#PBS -m abe

# This script computes PCA components from the training data, storing them in a folder for the algorithms to use later.
# Author: Ameya Daigavane

# External dependencies.
from __future__ import division
import os
import yaml
import numpy as np
import sys
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pathlib2 import Path

# Internal dependencies.
sys.path.append('/scratch_lg/image-content/ameyasd/europa-onboard-science/src/caps_els/') # Hack to import correctly.
sys.path.append('/scratch_lg/image-content/ameyasd/europa-onboard-science/src/caps_els/secondary_scripts/') # Hack to import correctly.
from data_utils import get_ELS_data
from transform import Transformation
from compute_file_list import list_of_ELS_files
from compute_algorithm_params import list_of_algorithms, preprocessing_parameters

# Get PCA components for 1, 2, 5 and 10 dimensions.
def compute_pca_components(BLUR_SIGMA, BIN_SELECTION, FILTER, FILTER_SIZE, TRANSFORM, **kwargs):

    # We compute PCA components using all the training data.
    file_list = list_of_ELS_files(DATA_DIR, CROSSINGS_DIR + 'new_labels/all/', MODE.replace('test', 'train'))

    # Append all ELS data one-by-one.
    all_counts = None
    for data_file in file_list:
        file_full = DATA_DIR + data_file
        counts, energy_range, times = get_ELS_data(file_full, 'anode5', datetime.min, datetime.max, blur_sigma=BLUR_SIGMA, bin_selection=BIN_SELECTION, filter=FILTER, filter_size=FILTER_SIZE)
        
        if all_counts is None:
            all_counts = counts
        else:
            all_counts = np.append(all_counts, counts, axis=0)
        
    # Apply transformation.
    all_counts = Transformation(TRANSFORM).transform(all_counts)

    # Learn PCA components from data.
    pca = PCA(n_components=10)
    pca.fit(all_counts)
    
    # We compute PCA components for these dimensions only.
    return {
        n_components: pca.components_[:n_components] for n_components in [1, 2, 5, 10]
    }


# Load paths from config.
CONFIG_FILE = os.environ['CONFIG_FILE']
with open(CONFIG_FILE, 'r') as config_file_object:
    config = yaml.safe_load(config_file_object)
    DATA_DIR = config['DATA_DIR']
    CROSSINGS_DIR = config['CROSSINGS_DIR']
    PCA_COMPONENTS_DIR = config['PCA_COMPONENTS_DIR']
    COMMANDS_FILE = config['COMMANDS_FILE']
    MODE = config['MODE']
    CUSTOM = config['CUSTOM']

# Create separate folders for each algorithm. 
# If in custom mode, each algorithm needs its own set of PCA components, because of the differences in pre-processing.
algorithms = list_of_algorithms(COMMANDS_FILE, remove_extension=True)
already_computed = False
prev_pca_components = None
for algorithm in algorithms:

    # Get the preprocessing parameters requested.
    algorithm_processing_params = preprocessing_parameters(algorithm, COMMANDS_FILE)

    # Fill in missing parameters with defaults from the config.
    for param, val in algorithm_processing_params.items():
        if val is None:
            algorithm_processing_params[param] = config[param]

    # If in 'custom' mode, then we always will recompute for each algorithm.
    if CUSTOM or not already_computed:
        pca_components = compute_pca_components(**algorithm_processing_params)
        already_computed = True
        prev_pca_components = pca_components
    else:
        pca_components = prev_pca_components

    # Check if the directory exists. Otherwise, make it.
    if not os.path.exists(PCA_COMPONENTS_DIR + '/' + algorithm):
        Path(PCA_COMPONENTS_DIR + '/' + algorithm).mkdir(parents=True, exist_ok=True)

    # Save all the components now.
    for n_components, components in pca_components.items():
        np.save(PCA_COMPONENTS_DIR + '/' + algorithm + '/pca%d_components.npy' % n_components, components)