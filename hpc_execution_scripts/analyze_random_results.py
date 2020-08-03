#!/usr/bin/env python
#PBS -q array
#PBS -l select=1:ncpus=1
#PBS -l walltime=1:00:00
#PBS -j oe
#PBS -o /scratch_lg/image-content/ameyasd/logs/combined
#PBS -M ameya.s.daigavane@jpl.nasa.gov
#PBS -m abe

# This script analyzes randomly picked files and the scores assigned by the algorithms on each.
# Author: Ameya Daigavane

# External dependencies.
from __future__ import division
import os
import yaml
from pathlib2 import Path
import random
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import h5py
import sys

# Seed for reproducibility.
random.seed(0)

# Internal dependencies.
sys.path.append('/scratch_lg/image-content/ameyasd/europa-onboard-science/src/caps_els/') # Hack to import correctly.
sys.path.append('/scratch_lg/image-content/ameyasd/europa-onboard-science/src/caps_els/secondary_scripts/') # Hack to import correctly.
from plot_els import plot_interpolated_ELS_data
from data_utils import datestring_to_float
from stats_plotters import StatsPlotter
from compute_file_list import list_of_ELS_files


# Gets the first file in a directory with this substring in its name.
def get_first_file(directory, substr=''):
    for results_file in sorted(os.listdir(directory)):
        if results_file[:len(substr)] == substr:
            return results_file


# Returns the colour to use for the crossings.
def crossing_color(LABELS_SUBDIR):
    if 'bs' in LABELS_SUBDIR:
        return 'black'
    elif 'mp' in LABELS_SUBDIR:
        return 'red'
    else:
        return 'gray'


# Load paths from config.
CONFIG_FILE = os.environ['CONFIG_FILE']
with open(CONFIG_FILE, 'r') as config_file_object:
    config = yaml.safe_load(config_file_object)
    MODE = config['MODE']
    BLUR_SIGMA = config['BLUR_SIGMA']
    BIN_SELECTION = config['BIN_SELECTION']
    FILTER = config['FILTER']
    FILTER_SIZE = config['FILTER_SIZE']
    DATA_DIR = config['DATA_DIR']
    LABELS_DIR = config['LABELS_DIR']
    LABELS_SUBDIR = config['LABELS_SUBDIR']
    RESULTS_DIR = config['RESULTS_DIR']
    RANDOM_PLOTS_DIR = config['RANDOM_PLOTS_DIR']

# Maps file name prefixes to algorithm names.
algorithm_classes_expanded = {
    'ba': 'L2-Diff',
    'ho': 'HOT SAX',
    'mp': 'Matrix Profile',
    'ru': 'RuLSIF',
    'st': 'Sticky HDP-HMM',
    'vh': 'Vanilla HMM',
}

# Pick 'num_samples' files randomly from each year.
num_samples = 10

# ELS files.
els_dirs = [os.path.splitext(data_file)[0] for data_file in sorted(list_of_ELS_files(DATA_DIR, LABELS_DIR, MODE))]
try:
    els_dirs_sampled = random.sample(els_dirs, num_samples)
except ValueError:
    els_dirs_sampled = els_dirs

# Create plots for each sampled ELS file.
for els_dir in els_dirs_sampled:

    # Define paths to the actual data and labels file.
    els_dir_full_path = RESULTS_DIR + els_dir + '/'
    els_data_file = DATA_DIR + els_dir + '.DAT'
    els_labels_file = LABELS_DIR + els_dir + '.yaml'

    # Extract out files for each algorithm.
    # Either get the first file in each directory for each algorithm.
    # algorithm_files = [get_first_file(els_dir_full_path, substr=prefix) for prefix in algorithm_classes_expanded]
    # or, use a predefined set:
    algorithm_files = [
        'baseline.hdf5',
        'hotsaxw50n2pca10.hdf5',
        'hotsaxw50n5pca10.hdf5',
        'mpw100n08pca10.hdf5',
        'mpw100n01pca10.hdf5',
        'rulsifn10k2.hdf5',
        'rulsifn10k5.hdf5',
        'stickyhmmn2sticky01pca10.hdf5',
        'stickyhmmn2sticky10.hdf5',
        'stickyhmmn3sticky1pca5.hdf5',
        'vhmmn2pca10.hdf5',
        'vhmmn3pca5.hdf5',
    ]

    # Subplots. Top-most is for the ELS data marked with crossings. Every other subplot is for an algorithms' scores.
    fig, axs = plt.subplots(nrows=len(algorithm_files) + 1, figsize=(10, 40), sharex=True)

    # Plot ELS data.
    plot_interpolated_ELS_data(fig, axs[0], els_data_file, 'anode5', colorbar_orientation='horizontal', blur_sigma=BLUR_SIGMA, bin_selection=BIN_SELECTION, filter=FILTER, filter_size=FILTER_SIZE)

    # Load labels.
    with open(els_labels_file, 'r') as labels_file_object:
        labels = yaml.safe_load(labels_file_object)
        crossings = labels['change_points']

    # Mark each crossing in the first plot with a window of the appropriate color.
    color = crossing_color(LABELS_SUBDIR)
    days_per_minute = 1/(24 * 60)
    window_size = 5*days_per_minute
    for crossing in crossings:
        crossing_time = datestring_to_float(crossing)
        axs[0].axvspan(crossing_time - window_size/2, crossing_time + window_size/2, facecolor=color, alpha=1)

    # Fill the remaining subplots with scores from each algorithm.
    plotter = StatsPlotter()
    for index, algorithm_file in enumerate(algorithm_files, start=1):
        with h5py.File(els_dir_full_path + algorithm_file, 'r') as filedata:
            scores = filedata['scores'][()]
            times = filedata['times'][()]
        plotter.plot_scores(fig, axs[index], times, scores)
        axs[index].set_xlabel(algorithm_file)

    # Create plots directory, if it doesn't exist.
    if not os.path.exists(RANDOM_PLOTS_DIR):
        Path(RANDOM_PLOTS_DIR).mkdir(parents=True, exist_ok=True)
    
    # Save figure.
    fig.suptitle(els_dir, y=0.90)
    fig.subplots_adjust(hspace=0.3)
    fig.savefig(RANDOM_PLOTS_DIR + els_dir + '.png', dpi=fig.dpi, bbox_inches='tight')


