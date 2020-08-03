#!/usr/bin/env python

# This script creates a common configuration file for the HPC scripts.
# Author: Ameya Daigavane

# External dependencies.
import yaml
import argparse
import os
import sys

# Internal dependencies.
from compute_file_list import list_of_ELS_files
from compute_algorithm_params import list_of_algorithms

# Parse command-line arguments.
parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
parser.add_argument('config_file', default='common_config.yaml',
                    help='Name of config file to output to.')
parser.add_argument('-t', '--tolerance', dest='TOLERANCE', default=20,
                    help='Time tolerance value (in minutes).')
parser.add_argument('--lbls', dest='LABELS_SUBDIR', default='new_labels/valid/', choices=('labels/', 'new_labels/all/', 'new_labels/valid/', 'new_labels/mp/all/', 'new_labels/mp/in/', 'new_labels/mp/out/', 'new_labels/bs/all/', 'new_labels/bs/in/', 'new_labels/bs/out/', 'new_labels/dg/', 'new_labels/sc/'),
                    help='Subdirectory of CROSSINGS_DIR where labels would be found.')
parser.add_argument('--transform', default='anscombe_transform', dest='TRANSFORM', choices=('anscombe_transform', 'log_transform', 'no_transform'),
                    help='Transform to be applied to data.')
parser.add_argument('--mode', default='test', dest='MODE', choices=('test', 'train', 'train-v2', 'test-v2'),
                    help='Subset of files to use: test set (after 2004) or the training set (2004 only).')
parser.add_argument('--results', dest='RESULTS_PARENTS_DIR_RELATIVE', default='crossings_test_results/',
                    help='Directory relative to $HOME where results will be placed.')
parser.add_argument('--commands', dest='COMMANDS_FILE', default=None,
                    help='Commands file, relative to SECONDARY_SCRIPTS_DIR. If unset, MODE will choose defaults.')
parser.add_argument('-b', '--blur_sigma', dest='BLUR_SIGMA', type=int, default=0,
                    help='Parameter sigma of the Gaussian blur applied to ELS data.')
parser.add_argument('--bin_selection', dest='BIN_SELECTION', choices=('all', 'center', 'ignore_unpaired'), default='all',
                    help='Selection of ELS bins.')
parser.add_argument('-f', '--filter', dest='FILTER', choices=('min_filter', 'median_filter', 'max_filter', 'no_filter'), default='no_filter',
                    help='Filter to pass ELS data through, after the Gaussian blur.')
parser.add_argument('-fsize', '--filter_size', dest='FILTER_SIZE', type=int, default=1,
                    help='Size of filter to pass ELS data through, after the Gaussian blur.')
parser.add_argument('--custom', dest='CUSTOM', action='store_true', default=False,
                    help='Custom mode for operation.')
args = vars(parser.parse_args())

# Set parameters.
TIME_TOLERANCE = args['TOLERANCE']
LABELS_SUBDIR = args['LABELS_SUBDIR']
TRANSFORM = args['TRANSFORM']
BLUR_SIGMA = args['BLUR_SIGMA']
BIN_SELECTION = args['BIN_SELECTION']
FILTER = args['FILTER']
FILTER_SIZE = args['FILTER_SIZE']
MODE = args['MODE']
COMMANDS_FILE = args['COMMANDS_FILE']
CUSTOM = args['CUSTOM']

# Set paths.
CREATION_COMMAND = ' '.join(sys.argv).replace('\n', ' ')
SCRIPTS_DIR = '/scratch_lg/image-content/ameyasd/europa-onboard-science/src/caps_els/'
SECONDARY_SCRIPTS_DIR = SCRIPTS_DIR + 'secondary_scripts/'
CROSSINGS_DIR = '/scratch_lg/image-content/ameyasd/crossings_updated/'
DATA_DIR = CROSSINGS_DIR + 'data/'
LABELS_DIR = CROSSINGS_DIR + LABELS_SUBDIR
RESULTS_PARENT_DIR = os.getenv('HOME', '/scratch_lg/image-content/ameyasd') + '/' + args['RESULTS_PARENTS_DIR_RELATIVE'] + '/'

if CUSTOM:
    RESULTS_DIR = RESULTS_PARENT_DIR + 'custom/'
else:
    RESULTS_DIR = RESULTS_PARENT_DIR + MODE + '/' + TRANSFORM + '/' + 'blur_' + str(BLUR_SIGMA) + '/' + BIN_SELECTION + '/' + FILTER + '_' + str(FILTER_SIZE) + '/'

RANDOM_PLOTS_DIR = RESULTS_DIR + 'random_plots/' + LABELS_SUBDIR
ERROR_ANALYSIS_DIR = RESULTS_DIR + 'error_analysis/' + LABELS_SUBDIR
COMBINED_MATRICES_DIR = RESULTS_DIR + str(TIME_TOLERANCE) + 'min/' + LABELS_SUBDIR
COMBINED_MATRICES_FILE_BASENAME = COMBINED_MATRICES_DIR + 'combined_matrices_%s.npz'
PCA_COMPONENTS_DIR = RESULTS_DIR + 'pca_components/'
THRESHOLDS_DIR = RESULTS_DIR + 'thresholds/' + LABELS_SUBDIR

# Set the commands file.
if COMMANDS_FILE is None:
    if MODE == 'train' or MODE == 'train-v2':
        COMMANDS_FILE = SECONDARY_SCRIPTS_DIR + 'commands/parameter_sweep_commands'
    elif MODE == 'test' or MODE == 'test-v2':
        COMMANDS_FILE = SECONDARY_SCRIPTS_DIR + 'commands/parameter_sweep_commands_testset'
else:
    COMMANDS_FILE = SECONDARY_SCRIPTS_DIR + COMMANDS_FILE

# How many algorithms in the commands file?
ALGORITHMS = list_of_algorithms(COMMANDS_FILE)
NUM_ALGORITHMS = len(ALGORITHMS)

# How many files are we working on?
FILES = list_of_ELS_files(DATA_DIR, LABELS_DIR, MODE)
NUM_DATAFILES = len(FILES)

# How big should the job arrays be, atleast?
MINIMUM_JOB_ARRAY_SIZE = NUM_ALGORITHMS * NUM_DATAFILES

# Package into a dictionary.
contents = {
    'TIME_TOLERANCE': TIME_TOLERANCE,
    'LABELS_SUBDIR': LABELS_SUBDIR,
    'TRANSFORM': TRANSFORM,
    'BLUR_SIGMA': BLUR_SIGMA,
    'BIN_SELECTION': BIN_SELECTION,
    'FILTER': FILTER,
    'FILTER_SIZE': FILTER_SIZE,
    'MODE': MODE,
    'COMMANDS_FILE': COMMANDS_FILE,
    'CREATION_COMMAND': CREATION_COMMAND,
    'SCRIPTS_DIR': SCRIPTS_DIR,
    'CROSSINGS_DIR': CROSSINGS_DIR,
    'DATA_DIR': DATA_DIR,
    'LABELS_DIR': LABELS_DIR,
    'RESULTS_DIR': RESULTS_DIR,
    'RANDOM_PLOTS_DIR': RANDOM_PLOTS_DIR,
    'ERROR_ANALYSIS_DIR': ERROR_ANALYSIS_DIR,
    'COMBINED_MATRICES_DIR': COMBINED_MATRICES_DIR,
    'COMBINED_MATRICES_FILE_BASENAME': COMBINED_MATRICES_FILE_BASENAME,
    'PCA_COMPONENTS_DIR': PCA_COMPONENTS_DIR,
    'THRESHOLDS_DIR': THRESHOLDS_DIR,
    'NUM_ALGORITHMS': NUM_ALGORITHMS,
    'NUM_DATAFILES': NUM_DATAFILES,
    'MINIMUM_JOB_ARRAY_SIZE': MINIMUM_JOB_ARRAY_SIZE,
    'CUSTOM': CUSTOM,
}

# Check if the specified config file exists. In this case, abort.
if os.path.exists(args['config_file']):
    raise ValueError('The config file already exists.')

# Write configuration to the config file.
with open(args['config_file'], 'w') as config_file_object:
    config = yaml.dump(contents, config_file_object)
