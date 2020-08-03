#!/usr/bin/env python

# This script computes the algorithm names (and preprocessing parameters) as given in the commands file, specified in the config.
# Author: Ameya Daigavane

import os
import re
import sys

# Internal dependencies.
sys.path.append('/scratch_lg/image-content/ameyasd/europa-onboard-science/src/caps_els/') # Hack to import correctly.
from data_utils import cache

@cache('temp/cachedir/')
def list_of_algorithms(COMMANDS_FILE, remove_extension=False):
    """
    Returns a list of algorithms from COMMANDS_FILE.

    If remove_extension is False, returns a list such as ['rulsifn10k2.hdf5', 'stickyhdphmmn2.hdf5'].
    If remove_extension is True, returns a list such as ['rulsifn10k2', 'stickyhdphmmn2'].
    """

    # Read the commands file.
    with open(COMMANDS_FILE, 'r') as commands_file:
        commands_file_contents = commands_file.read()
    
    # Regular expression to match.
    pattern = re.compile('\\w*.hdf5')
    algorithms = [substr.group() for substr in pattern.finditer(commands_file_contents)]

    # Remove the .hdf5 extension?
    if remove_extension:
        algorithms = [os.path.splitext(algorithm)[0] for algorithm in algorithms]

    return algorithms


def preprocessing_parameters(algorithm, COMMANDS_FILE):
    """
    Returns a dictionary of preprocessing parameters for the given algorithm in the commands file.
    """

    # Parameters (with options as in the commands file).
    params_with_options = {
        'BLUR_SIGMA': ['--blur_sigma'],
        'BIN_SELECTION': ['--bin_selection'],
        'FILTER': ['--filter'],
        'FILTER_SIZE': ['--filter_size'],
        'TRANSFORM': ['--transform'],
    }
        
    # Extracts out parameters from a single line.
    def get_params(line):
        line_split = line.split()       
        def get_param(param, param_options):
            for index, token in enumerate(line_split):
                if token in param_options:
                    param_value = line_split[index + 1]
                    if '${' in param_value:
                        return None
                    try:
                        param_value = int(param_value)
                    except ValueError:
                        pass
                    return param_value
                
        return {param: get_param(param, param_options) for param, param_options in params_with_options.items()}

    # Read the commands file.
    with open(COMMANDS_FILE, 'r') as commands_file:
        commands_file_lines = commands_file.readlines()
    
    # Iterate over all lines, until match with algorithm.
    for line in commands_file_lines:
        if algorithm in line:
            return get_params(line)