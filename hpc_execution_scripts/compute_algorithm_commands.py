#!/usr/bin/env python

import os
import yaml
import sys
from string import Template

# Get algorithm name as input.
try:
    algorithm = sys.argv[1]
except IndexError:
    raise ValueError('Please enter an algorithm')

# Load paths from config.
CONFIG_FILE = os.environ['CONFIG_FILE']
with open(CONFIG_FILE, 'r') as config_file_object:
    config = yaml.safe_load(config_file_object)
    COMMANDS_FILE = config['COMMANDS_FILE']
    TRANSFORM = config['TRANSFORM']
    BLUR_SIGMA = config['BLUR_SIGMA']
    FILTER = config['FILTER']
    FILTER_SIZE = config['FILTER_SIZE']
    BIN_SELECTION = config['BIN_SELECTION']

# Package all these as a dictionary.
params_dict = {
    'TRANSFORM': TRANSFORM,
    'BLUR_SIGMA': BLUR_SIGMA,
    'FILTER': FILTER,
    'FILTER_SIZE': FILTER_SIZE,
    'BIN_SELECTION': BIN_SELECTION,
}

# Read the contents of COMMANDS_FILE.
with open(COMMANDS_FILE, 'r') as f:
    lines = f.readlines()

# Search for matching line.
for line in lines:
    if line[:2] == '${' and algorithm in line:
        # Remove newline at the end, and convert to a template string.
        line = line.rstrip()
        line = Template(line)

        # Fill in each line with the missing paths and parameters.
        filled_line = line.safe_substitute(**params_dict)
        print filled_line
        break


