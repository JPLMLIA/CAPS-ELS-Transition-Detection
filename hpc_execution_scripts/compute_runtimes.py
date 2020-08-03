#!/usr/bin/env python

# External dependencies.
import os
import numpy as np
import yaml
import h5py
import sys
from collections import defaultdict

# Load paths from config.
CONFIG_FILE = os.environ['CONFIG_FILE']
with open(CONFIG_FILE, 'r') as config_file_object:
    config = yaml.safe_load(config_file_object)
    RESULTS_DIR = config['RESULTS_DIR']

times = defaultdict(lambda: [])
folders = [RESULTS_DIR + dir for dir in os.listdir(RESULTS_DIR) if 'ELS' in dir]
for folder in folders:
    for file in os.listdir(folder):
        if file.endswith('.hdf5'):
            algorithm = os.path.splitext(file)[0]
            file_full_path = folder + '/' + file
            with h5py.File(file_full_path, 'r') as filedata:
                times[algorithm].append(filedata['time_taken'][()])

print 'Runtimes:'
for algorithm, algorithm_times in sorted(times.items()):
    print '- Algorithm %s: Min = %0.2f, Mean = %0.2f s, Max = %0.2f s, STD = %0.2f s.' % (algorithm, np.min(algorithm_times), np.mean(algorithm_times), np.max(algorithm_times), np.std(algorithm_times))