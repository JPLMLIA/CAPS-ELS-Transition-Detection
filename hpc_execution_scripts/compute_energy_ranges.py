#!/usr/bin/env python

# External dependencies.
from __future__ import division
import os
import numpy as np
import sys
from datetime import datetime
import matplotlib.pyplot as plt
plt.switch_backend('agg')

# Internal dependencies.
sys.path.append('/halo_nobackup/image-content/ameyasd/europa-onboard-science/src/caps_els/') # Hack to import correctly.
from data_utils import get_ELS_data

DATA_DIR = '/halo_nobackup/image-content/ameyasd/crossings_updated/data/'
OUTPUT_DIR = '/halo_nobackup/image-content/ameyasd/optimization/training_set_statistics/'
file_list = set([os.path.splitext(file_name)[0] for file_name in os.listdir(DATA_DIR) if '2004' in file_name])

energy_ranges = {}

for file_no_ext in file_list:
    file_full = DATA_DIR + file_no_ext + '.DAT' 
    counts, file_energy_range, times = get_ELS_data(file_full, 'anode5', datetime.min, datetime.max)

    min_energy_range = np.min(file_energy_range[0])
    max_energy_range = np.max(file_energy_range[0])
   
    energy_ranges[file_no_ext] = [min_energy_range, max_energy_range]   

with open(OUTPUT_DIR + 'energy_ranges.txt', 'w') as f:
    f.write('File MinER MaxER \n')
    for file_no_ext, file_stats in energy_ranges.items():
        min_energy_range, max_energy_range = file_stats
        f.write('%s %0.10f %0.10f \n' % (file_no_ext, min_energy_range, max_energy_range))

for index, file_no_ext in enumerate(energy_ranges):
    plt.plot([index, index], energy_ranges[file_no_ext], lw=10)
plt.yscale('log')
plt.title('Energy Ranges')
plt.savefig(OUTPUT_DIR + 'energy_ranges.png')

