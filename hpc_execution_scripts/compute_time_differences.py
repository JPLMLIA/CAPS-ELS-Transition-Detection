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
OUTPUT_DIR = '/halo_nobackup/image-content/ameyasd/optimization/'
file_list = set([os.path.splitext(file_name)[0] for file_name in os.listdir(DATA_DIR) if '2004' in file_name])

time_diffs = {}
max_time_diff_ratio = []
median_time_diff_ratio = []
mean_time_diff_ratio = []

for file_no_ext in file_list:
    file_full = DATA_DIR + file_no_ext + '.DAT' 
    counts, energy_range, times = get_ELS_data(file_full, 'anode5', datetime.min, datetime.max)

    time_differences = np.diff(times)
    min_time_diff = np.min(time_differences)
    max_time_diff = np.max(time_differences)
    median_time_diff = np.median(time_differences)
    mean_time_diff = np.mean(time_differences)  

    time_diffs[file_no_ext] = [min_time_diff, max_time_diff, median_time_diff, mean_time_diff]
    max_time_diff_ratio.append(max_time_diff/min_time_diff)
    mean_time_diff_ratio.append(mean_time_diff/min_time_diff)
    median_time_diff_ratio.append(median_time_diff/min_time_diff)    


with open(OUTPUT_DIR + 'time_differences.txt', 'w') as f:
    f.write('File MinTD MaxTD MedianTD MeanTD \n')
    for file_no_ext in time_diffs:
        min_time_diff, median_time_diff, mean_time_diff, max_time_diff = time_diffs[file_no_ext]
        f.write('%s %0.10f %0.10f %0.10f %0.10f \n' % (file_no_ext, min_time_diff, max_time_diff, median_time_diff, mean_time_diff))

fig, axs = plt.subplots(nrows=3)
axs[0].hist(max_time_diff_ratio, bins=20, range=(1, 2))
axs[0].set_title('Max Time Difference / Min Time Difference')
axs[1].hist(median_time_diff_ratio, bins=20, range=(1, 2))
axs[1].set_title('Median Time Difference / Min Time Difference')
axs[2].hist(mean_time_diff_ratio, bins=20, range=(1, 2))
axs[2].set_title('Mean Time Difference / Min Time Difference')
plt.suptitle('Histograms')
plt.tight_layout()
plt.subplots_adjust(top=0.85)
plt.savefig(OUTPUT_DIR + 'time_differences_ratio.png')

