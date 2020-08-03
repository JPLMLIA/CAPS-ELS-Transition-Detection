#!/usr/bin/env python

# External dependencies.
from __future__ import division
import numpy as np
from datetime import datetime as dt
from collections import defaultdict

# Internal dependencies.
from compute_file_list import list_of_ELS_files
from data_utils import get_ELS_data_no_interpolation

els_files = list_of_ELS_files('../temp/crossings_updated/data/', '../new_labels/all/', 'train')

missing = 0
total = 0
num_bins = defaultdict(lambda : 0)

for els_file in els_files:
    counts, energy_ranges, times = get_ELS_data_no_interpolation('../temp/crossings_updated/data/' + els_file, quantity='anode5', start_time=dt.min, end_time=dt.max)
    
    for timestep_counts, timestep_energy_range in zip(counts, energy_ranges):        
        valid_bins = ~np.isnan(timestep_energy_range)
        num_valid_bins = np.sum(valid_bins)
        corresponding_counts = timestep_counts[valid_bins]

        missing += np.sum(np.isnan(corresponding_counts))
        total += num_valid_bins

        num_bins[num_valid_bins] += 1

print 'Missing %d (%0.2f%%) out of total %d entries.' % (missing, missing/total * 100, total)
print num_bins
