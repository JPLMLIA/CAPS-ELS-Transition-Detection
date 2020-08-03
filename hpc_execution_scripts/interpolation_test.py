#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import os
import yaml
import sys
from datetime import datetime
from collections import defaultdict
import sys

# Internal dependencies.
sys.path.append('../')
from data_utils import get_ELS_data
from plot_els import plot_interpolated_ELS_data, plot_raw_ELS_data

def set_interpolation(type_1, type_2):
    os.environ['STEP_1_INTERPOLATION_TYPE'] = type_1
    os.environ['STEP_2_INTERPOLATION_TYPE'] = type_2


def log_invalids(els_data_file):
    type_1 = os.environ['STEP_1_INTERPOLATION_TYPE']
    type_2 = os.environ['STEP_2_INTERPOLATION_TYPE']
    counts = get_ELS_data(els_data_file, 'anode5', datetime.min, datetime.max)[0]
    invalid_counts[type_1 + ' + ' + type_2] += np.sum(counts < 0)
    total_counts[type_1 + ' + ' + type_2] += counts.size


# Load paths from config.
with open('common_config.yaml', 'r') as config_file_object:
    config = yaml.safe_load(config_file_object)
    DATA_DIR = config['DATA_DIR']

invalid_counts = defaultdict(lambda: 0)
total_counts = defaultdict(lambda: 0)
for data_file in os.listdir(DATA_DIR):
    file_name, ext = os.path.splitext(data_file)
    if ext == '.DAT':
        fig, axs = plt.subplots(ncols=5, figsize=(25, 5))
        print(data_file)
        plot_raw_ELS_data(fig, axs[0], DATA_DIR + data_file, 'anode5')
        axs[0].set_title('Raw')
        set_interpolation('slinear', 'slinear')
        plot_interpolated_ELS_data(fig, axs[1], DATA_DIR + data_file, 'anode5')
        axs[1].set_title('Interpolated\nLinear along Energy, Linear along Time')
        log_invalids(DATA_DIR + data_file)
        set_interpolation('slinear', 'cubic')
        plot_interpolated_ELS_data(fig, axs[2], DATA_DIR + data_file, 'anode5')
        axs[2].set_title('Interpolated\nLinear along Energy, Cubic along Time')
        log_invalids(DATA_DIR + data_file)
        set_interpolation('cubic', 'slinear')
        plot_interpolated_ELS_data(fig, axs[3], DATA_DIR + data_file, 'anode5')
        axs[3].set_title('Interpolated\nCubic along Energy, Linear along Time')
        log_invalids(DATA_DIR + data_file)
        set_interpolation('cubic', 'cubic')
        plot_interpolated_ELS_data(fig, axs[4], DATA_DIR + data_file, 'anode5')
        axs[4].set_title('Interpolated\nCubic along Energy, Cubic along Time')
        log_invalids(DATA_DIR + data_file)
        plt.suptitle('%s' % file_name)
        plt.tight_layout(rect=[0.03, 0, 0.97, 0.95], w_pad=3)
        plt.savefig('../temp/interpolation/new/%s.png' % file_name)
        plt.close()
        
        print(invalid_counts)
        print(total_counts)
    
