#!/usr/bin/env python

# This script computes how many files we are working on, for the config file.
# Author: Ameya Daigavane

# External dependencies.
import os
import sys
import yaml
import numpy as np
from datetime import datetime
from itertools import chain

# Internal dependencies.
sys.path.append('/scratch_lg/image-content/ameyasd/europa-onboard-science/src/caps_els/') # Hack to import correctly.
sys.path.append('../') # Hack to import correctly.
from data_utils import cache, get_ELS_data, datestring_to_float

def filter_excluding_years(file_list, years):
    """
    >>> file_list = ['ELS_200418018_V01', 'ELS_200418118_V01', 'ELS_200518018_V01', 'ELS_200618018_V01', 'ELS_200818018_V01']
    >>> filter_excluding_years(file_list, [])
    ['ELS_200418018_V01', 'ELS_200418118_V01', 'ELS_200518018_V01', 'ELS_200618018_V01', 'ELS_200818018_V01']
    >>> filter_excluding_years(file_list, [2004])
    ['ELS_200518018_V01', 'ELS_200618018_V01', 'ELS_200818018_V01']
    >>> filter_excluding_years(file_list, [2004, 2005])
    ['ELS_200618018_V01', 'ELS_200818018_V01']
    >>> filter_excluding_years(file_list, [2004, 2008])
    ['ELS_200518018_V01', 'ELS_200618018_V01']
    >>> filter_excluding_years(file_list, [2007])
    ['ELS_200418018_V01', 'ELS_200418118_V01', 'ELS_200518018_V01', 'ELS_200618018_V01', 'ELS_200818018_V01']
    """

    def check_excluding_years(years):
        def check(file_name):
            return np.all(['ELS_%d' % year not in file_name for year in years])
        return check

    check = check_excluding_years(years)
    return [file_name for file_name in file_list if check(file_name)]


def filter_including_years(file_list, years):
    """
    >>> file_list = ['ELS_200418018_V01', 'ELS_200418118_V01', 'ELS_200518018_V01', 'ELS_200618018_V01', 'ELS_200818018_V01']
    >>> filter_including_years(file_list, [])
    []
    >>> filter_including_years(file_list, [2004])
    ['ELS_200418018_V01', 'ELS_200418118_V01']
    >>> filter_including_years(file_list, [2004, 2005])
    ['ELS_200418018_V01', 'ELS_200418118_V01', 'ELS_200518018_V01']
    >>> filter_including_years(file_list, [2004, 2008])
    ['ELS_200418018_V01', 'ELS_200418118_V01', 'ELS_200818018_V01']
    >>> filter_including_years(file_list, [2007])
    []
    """

    def check_including_years(years):
        def check(file_name):
            return np.any(['ELS_%d' % year in file_name for year in years])
        return check

    check = check_including_years(years)
    return [file_name for file_name in file_list if check(file_name)]

# Checks if the labels file exists, and if it does, check if we have ELS data for atleast one label.
def check_labels(els_file_name, labels_file_name):

    # Check if the labels file exists.
    if not os.path.exists(labels_file_name):
        return False

    # Load labels.
    with open(labels_file_name, 'r') as labels_file_object:
        labels = yaml.safe_load(labels_file_object)
        crossings = labels['change_points']

    # Convert to float (unit days).
    crossing_floats = datestring_to_float(crossings)

    # Check if atleast one label is valid.
    atleast_one_valid_label = False
    for crossing_float in crossing_floats:
        try:
            times = get_ELS_data(els_file_name, quantity='anode5', start_time=datetime.min, end_time=datetime.max)[2]
            if times[0] <= crossing_float <= times[-1]:
                atleast_one_valid_label = True
                break

        except ValueError:
            pass

    return atleast_one_valid_label


@cache('temp/cachedir/')
def list_of_ELS_files(DATA_DIR, LABELS_DIR, MODE, remove_extension=False):
    """
    Returns a list of ELS files from DATA_DIR corresponding to MODE, whose labels are present in LABELS_DIR.

    If remove_extension is False, returns a list such as ['ELS_200418018_V01.DAT', 'ELS_200500318_V01.DAT'].
    If remove_extension is True, returns a list such as ['ELS_200418018_V01', 'ELS_200500318_V01'].
    """

    # How many files do we have in this MODE?
    all_dat_files = sorted([file_name for file_name in os.listdir(DATA_DIR) if 'DAT' in file_name])
    if MODE == 'train':
        files_for_mode = filter_including_years(all_dat_files, [2004])
    elif MODE == 'test':
        files_for_mode = filter_excluding_years(all_dat_files, [2004])
    elif MODE == 'train-v2':
        files_for_mode = filter_including_years(all_dat_files, [2011])
    elif MODE == 'test-v2':
        files_for_mode = filter_including_years(all_dat_files, [2012])

    # We only count files which have the labels we want.
    labels_files_for_mode = [os.path.splitext(file_name)[0] + '.yaml' for file_name in files_for_mode]
    files_with_labels = [file_name for file_name, labels_file_name in zip(files_for_mode, labels_files_for_mode) if check_labels(DATA_DIR + file_name, LABELS_DIR + labels_file_name)]

    # Remove the .DAT extension?
    if remove_extension:
        files_with_labels = [os.path.splitext(file_name)[0] for file_name in files_with_labels]

    return files_with_labels
