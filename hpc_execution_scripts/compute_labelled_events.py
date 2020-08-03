#!/usr/bin/env python

# This script computes the events of given types occurring within a given ELS file.
# Author: Ameya Daigavane

import os
import sys
import yaml

# Internal dependencies.
sys.path.append('/scratch_lg/image-content/ameyasd/europa-onboard-science/src/caps_els/') # Hack to import correctly.
from data_utils import cache, convert_to_dt


@cache('temp/cachedir/')
def list_of_events(els_basename, CROSSINGS_DIR, label_categories=None):
    """
    Returns a list of (event_type, event_time) for the given ELS file.
    Default label categories are 'BS-ALL' and 'MP-ALL'.
    If a event belongs to multiple categories, it is repeated for each.

    >>> list_of_events('ELS_200418018_V01', '/scratch_lg/image-content/ameyasd/crossings_updated/')
    [
     ('MP-ALL', '28-06-2004/18:43:00'),
     ('MP-ALL', '28-06-2004/18:49:00'),
     ('MP-ALL', '28-06-2004/19:53:00'),
     ('MP-ALL', '28-06-2004/20:23:00'),
     ('MP-ALL', '28-06-2004/20:27:00'),
     ('MP-ALL', '28-06-2004/20:44:00'),
     ('MP-ALL', '28-06-2004/21:09:00'),
     ('MP-ALL', '28-06-2004/21:32:00'),
     ('MP-ALL', '28-06-2004/21:56:00'),
     ('MP-ALL', '28-06-2004/23:07:00')
    ]

    >>> list_of_events('ELS_200418018_V01', '/scratch_lg/image-content/ameyasd/crossings_updated/', label_categories=['MP-IN', 'MP-OUT'])
    [
     ('MP-IN', '28-06-2004/18:43:00'),
     ('MP-OUT', '28-06-2004/18:49:00'),
     ('MP-IN', '28-06-2004/19:53:00'),
     ('MP-OUT', '28-06-2004/20:23:00'),
     ('MP-IN', '28-06-2004/20:27:00'),
     ('MP-OUT', '28-06-2004/20:44:00'),
     ('MP-IN', '28-06-2004/21:09:00'),
     ('MP-OUT', '28-06-2004/21:32:00'),
     ('MP-IN', '28-06-2004/21:56:00'),
     ('MP-OUT', '28-06-2004/23:07:00'),
    ]
    """

    # Default options.
    if label_categories is None:
        label_categories = ['BS-ALL', 'MP-ALL']
    
    # Set of all valid categories.
    valid_categories_set = set(['BS-ALL', 'BS-IN', 'BS-OUT', 'MP-ALL', 'MP-IN', 'MP-OUT', 'DG', 'SC'])
    def valid_categories(label_categories):
        for category in label_categories:
            if category not in valid_categories_set:
                print category
                return False
        return True

    # Check if we have been passed valid categories.
    if not valid_categories(label_categories):
        raise ValueError('Categories must be in %s only.' % valid_categories_set)

    # Map each label category to a directory. This is effectively hard-coded, based on how the labels are organized.
    def category_to_dir(category):
        return 'new_labels/' + '/'.join(category.lower().split('-')) + '/'
    label_dirs = [category_to_dir(category) for category in label_categories]

    # Read each labels file.
    event_list = []
    for category, dir in zip(label_categories, label_dirs):
        try:
            with open(CROSSINGS_DIR + dir + els_basename + '.yaml', 'r') as labels_file_object:
                labels = yaml.safe_load(labels_file_object)
                crossing_times = labels['change_points']

            event_list.extend([(category, crossing_time) for crossing_time in crossing_times])
        except IOError:
            pass
    
    # Sort by increasing order of event time.
    event_list.sort(key=lambda (type, time): convert_to_dt(time))

    return event_list