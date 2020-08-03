#!/usr/bin/env python
#PBS -q array
#PBS -l select=1:ncpus=1
#PBS -l walltime=1:00:00
#PBS -j oe
#PBS -o /scratch_lg/image-content/ameyasd/logs/combined
#PBS -M ameya.s.daigavane@jpl.nasa.gov
#PBS -m abe

# This script computes the results from the combined confusion matrices.
# Author: Ameya Daigavane

# External dependencies.
import numpy as np
import os
import sys
import re
import os
from collections import defaultdict
import yaml
from pathlib2 import Path

# Internal dependencies.
sys.path.append('/scratch_lg/image-content/ameyasd/europa-onboard-science/src/caps_els/') # Hack to import correctly.
from evaluate_methods_time_tolerance import Metrics

# Maps algorithms to their formatted names.
def get_class(algorithm):
    """
    >>> get_class('rulsifn10k2') = 'RuLSIF'
    >>> get_class('stickyhmmn3sticky1pca5') = 'Sticky HDP-HMM'
    """

    # Assuming we keep the standard of unique file prefixes.
    prefix = algorithm[:2]

    # Maps file name prefixes to algorithm names.
    algorithm_classes_expanded = {
        'ba': 'L2-Diff',
        'ho': 'HOT SAX',
        'mp': 'Matrix Profile',
        'ru': 'RuLSIF',
        'st': 'Sticky HDP-HMM',
        'vh': 'Vanilla HMM',
    }

    return algorithm_classes_expanded[prefix]


# Load paths from config.
CONFIG_FILE = os.environ['CONFIG_FILE']
with open(CONFIG_FILE, 'r') as config_file_object:
    config = yaml.safe_load(config_file_object)
    TIME_TOLERANCE = config['TIME_TOLERANCE']
    COMBINED_MATRICES_DIR = config['COMBINED_MATRICES_DIR']
    COMBINED_MATRICES_FILE_BASENAME = config['COMBINED_MATRICES_FILE_BASENAME']
    MODE = config['MODE']

# Iterate over all years.
for year in ['all'] + range(2004, 2013):

    # For each algorithm, use the combined matrices to compute precision-recall.
    try:
        combined_matrices = np.load(COMBINED_MATRICES_FILE_BASENAME % str(year))
    except IOError:
        continue

    # Maintain the best parameters, based on maximum recall at some fixed precision values.
    precision_thresholds = [0.5, 0.8]
    best_params = defaultdict(lambda: defaultdict())
    best_recall = defaultdict(lambda: defaultdict(lambda: 0))
    algorithm_classes = set()

    for algorithm in sorted(combined_matrices):
        stats = Metrics()

        # Sweep through all confusion matrices.
        for confusion_matrix in combined_matrices[algorithm]:
            (true_positives, true_negatives, false_positives, false_negatives) = confusion_matrix
            if true_positives + false_positives > 0:
                stats.compute_precision(confusion_matrix)
                stats.compute_recall(confusion_matrix)
                stats.compute_f1(confusion_matrix)

        # Save the class of this algorithm.
        algorithm_class = get_class(algorithm)
        algorithm_classes.add(algorithm_class)

        # Update best parameters, if valid.
        for precision, recall in zip(stats.precision_list, stats.recall_list):
            for precision_threshold in precision_thresholds:
                if precision >= precision_threshold and recall > best_recall[algorithm_class][precision_threshold]:
                    best_recall[algorithm_class][precision_threshold] = recall
                    best_params[algorithm_class][precision_threshold] = algorithm

        # Compute AUPRC and package.
        stats.compute_summaries()
        algorithm_results = {
            'precision': stats.precision_list,
            'recall': stats.recall_list,
            'auprc': stats.computed_auprc,
        }

        # Create directory for these results.
        yearwise_dir = COMBINED_MATRICES_DIR + str(year) + '/'
        if not os.path.exists(yearwise_dir):
            Path(yearwise_dir).mkdir(parents=True, exist_ok=True)

        # Save to file.
        np.savez(yearwise_dir + '%s_pr.npz' % algorithm, **algorithm_results)

    # Print to console.
    print 'Config file: %s' % CONFIG_FILE
    print 'Year %s:' % str(year)
    for precision_threshold in precision_thresholds:
        print '\t - Precision Threshold %0.2f' % precision_threshold
        for algorithm_class in sorted(algorithm_classes):
            try:
                print '\t \t -- %s: Maximum recall = %0.3f for %s.' % (algorithm_class, best_recall[algorithm_class][precision_threshold], best_params[algorithm_class][precision_threshold])
            except KeyError:
                print '\t \t -- %s did not reach this precision threshold.' % (algorithm_class)
    print