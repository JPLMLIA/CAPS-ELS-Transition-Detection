#!/usr/bin/env python
#PBS -q array
#PBS -l select=1:ncpus=1
#PBS -l walltime=1:00:00
#PBS -j oe
#PBS -o /scratch_lg/image-content/ameyasd/logs/combined
#PBS -M ameya.s.daigavane@jpl.nasa.gov
#PBS -m abe

# This script analyzes errors (false positives/false negatives) for individual algorithms on the crossings dataset.
# Author: Ameya Daigavane

# External dependencies.
from __future__ import division
import os
import yaml
import sys
from pathlib2 import Path
import h5py
import numpy as np

# Internal dependencies.
sys.path.append('/scratch_lg/image-content/ameyasd/europa-onboard-science/src/caps_els/') # Hack to import correctly.
sys.path.append('/scratch_lg/image-content/ameyasd/europa-onboard-science/src/caps_els/secondary_scripts/') # Hack to import correctly.
from compute_file_list import list_of_ELS_files
from compute_algorithm_params import list_of_algorithms
from data_utils import closest_index, datestring_to_float, float_to_datestring, peaks, closest_distance


# Assigns a score to each crossing as the maximum score in a window of one minute before.
def compute_crossing_score(crossing, scores, times):

    # How many minutes does one time-step correspond to?
    timesteps_per_minute = int(round(1/(60 * 24 * (times[1] - times[0]))))

    # What is the closest index in 'times' that matches up to the crossing event?
    crossing_index = closest_index(crossing, times)

    # Construct the window around the crossing. This matches up exactly with how the evaluation framework does it.
    start_index = max(crossing_index - 1 * timesteps_per_minute, 0)
    end_index = crossing_index + TIME_TOLERANCE * timesteps_per_minute

    # Return the maximum in this window.
    return np.max(scores[start_index: end_index])


# Return detections not close to any peak.
def get_false_detection_indices(crossing_times, scores, times):

    # How many minutes does one time-step correspond to?
    timesteps_per_minute = int(round(1/(60 * 24 * (times[1] - times[0]))))

    # Indices corresponding to peaks of scores (ie, detections).
    peak_indices = peaks(scores, neighbourhood=(TIME_TOLERANCE * timesteps_per_minute)//2)

    # Times for these indices.
    peak_times = times[peak_indices]

    # For each of these detections, what is the closest peak?
    closest_distance_to_label = closest_distance(peak_times, crossing_times)

    # Only keep those detections more than 2 * TIME_THRESHOLD minutes before/after an actual peak.
    days_per_minute = 1/(60 * 24)
    return [peak_index for peak_index, dist in zip(peak_indices, closest_distance_to_label) if dist > 2 * TIME_TOLERANCE * days_per_minute]


# Load paths from config.
CONFIG_FILE = os.environ['CONFIG_FILE']
with open(CONFIG_FILE, 'r') as config_file_object:
    config = yaml.safe_load(config_file_object)
    SCRIPTS_DIR = config['SCRIPTS_DIR']
    DATA_DIR = config['DATA_DIR']
    LABELS_SUBDIR = config['LABELS_SUBDIR']
    LABELS_DIR = config['LABELS_DIR']
    MODE = config['MODE']
    RESULTS_DIR = config['RESULTS_DIR']
    TIME_TOLERANCE = config['TIME_TOLERANCE']
    THRESHOLDS_DIR = config['THRESHOLDS_DIR']
    COMMANDS_FILE = config['COMMANDS_FILE']
    NUM_ALGORITHMS = config['NUM_ALGORITHMS']
    ERROR_ANALYSIS_DIR = config['ERROR_ANALYSIS_DIR']

# The algorithms that we're testing.
algorithms = list_of_algorithms(COMMANDS_FILE, remove_extension=True)

# If we're in 'test' mode, print this message.
if MODE == 'test' or MODE == 'test-random':
    print 'Ignoring MODE = \'test\' in config, as analysis is only supported over training data.'

# The data files and labels we're testing each algorithm over.
files = list_of_ELS_files(DATA_DIR, LABELS_DIR, 'train', remove_extension=True)
folders = [RESULTS_DIR + dir for dir in files]
labels_files = [LABELS_DIR + dir + '.yaml' for dir in files]

# Iterate over all algorithms.
for algorithm in algorithms:

    # False negatives: crossings that are not detected well.
    all_crossings = []

    # False positives: detections predicted but not close to any actual crossing.
    all_false_detections = []

    # Initialize.
    num_timesteps = 0
    scores_sum = 0
    scores_min = 1e9
    scores_max = -1e9

    # Iterate over each folder, filling up the two lists above.
    for folder, labels_file in zip(folders, labels_files):

        # Load the labels for this file.
        with open(labels_file, 'r') as labels_file_object:
            crossing_datestrings = yaml.safe_load(labels_file_object)['change_points']

        # Convert to float (units as days).
        crossing_times = datestring_to_float(crossing_datestrings)

        # Load the scores for this file.
        try:
            file_full_path = folder + '/' + algorithm + '.hdf5'
            with h5py.File(file_full_path, 'r') as filedata:
                scores = filedata['scores'][()]
                times = filedata['times'][()]
        except IOError:
            raise IOError('File %s cannot be found. Have you run these algorithms on the training set?' % (file_full_path))

        # Update scores stats.
        num_timesteps += len(scores)
        scores_sum += np.sum(scores)
        scores_min = min(scores_min, np.min(scores))
        scores_max = max(scores_max, np.max(scores))

        # First, false negatives.
        # Compute scores for these crossing events.
        crossing_scores = [compute_crossing_score(crossing, scores, times) for crossing in crossing_times]

        # Add these crossings to the first list.
        all_crossings.extend(zip(crossing_scores, crossing_datestrings))

        # Now, false positives.
        false_detection_indices = get_false_detection_indices(crossing_times, scores, times)
        false_detection_times = times[false_detection_indices]
        false_detection_scores = scores[false_detection_indices]
        false_detection_datestrings = float_to_datestring(false_detection_times)

        # Add these false detections to the second list.
        all_false_detections.extend(zip(false_detection_scores, false_detection_datestrings))

    # Compute a summary of this algorithms' scores.
    scores_mean = scores_sum / num_timesteps
    scores_summary = np.array([scores_min, scores_mean, scores_max])

    # Rank crossings by scores.
    all_crossings.sort()

    # Rank false detections by scores, in descending order.
    all_false_detections.sort(reverse=True)

    # Take the top 10 false negatives and false positives.
    worst_crossings = all_crossings[:10]
    worst_false_detections = all_false_detections[:10]

    # Print stats for this algorithm.
    print 'Algorithm: %s' % algorithm
    print 'Crossings detected the worst (false negatives):'
    print ''.join(['\t- %s with score %0.2f.\n' % (crossing_datestring, crossing_score) for crossing_score, crossing_datestring in worst_crossings])
    print 'Detections falsely predicted the worst (false positives):'
    print ''.join(['\t- %s with score %0.2f.\n' % (detection_datestring, detection_score) for detection_score, detection_datestring in worst_false_detections])
    print 

    # Create the directory if it doesn't exist.
    if not os.path.exists(ERROR_ANALYSIS_DIR):
        Path(ERROR_ANALYSIS_DIR).mkdir(parents=True, exist_ok=True)

    # Save all arrays to disk.
    np.save(ERROR_ANALYSIS_DIR + algorithm + '_scores_summary.npy', scores_summary)
    np.save(ERROR_ANALYSIS_DIR + algorithm + '_all_false_negatives.npy', all_crossings)
    np.save(ERROR_ANALYSIS_DIR + algorithm + '_all_false_positives.npy', all_false_detections)
    np.save(ERROR_ANALYSIS_DIR + algorithm + '_worst_false_negatives.npy', worst_crossings)
    np.save(ERROR_ANALYSIS_DIR + algorithm + '_worst_false_positives.npy', worst_false_detections)
