#!/usr/bin/env python

# External Dependencies.
from __future__ import division
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import auc
from pandas import DataFrame
import yaml
import os
import h5py
from logging import warning
from pathlib2 import Path

# Internal Dependencies.
from els_data import ELS
from data_utils import closest_index, datestring_to_float, format_dict, closest_distance, float_to_datestring, peaks
from plot_els import plot_raw_ELS_data, plot_interpolated_ELS_data
from stats_plotters import StatsPlotter

# Stores metrics for creating plots.
class Metrics:
    def __init__(self):
        self.tpr_list = []
        self.fpr_list = []
        self.precision_list = []
        self.recall_list = []
        self.f1_score_list = []
        self.accuracy_list = []
        self.thresholds = []
        self.positive_intervals_list = []
        self.negative_intervals_list = []
        self.predicted_positive_intervals_list = []
        self.predicted_negative_intervals_list = []
        self.confusion_matrices = []
        self.time_differences = {'predicted': [], 'labelled': [], 'false_positives': [], 'false_positives_rate': []}
        self.auprc_baseline = None
        self.auroc_baseline = 0.5
        self.computed_auprc = None
        self.computed_auprc_normalized = None

    # Compute the metrics we need.
    # True Positive Rate.
    def compute_tpr(self, confusion_matrix):
        true_positives, true_negatives, false_positives, false_negatives = confusion_matrix
        true_positive_rate = true_positives / (true_positives + false_negatives)
        self.tpr_list.append(true_positive_rate)

    # False Positive Rate.
    def compute_fpr(self, confusion_matrix):
        true_positives, true_negatives, false_positives, false_negatives = confusion_matrix
        false_positive_rate = false_positives / (false_positives + true_negatives)
        self.fpr_list.append(false_positive_rate)

    # Accuracy.
    def compute_accuracy(self, confusion_matrix):
        true_positives, true_negatives, false_positives, false_negatives = confusion_matrix
        accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)
        self.accuracy_list.append(accuracy)

    # Precision.
    def compute_precision(self, confusion_matrix):
        true_positives, true_negatives, false_positives, false_negatives = confusion_matrix
        if true_positives + false_positives > 0:
            precision = true_positives / (true_positives + false_positives)
        else:
            precision = np.nan
        self.precision_list.append(precision)

    # Recall.
    def compute_recall(self, confusion_matrix):
        true_positives, true_negatives, false_positives, false_negatives = confusion_matrix
        recall = true_positives / (true_positives + false_negatives)
        self.recall_list.append(recall)

    # F1 score.
    def compute_f1(self, confusion_matrix):
        true_positives, true_negatives, false_positives, false_negatives = confusion_matrix
        if true_positives + false_positives > 0:
            precision = true_positives / (true_positives + false_positives)
        else:
            precision = np.nan

        recall = true_positives / (true_positives + false_negatives)

        if precision + recall > 0:
            f1_score = 2 * precision * recall / (precision + recall)
        else:
            f1_score = np.nan

        self.f1_score_list.append(f1_score)

    # Baseline for Area under the Precision-Recall curve.
    def compute_auprc_baseline(self, confusion_matrix):
        true_positives, true_negatives, false_positives, false_negatives = confusion_matrix
        positive_proportion =  (true_positives + false_negatives) / (true_positives + false_negatives + true_negatives + false_positives)
        self.auprc_baseline = positive_proportion

    # Compute summaries such as AUROC and AUPRC.
    def compute_summaries(self):
        #self.computed_auroc = auc(self.fpr_list, self.tpr_list)
        self.computed_auprc = auc(self.recall_list, self.precision_list)
        self.computed_auprc_normalized = self.computed_auprc / (np.max(self.recall_list) - np.min(self.recall_list))
    
    # Print summaries of statistics.
    def print_summaries(self):
        if len(self.thresholds) == 1:
            print 'Threshold: %0.3f.' % (self.thresholds[0])
            print 'TPR estimated as %0.8f.' % self.tpr_list[0]
            print 'FPR estimated as %0.8f.' % self.fpr_list[0]
            print 'Accuracy estimated as %0.8f.' % self.accuracy_list[0]
            print 'Precision estimated as %0.8f.' % self.precision_list[0]
            print 'F1-score estimated as %0.8f.' % self.f1_score_list[0]
        else:
            print 'Threshold Range: %0.3f to %0.3f.' % (self.thresholds[0], self.thresholds[-1])
            print 'Computed AUROC = %0.3f over FPR range %0.3f to %0.3f.' % (self.computed_auroc, np.min(self.fpr_list), np.max(self.fpr_list))
            print 'Computed AUPRC = %0.3f over recall range %0.3f to %0.3f. AUPRC baseline is %0.4f.' % (self.computed_auprc, np.min(self.recall_list), np.max(self.recall_list), self.auprc_baseline)
            print 'Best F1-score = %0.3f.' % np.max(self.f1_score_list)


# Stores query parameters.
class QueryParameters:
    def __init__(self, start_time, end_time, anomaly_type):
        self.start_time = start_time
        self.end_time = end_time
        self.anomaly_type = anomaly_type


# Load anomalies from the labels file, within a time range.
# To generate labels, see generate_labels.py in this directory.
def load_anomalies(labels_file, params):

    # Unpack parameters.
    start_time = params.start_time
    end_time = params.end_time
    anomaly_type = params.anomaly_type

    # Check if labels file exists.
    if not os.path.exists(labels_file):
        raise OSError('Could not find labels file %s.' % labels_file)

    # Load anomalies from file.
    with open(labels_file, 'r') as labels_file_obj:
        labelled_anomalies = yaml.safe_load(labels_file_obj)[anomaly_type]

    # Convert to float (unit days).
    labelled_anomalies = datestring_to_float(labelled_anomalies)

    # Sort by start-time.
    labelled_anomalies = np.sort(labelled_anomalies)

    # Return change-points within the time span.
    return labelled_anomalies[np.logical_and(labelled_anomalies >= start_time, labelled_anomalies <= end_time)]


# Filter anomalies, to those with a score above a threshold.
def filter_anomalies(times, scores, threshold):
    return [time for time, score in zip(times, scores) if score >= threshold], [score for time, score in zip(times, scores) if score >= threshold], [index for index, score in enumerate(scores) if score >= threshold]


# Sweep through thresholds, filling up metrics.
def sweep_thresholds(times, scores, labelled_anomalies, time_tolerance_minutes, score_thresholds, params):

    # Create Metrics object to store stats.
    stats = Metrics()
    stats.thresholds = score_thresholds

    # Convert to units of the time array, ie, how many indices does this tolerance correspond to?
    days_per_array_unit = times[1] - times[0]
    minutes_per_array_unit = days_per_array_unit * 60 * 24
    minute_in_array_units = int(np.floor((1 / minutes_per_array_unit) + 0.5))
    time_tolerance_in_array_units = time_tolerance_minutes * minute_in_array_units

    # Mark regions around labelled anomalies, based on time tolerance.
    marked = defaultdict(set)
    for labelled_anomaly_number, labelled_anomaly in enumerate(labelled_anomalies):
        labelled_anomaly_time_index = closest_index(labelled_anomaly, times)

        window_start = labelled_anomaly_time_index - 1 * minute_in_array_units
        window_end = labelled_anomaly_time_index + time_tolerance_in_array_units
        for index in range(window_start, window_end):
            marked[index].add(labelled_anomaly_number)

    # Scan and filter!
    for index, score_threshold in enumerate(score_thresholds):

        # Filter anomalies to those with score above threshold.
        _, _, candidate_indices = filter_anomalies(times, scores, score_threshold)
        detection_indices = np.intersect1d(candidate_indices, peaks(scores, neighbourhood=time_tolerance_in_array_units//2))

        # Set containing all labelled anomalies, which have a detection close by.
        detected = set()

        # TN set to None as we don't actually fill it in, but we want to take advantage of the Metrics class methods.
        true_positives = 0
        false_positives = 0
        true_negatives = None
        false_negatives = 0

        # Evaluate predicted positive class, the detections.
        # False positives are all the detections with no labelled anomalies close by.
        for detection in detection_indices:
            if detection in marked:
                true_positives += 1
                for labelled_anomaly_number in marked[detection]:
                    detected.add(labelled_anomaly_number)
            else:
                false_positives += 1

        # False negatives are all the labelled anomalies with no detections close by.
        num_labelled = len(labelled_anomalies)
        num_labelled_and_detected = len(detected)
        num_labelled_and_not_detected = num_labelled - num_labelled_and_detected
        false_negatives = num_labelled_and_not_detected

        # Create a confusion matrix.
        confusion_matrix = (true_positives, true_negatives, false_positives, false_negatives)

        # Compute precision and recall.
        if true_positives + false_positives > 0:
           stats.compute_precision(confusion_matrix)
           stats.compute_recall(confusion_matrix)
           stats.compute_f1(confusion_matrix)
        stats.confusion_matrices.append(confusion_matrix)

    #stats.compute_summaries()
    return stats


def main(results_file, labels_file, show_plots, time_tolerances, thresholds_type, thresholds_folder, output_plots_directory):

    # Load algorithm results.
    with h5py.File(results_file, 'r') as filedata:
        scores = filedata['scores'][()]
        times = filedata['times'][()]
        anomaly_type = filedata['anomaly_type'][()]
        time_taken = filedata['time_taken'][()]
        start_time = filedata['start_time'][()]
        end_time = filedata['end_time'][()]
        data_file = filedata['data_file'][()]
        dataset = filedata['dataset'][()]
        quantity = filedata['quantity'][()]
        algorithm_name = filedata['algorithm_name'][()]
        parameters = filedata['parameters'][()]
        blur_sigma = filedata['blur_sigma'][()]
        bin_selection = filedata['bin_selection'][()]

    print('Start time: %s.' % float_to_datestring(start_time))
    print('End time: %s.' % float_to_datestring(end_time))

    # Parameters for the query.
    params = QueryParameters(start_time, end_time, anomaly_type)

    # Load labels!
    if labels_file is None:
        labels_file = os.path.splitext(data_file)[0] + '.yaml'

    labelled_anomalies = load_anomalies(labels_file, params)

    if len(labelled_anomalies) == 0:
       raise ValueError('No anomalies found in time-range supplied.')

    print '%d labelled anomalies loaded as type \'%s\'.' % (len(labelled_anomalies), anomaly_type)

    # Sort anomalies by time of occurrence.
    order = np.argsort(times)
    times = np.array(times)[order]
    scores = np.array(scores)[order]

    # Scan through range of score thresholds.
    if thresholds_type == 'preloaded':
        if thresholds_folder is None:
            raise ValueError('Please pass a directory for parameter \'thresholds_folder\' with the \'-tf\' option.')
        try:
            file_name = os.path.splitext(os.path.basename(results_file))[0]
            thresholds = np.load(thresholds_folder + file_name + '_thresholds.npy')
        except (IOError, OSError):
            file_name = file_name.split('_')[0]
            thresholds = np.load(thresholds_folder + file_name + '_thresholds.npy')
    elif thresholds_type == 'percentile':
        threshold_step = 1
        threshold_percentiles = np.arange(100 + threshold_step, step=threshold_step)
        thresholds = np.percentile(scores, threshold_percentiles)
    elif thresholds_type == 'linspace':
        thresholds = np.linspace(np.min(scores), np.max(scores), 100)
    
    epsilon = 1e-7
    thresholds[0] -= epsilon
    thresholds[-1] += epsilon

    # Check that monotonicity is maintained.
    for index, threshold in enumerate(thresholds[:-1]):
        next_threshold = thresholds[index + 1]
        if threshold > next_threshold:
            warning('Thresholds %s and %s violate monotonicity. Fixing...' % (threshold, next_threshold))
            thresholds[index + 1] = threshold
    
    # Sweep through time tolerances to compute metrics.
    stats = {}
    for time_tolerance_minutes in time_tolerances:
        
        print('Evaluating \'%s\' at time tolerance of %d minute(s)...' % (algorithm_name, time_tolerance_minutes))

        # Sweep through score thresholds.
        stats[time_tolerance_minutes] = sweep_thresholds(times, scores, labelled_anomalies, time_tolerance_minutes, thresholds, params)

    # Output confusion matrices.
    if output_plots_directory is not None:
        if not os.path.exists(output_plots_directory + '/confusion_matrices_time_tolerance/'):
            print 'Creating directory for output plots...'
            Path(output_plots_directory + '/confusion_matrices_time_tolerance/').mkdir(parents=True, exist_ok=True)

        # Different naming conventions: if we have many, append each time tolerance (in minutes) to file names.
        if len(time_tolerances) == 1:
            confusion_matrices_dict = {'%s.npz' % (os.path.splitext(os.path.basename(results_file))[0]): np.array(stats_for_time_tolerance.confusion_matrices) for time_tolerance_minutes, stats_for_time_tolerance in stats.items()}
        else:
            confusion_matrices_dict = {'%s_%d_min.npz' % (os.path.splitext(os.path.basename(results_file))[0], time_tolerance_minutes): np.array(stats_for_time_tolerance.confusion_matrices) for time_tolerance_minutes, stats_for_time_tolerance in stats.items()}

        for file_name, confusion_matrices in confusion_matrices_dict.items():
            np.savez(output_plots_directory + '/confusion_matrices_time_tolerance/' + file_name, confusion_matrices=confusion_matrices)

    # Show PR curve plot.
    if show_plots:

        # Create StatsPlotter object.
        plotter = StatsPlotter()

        # Create subplots - one for the time-series data, and the other for the scores.
        print 'Plotting scores and ELS data...'
        fig, axs = plt.subplots(nrows=2, sharex=True)

        # Plot the time-series data on top.
        plot_interpolated_ELS_data(fig, axs[0], data_file, quantity, start_time, end_time, blur_sigma=blur_sigma, bin_selection=bin_selection, colorbar_orientation='horizontal')
        
        # Plot intervals.
        title_params = {
            'algorithm_name': algorithm_name,
            'parameters': parameters,
        }
        plotter.plot_scores(fig, axs[1], times, scores, title_params=title_params)
        plt.setp(axs[1].get_xticklabels(), rotation=30, ha='right')
        plt.subplots_adjust(bottom=0.2)

        # Plot PR curve.
        print 'Plotting curves...'
        fig, ax = plt.subplots(1, 1)

        # Colormap for distinguishing multiple plots.
        cmap = plt.get_cmap('viridis')
        
        # Sweep through time thresholds, and plot.
        time_tolerance_max = np.max(time_tolerances)
        for time_tolerance, stats_for_time_tolerance in stats.items():
            plotter = StatsPlotter(stats_for_time_tolerance)
            plot_color = cmap(time_tolerance/time_tolerance_max)
            plotter.plot_pr(fig, ax, title_params=title_params, legend='%d Minute(s)' % time_tolerance, legend_title='Time Threshold', color=plot_color)

        plt.tight_layout()
        if output_plots_directory is None:
            plt.show()
        else:
            plt.savefig(output_plots_directory + 'pr.png', bbox_inches='tight')


if __name__ == '__main__':

    # Parser for command-line arguments.
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, add_help=True)
    parser.add_argument('results_file', 
                        help='Results file generated by find_scores.py.')
    parser.add_argument('-l', '--labels', dest='labels_file', default=None,
                        help='File containing the labels for the data.')
    parser.add_argument('-ttype', '--thresholds_type', dest='thresholds_type', default='preloaded', choices=['preloaded', 'percentile', 'linspace'],
                        help='How to choose thresholds for the scores.')
    parser.add_argument('-tf', '--thresholds_folder', dest='thresholds_folder', default=None,
                        help='File containing the thresholds (if \'thresholds_type\' is \'preloaded\') for the scores.')
    parser.add_argument('-t', '--tolerances', dest='time_tolerances', default=[20], nargs='+', type=int,
                        help='Time tolerance values (in minutes).')
    parser.add_argument('--no_plots', dest='show_plots', default=True, action='store_false', 
                        help='Do not show plots.')
    parser.add_argument('-opd', dest='output_plots_directory', default=None,
                        help='Directory to save plots in.')
    
    # Parse arguments and pass on to main().
    args = parser.parse_args()
    main(**vars(args))
