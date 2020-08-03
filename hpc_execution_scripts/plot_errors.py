#!/usr/bin/env python

# External dependencies.
from __future__ import division
import os
import sys
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
from datetime import timedelta
import yaml

# Internal dependencies.
sys.path.append('/scratch_lg/image-content/ameyasd/europa-onboard-science/src/caps_els/') # Hack to import correctly.
sys.path.append('/scratch_lg/image-content/ameyasd/europa-onboard-science/src/caps_els/secondary_scripts') # Hack to import correctly.
from plot_els import plot_interpolated_ELS_data
from data_utils import datestring_to_float, get_ELS_file_name, convert_to_dt
from compute_algorithm_params import list_of_algorithms
from compute_labelled_events import list_of_events


# Returns the colour to use for the crossings.
def crossing_color(event_type):
    if 'BS' in event_type:
        return 'black'
    elif 'MP' in event_type:
        return 'red'
    else:
        return 'gray'


# Plot all examples on one combined plot.
def plot_all(algorithms, title, suffix, savefile):
    fig, axs = plt.subplots(nrows=len(algorithms), sharex=True, figsize=(10, 18))
    colors = plt.rcParams['axes.prop_cycle']()

    for algorithm, ax in zip(algorithms, axs):

        # Load crossings with scores.
        all_crossings_file = ERROR_ANALYSIS_DIR + algorithm + suffix
        all_crossings = np.load(all_crossings_file)

        all_crossings_scores = np.array(all_crossings[:, 0], dtype=float)
        all_crossings_times = datestring_to_float(all_crossings[:, 1])

        # Create scatterplot.
        ax.scatter(all_crossings_times, all_crossings_scores, label=algorithm, s=10, alpha=0.8, **next(colors))

        # Load scores summary.
        scores_summary_file = ERROR_ANALYSIS_DIR + algorithm + '_scores_summary.npy'
        scores_min, scores_mean, scores_max = np.load(scores_summary_file)

        # Horizontal lines indicating min, mean and max of 'actual' scores.
        ax.axhline(y=scores_min, linestyle='-', c='gray', alpha=0.5)
        ax.axhline(y=scores_mean, linestyle='--', c='black')
        ax.axhline(y=scores_max, linestyle='-', c='gray', alpha=0.5)

    # Set x-axis tick range.
    start = datestring_to_float('01-01-2004/00:00:00')
    end = datestring_to_float('01-01-2005/00:00:00')
    ax.set_xlim(start, end)

    # Set x-axis formatting of dates.
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y/%H:%M'))
    ax.xaxis.set_tick_params(labelsize=8)

    # Tilts dates to the left for easier reading.
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')

    # Set title.
    fig.suptitle(title % LABELS_SUBDIR, y=0.92, fontweight='bold')

    # Set x-label.
    ax.set_xlabel('Datetime')
    fig.text(x=0.02, y=0.5, s='Scores')

    # Show legend, common across all subplots.
    labels_handles = {
        label: handle for ax in fig.axes for handle, label in zip(*ax.get_legend_handles_labels())
    }

    fig.legend(
        labels_handles.values(),
        labels_handles.keys(),
        loc='center right',
        title='$\\bf{Algorithm}$',
        fancybox=True,
        shadow=True,
    )

    # Fix dimensions.
    fig.subplots_adjust(left=0.15, bottom=0.15, right=0.7)

    # Save to file.
    fig.savefig(ERROR_ANALYSIS_DIR + savefile, dpi=fig.dpi, bbox_inches='tight')


def plot_all_false_negatives(algorithms):
    return plot_all(algorithms, title='Algorithm Scores on \'%s\' Crossings',
                                     suffix='_all_false_negatives.npy',
                                     savefile='all_crossings.png')

def plot_all_false_positives(algorithms):
    return plot_all(algorithms, title='Algorithm False Detections on \'%s\' Crossings',
                                     suffix='_all_false_positives.npy',
                                     savefile='all_false_detections.png')


# Plot samples for each algorithm separately.
def plot_worst(algorithms, title, suffix, savefile, num_samples=5):

    for algorithm in algorithms:

        # Each algorithm gets its own figure.
        fig, axs = plt.subplots(ncols=num_samples, figsize=(25, 5))
        
        # Load crossings with scores.
        worst_detections_file = ERROR_ANALYSIS_DIR + algorithm + suffix
        worst_detections = np.load(worst_detections_file)[:num_samples]

        # Parse each column.
        worst_detections_scores = np.array(worst_detections[:, 0], dtype=float)
        worst_detections_times = worst_detections[:, 1]

        # Plot each of the worst crossings.
        for detection_time, detection_score, ax in zip(worst_detections_times, worst_detections_scores, axs):
            
            # Base ELS file name (without the .DAT extension) for this detection.
            els_basename = get_ELS_file_name(detection_time, remove_extension=True)
            
            # The time of the detection as a datetime object, and the size of the window used for plotting.
            detection_time_dt = convert_to_dt(detection_time) 
            time_diff = timedelta(minutes=TIME_TOLERANCE//2)
            
            # Plot ELS data first.
            els_data_file = DATA_DIR + els_basename + '.DAT'
            plot_interpolated_ELS_data(fig, ax, els_data_file, 
                                        start_time=detection_time_dt - time_diff,
                                        end_time=detection_time_dt + time_diff,
                                        colorbar_orientation='horizontal', quantity='anode5',
                                        blur_sigma=BLUR_SIGMA, bin_selection=BIN_SELECTION, filter=FILTER, filter_size=FILTER_SIZE)
            
            # Obtain the list of events occurring in this file.
            labels = list_of_events(els_basename, CROSSINGS_DIR)

            # How large is the width of the rectangle around each labelled event?
            days_per_minute = 1/(24 * 60)
            window_size = 1*days_per_minute

            # Annotate plot with labelled events.
            for label_type, crossing_timestring in labels:
                if detection_time_dt - time_diff <= convert_to_dt(crossing_timestring) <= detection_time_dt + time_diff:
                    crossing_time = datestring_to_float(crossing_timestring)
                    ax.axvspan(crossing_time - window_size/2, crossing_time + window_size/2, facecolor=crossing_color(label_type), alpha=1)

            # Set title as the score.
            ax.set_title('Score %0.2f' % detection_score, pad=55)

        # Set title.
        fig.suptitle(title % (LABELS_SUBDIR, algorithm), x=0.45, y=0.92, fontweight='bold')

        # Fix dimensions.
        fig.subplots_adjust(left=0.15, bottom=0.15, right=0.7, top=0.65, wspace=0.4)

        # Save to file.
        fig.savefig(ERROR_ANALYSIS_DIR + savefile % algorithm, dpi=fig.dpi, bbox_inches='tight')


def plot_worst_false_negatives(algorithms):
    plot_worst(algorithms, title='\'%s\' Crossings \n Worst False Negatives for Algorithm \'%s\'',
                           suffix='_worst_false_negatives.npy',
                           savefile='%s_worst_crossings.png')

def plot_worst_false_positives(algorithms):
    plot_worst(algorithms, title='\'%s\' Crossings \n Worst False Positives for Algorithm \'%s\'',
                           suffix='_worst_false_positives.npy',
                           savefile='%s_worst_false_detections.png')


# Load paths from config.
CONFIG_FILE = os.environ['CONFIG_FILE']
with open(CONFIG_FILE, 'r') as config_file_object:
    config = yaml.safe_load(config_file_object)
    MODE = config['MODE']
    BLUR_SIGMA = config['BLUR_SIGMA']
    BIN_SELECTION = config['BIN_SELECTION']
    FILTER = config['FILTER']
    FILTER_SIZE = config['FILTER_SIZE']
    DATA_DIR = config['DATA_DIR']
    LABELS_DIR = config['LABELS_DIR']
    LABELS_SUBDIR = config['LABELS_SUBDIR']
    RESULTS_DIR = config['RESULTS_DIR']
    RANDOM_PLOTS_DIR = config['RANDOM_PLOTS_DIR']
    ERROR_ANALYSIS_DIR = config['ERROR_ANALYSIS_DIR']
    COMMANDS_FILE = config['COMMANDS_FILE']
    CROSSINGS_DIR = config['CROSSINGS_DIR']
    TIME_TOLERANCE = config['TIME_TOLERANCE']

# The algorithms that we're testing.
algorithms = list_of_algorithms(COMMANDS_FILE, remove_extension=True)

# plot_all_false_negatives(algorithms)
plot_worst_false_negatives(algorithms)
# plot_all_false_positives(algorithms)
plot_worst_false_positives(algorithms)
