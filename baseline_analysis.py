#!/usr/bin/env python
"""
Applies the baseline (L2-Diff) algorithm to a time-series.
"""

# External dependencies
from __future__ import division
from datetime import datetime
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Internal dependencies
from plot_els import plot_interpolated_ELS_data
from data_utils import get_ELS_data
from transform import Transformation

# Function to be imported from other modules.
def baseline_analysis(input_data, transform='no_transform'):
    counts, energy_ranges, times = input_data

    # Transform counts if required.
    counts = Transformation(transform).transform(counts)

    # Scores are defined as the L2-distance between counts at this time-step,
    # and counts at the previous timestep.
    scores = np.zeros_like(times)
    scores[1:] = np.linalg.norm(counts[1:] - counts[:-1], axis=1)

    return times, scores


def main(els_data_file, quantity, start_time, end_time, output_file, **kwargs):

    # Check input arguments - start and end times should be valid.
    if start_time is not None:
        try:
            start_time = datetime.strptime(start_time, '%d-%m-%Y/%H:%M')
        except ValueError:
            raise
    else:
        start_time = datetime.min

    if end_time is not None:
        try:
            end_time = datetime.strptime(end_time, '%d-%m-%Y/%H:%M').replace(second=59, microsecond=999999)
        except ValueError:
            raise
    else:
        end_time = datetime.max

    # Get data.
    data = get_ELS_data(els_data_file, quantity, start_time, end_time, **kwargs)

    # Compute scores.
    times, scores = baseline_analysis(data)

    # Plot ELS data.
    print 'Plotting...'
    fig, axs = plt.subplots(nrows=2, sharex=True)
    plot_interpolated_ELS_data(fig, axs[0], els_data_file, quantity,
                               start_time, end_time,
                               colorbar_range='subset',
                               colorbar_orientation='horizontal', **kwargs)
    axs[0].set_xlabel('')

    # Plot scores.
    axs[1].plot(times, scores)
    axs[1].set_xlabel('Date/Time')
    axs[1].set_ylabel('Change-point Score')
    axs[1].xaxis.set_tick_params(labelsize=8)
    axs[1].margins(0, 0)
    plt.setp(axs[1].get_xticklabels(), rotation=30, ha='right')

    # Place title below.
    if kwargs['filter'] is None:
        title_subtext = 'Blur %d' % (kwargs['blur_sigma'])
    else:
        title_subtext = 'Blur %d, Filter %s of Size %d' % (kwargs['blur_sigma'], kwargs['filter'], kwargs['filter_size'])
    fig.text(s='Change-point Scores for ELS Data \n %s' % title_subtext, x=0.5, y=0.03,
             horizontalalignment='center', fontsize=13)

    plt.subplots_adjust(bottom=0.3, left=0.2)

    # Save plot.
    if output_file is None:
        plt.show()
    else:
        plt.savefig(output_file, bbox_inches='tight')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)

    # Define command-line arguments.
    parser.add_argument('els_data_file',
                        help='ELS DAT file.')
    parser.add_argument('-st', '--start_time', default=None,
                        help='Start time in dd-mm-yyyy/HH:MM. Restricts data to those recorded on or after this time.')
    parser.add_argument('-et', '--end_time', default=None,
                        help='End time in dd-mm-yyyy/HH:MM. Restricts data to those recorded upto and including this time.')
    parser.add_argument('-q', '--quantity', default='anode5', choices=(
        ['iso'] +
        [('anode%d' % a) for a in range(1, 9)] +
        [('def%d'   % a) for a in range(1, 9)] +
        [('dnf%d'   % a) for a in range(1, 9)] +
        [('psd%d'   % a) for a in range(1, 9)]
    ))
    parser.add_argument('-b', '--blur_sigma', type=int, default=0,
                        help='Parameter sigma of the Gaussian blur applied to ELS data.')
    parser.add_argument('--bin_selection', choices=('all', 'center', 'ignore_unpaired'), default='all',
                        help='Selection of ELS bins.')
    parser.add_argument('-f', '--filter', choices=('min_filter', 'median_filter', 'max_filter', 'no_filter'), default='no_filter',
                        help='Filter to pass ELS data through, after the Gaussian blur.')
    parser.add_argument('-fsize', '--filter_size', type=int, default=1,
                        help='Size of filter to pass ELS data through, after the Gaussian blur.')
    parser.add_argument('-o', '--output_file', default=None,
                        help='File to store output plot.')

    # Parse command-line arguments.
    args = parser.parse_args()

    # Analyze.
    main(**vars(args))
