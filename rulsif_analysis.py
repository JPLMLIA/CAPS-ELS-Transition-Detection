#!/usr/bin/env python
"""
Applies RuLSIF to a time-series.

Author: Ameya Daigavane
"""

# External dependencies
from __future__ import division
import os
import argparse
from datetime import datetime
from densratio import densratio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Internal dependencies
from plot_els import plot_raw_ELS_data
from data_utils import get_top_counts, get_ELS_data, pack, get_median_pairwise_distance
from transform import Transformation

# Function to be imported from other modules.
# This function is not called inside this script at all,
# and contains significant code overlap with main().
def rulsif_analysis(input_data, k, n, perform_hyperparameter_estimation, alpha,
                    anomaly_type='change_points', transform='no_transform'):

    # Unpack input.
    counts, energy_range, times = input_data

    # Transform counts if required.
    counts = Transformation(transform).transform(counts)

    # Pack sequence into blocks.
    counts_packed = pack(counts, k)

    # Median distance between subsequences.
    dmed = get_median_pairwise_distance(counts_packed)

    # Range of values the hyperparameters were supposed to take, according to the reference.
    sigma_range = np.array([
        0.6 * dmed,
        0.8 * dmed,
        1.0 * dmed,
        1.2 * dmed,
        1.4 * dmed
    ])
    sigma_forward_range = sigma_backward_range = sigma_range
    lambda_range = np.array([1e-3, 1e-2, 1e-1, 1e0, 1e1])
    lambda_forward_range = lambda_backward_range = lambda_range

    # Restrict range further by taking the most common hyperparameters,
    # selected by fitting random samples.
    if perform_hyperparameter_estimation:
        sigma_forward_range, sigma_backward_range, lambda_forward_range, lambda_backward_range = \
                estimate_hyperparameters(counts_packed, window_size=n,
                                         sigma_range=sigma_range,
                                         lambda_range=lambda_range,
                                         alpha=alpha, num_rank=2)

    # Change-point scores.
    packed_sequence_size = counts_packed.shape[0]
    original_sequence_size = counts.shape[0]
    scores = np.zeros(original_sequence_size)

    # Sliding-window over packed sequence.
    for i in range(n, packed_sequence_size - n + 1):
        forward_window = counts_packed[i: i + n]
        backward_window = counts_packed[i - n: i]
        forward_density_obj = densratio(backward_window, forward_window,
                                        alpha=alpha,
                                        sigma_range=sigma_forward_range,
                                        lambda_range=lambda_forward_range,
                                        verbose=False)
        forward_divergence = forward_density_obj.alpha_PE
        backward_density_obj = densratio(forward_window, backward_window,
                                         alpha=alpha,
                                         sigma_range=sigma_backward_range,
                                         lambda_range=lambda_backward_range,
                                         verbose=False)
        backward_divergence = backward_density_obj.alpha_PE
        change_point_score = forward_divergence + backward_divergence

        # Use larger range of hyperparameters if we can't get a good fit with the smaller one.
        if change_point_score < 0:
            sigma_range = np.array([
                0.7 * dmed,
                0.8 * dmed,
                0.9 * dmed,
                dmed,
                1.1 * dmed,
                1.2 * dmed,
                1.3 * dmed
            ])

            forward_density_obj = densratio(backward_window, forward_window,
                                            alpha=alpha,
                                            sigma_range=sigma_range,
                                            verbose=False)
            forward_divergence = forward_density_obj.alpha_PE
            backward_density_obj = densratio(forward_window, backward_window,
                                             alpha=alpha,
                                             sigma_range=sigma_range,
                                             verbose=False)
            backward_divergence = backward_density_obj.alpha_PE

            change_point_score = forward_divergence + backward_divergence

        # Shift score ahead because of packing.
        scores[i + k//2] = change_point_score

    # Cut off scores at 0, no negative values.
    scores[scores < 0] = 0

    # Return a list of times and scores, for change-points.
    # Convert these to intervals for other anomalies.
    if anomaly_type == 'change_points':
        return times, scores

    elif anomaly_type in ['bimodality', 'negative_ions']:
        intervals = []
        interval_scores = []

        # Compute maximum position and indices.
        max_index = np.argmax(scores)
        max_score = np.max(scores)

        while max_score > 0:

            # Idea: Full-Width at Quarter-Maximum
            # Go towards the left.
            for start_index, score in reversed(list(enumerate(scores[:max_index]))):
                if score < max_score/4:
                    start_index += 1
                    break

            # Now towards the right.
            for end_index, score in enumerate(scores[max_index:], start=max_index):
                if score < max_score/4:
                    break

            # Add this as an interval.
            # The interval's score as the mean score of all points within.
            if start_index != end_index:
                intervals.append((times[start_index], times[end_index]))
                interval_scores.append(np.sum(scores[start_index: end_index]))

                # Mask these indices.
                scores[start_index: end_index] = -np.inf

            # Compute maximum position and indices.
            max_index = np.argmax(scores)
            max_score = np.max(scores)

        # Aggregating zero-scored timesteps into intervals.
        start_index = 0
        while start_index < len(scores):
            if scores[start_index] == 0:
                for end_index, score in enumerate(scores[start_index:], start=start_index):
                    if score == -np.inf:
                        break

                intervals.append((times[start_index], times[end_index]))
                interval_scores.append(np.mean(scores[start_index: end_index]))
                start_index = end_index

            start_index += 1

        return intervals, interval_scores


# Estimate hyperparameters by random sampling, and finding the most likely hyperparameters.
def estimate_hyperparameters(packed_sequence, window_size=50, alpha=0.1, sigma_range=None, lambda_range=None, num_samples=50, num_rank=2):

    sigmas_forward = np.zeros(num_samples)
    sigmas_backward = np.zeros(num_samples)
    lambdas_forward = np.zeros(num_samples)
    lambdas_backward = np.zeros(num_samples)

    print 'Sampling for hyperparameter estimation:...'

    for iteration in range(num_samples):
        i = np.random.randint(low=window_size,
                              high=packed_sequence.shape[0] - window_size)

        backward_window = packed_sequence[i - window_size: i]
        forward_window = packed_sequence[i: i + window_size]

        ratio_forward_obj = densratio(backward_window, forward_window,
                                      alpha=alpha,
                                      sigma_range=sigma_range,
                                      lambda_range=lambda_range,
                                      verbose=False)
        ratio_backward_obj = densratio(forward_window, backward_window,
                                       alpha=alpha,
                                       sigma_range=sigma_range,
                                       lambda_range=lambda_range,
                                       verbose=False)

        sigmas_forward[iteration] = ratio_forward_obj.kernel_info.sigma
        lambdas_forward[iteration] = ratio_forward_obj.lambda_

        sigmas_backward[iteration] = ratio_backward_obj.kernel_info.sigma
        lambdas_backward[iteration] = ratio_backward_obj.lambda_

        print 'Iteration', iteration, 'complete.'

    print 'Sampling for hyperparameter estimation complete.'

    return get_top_counts(sigmas_forward, num_rank), \
           get_top_counts(sigmas_backward, num_rank), \
           get_top_counts(lambdas_forward, num_rank), \
           get_top_counts(lambdas_backward, num_rank)


def main(els_data_file, output_file, perform_hyperparameter_estimation,
         load_from_file, save_to_file, quantity,
         start_time, end_time, run_tests, plot_processed_sequence, k, n):

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

    # Run doctests.
    if run_tests:
        import doctest
        import data_utils
        doctest.testmod(data_utils, verbose=True,
                        optionflags=doctest.NORMALIZE_WHITESPACE)
        doctest.testmod(verbose=True,
                        optionflags=doctest.NORMALIZE_WHITESPACE)

    # RuLSIF parameter.
    alpha = 0.1

    # Set random seed for reproducibility.
    random_seed = 7
    np.random.seed(random_seed)

    # Load processed sequence, if file found.
    els_sequence_file = os.path.splitext(els_data_file)[0] + '_RuLSIF_sequence'
    if load_from_file and os.path.exists(els_sequence_file + '.npz'):
        print 'Loading processed sequence from sequence file...'
        filedata = np.load(els_sequence_file + '.npz')
        counts_packed = filedata['counts_packed']
        energy_range = filedata['energy_range']
        times = filedata['times']
        dmed = filedata['dmed']
    else:
        print 'Sequence file not found. Extracting data from original ELS file and processing...'
        counts, energy_range, times = get_ELS_data(els_data_file, quantity, start_time, end_time)

        # import pdb; pdb.set_trace() # For debugging.

        # Process counts.
        # counts = gaussian_blur(counts, sigma=0.5)
        # counts = np.ma.log(counts)

        # See the sequence plotted (lineplot for 1D data, colourplot for 2D data).
        if plot_processed_sequence:
            print 'Plotting processed sequence...'
            fig, ax = plt.subplots(1, 1)
            ax.set_title('Processed Sequence')

            if len(counts.shape) == 1:
                ax.xaxis_date()
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y/%H:%M'))
                fig.autofmt_xdate()
                ax.plot(times, counts)

            elif len(counts.shape) == 2:
                plt.imshow(counts.T, origin='lower', interpolation='none')
                ax.set_aspect('auto')
                plt.colorbar(ax=ax, orientation='vertical')

            plt.show()

        # Pack sequence into blocks.
        print 'Packing sequence into blocks...'
        counts_packed = pack(counts, k)
        print 'Sequence packed into shape %s.' % (counts_packed.shape,)

        # Median distance between subsequences.
        print 'Computing median distance between packed samples...'
        dmed = get_median_pairwise_distance(counts_packed)
        print 'Median distance between packed samples, dmed =', dmed

        # Save values to file.
        if save_to_file:
            arrays_with_names = {'counts_packed': counts_packed,
                                 'energy_range': energy_range,
                                 'times': times,
                                 'dmed': np.array(dmed)}
            np.savez(els_sequence_file, **arrays_with_names)

    # Range of values the hyperparameters were supposed to take, according to the reference.
    sigma_range = np.array([dmed])
    sigma_forward_range = sigma_backward_range = sigma_range
    lambda_range = np.array([1e-3, 1e-2, 1e-1, 1e0, 1e1])
    lambda_forward_range = lambda_backward_range = lambda_range

    # Restrict range further by taking the most common hyperparameters selected for fitting random samples.
    if perform_hyperparameter_estimation:
        els_hyperparameters_file = os.path.splitext(els_data_file)[0] + '_RuLSIF_hyperparameters'
        if load_from_file and os.path.exists(els_hyperparameters_file + '.npz'):
            print 'Hyperparameters file found. Loading from file...'
            filedata = np.load(els_hyperparameters_file + '.npz')
            sigma_forward_range = filedata['sigma_forward_range']
            sigma_backward_range = filedata['sigma_backward_range']
            lambda_forward_range = filedata['lambda_forward_range']
            lambda_backward_range = filedata['lambda_backward_range']
        else:
            print 'Hyperparameters file not found. Performing estimation...'
            sigma_forward_range, sigma_backward_range, \
            lambda_forward_range, lambda_backward_range = \
                estimate_hyperparameters(counts_packed, window_size=n,
                                         sigma_range=sigma_range,
                                         lambda_range=lambda_range,
                                         alpha=alpha, num_rank=2)

            if save_to_file:
                arrays_with_names = {'sigma_forward_range': sigma_forward_range,
                                     'sigma_backward_range': sigma_backward_range,
                                     'lambda_forward_range': lambda_forward_range,
                                     'lambda_backward_range': lambda_backward_range}
                np.savez(els_hyperparameters_file, **arrays_with_names)

    print 'Hyperparameters will be selected from the ranges:'
    print 'sigma_forward_range =', sigma_forward_range
    print 'sigma_backward_range =', sigma_backward_range
    print 'lambda_forward_range =', lambda_forward_range
    print 'lambda_backward_range =', lambda_backward_range

    # Change-point scores.
    packed_sequence_size = counts_packed.shape[0]
    original_sequence_size = counts.shape[0]
    scores = np.ma.masked_all(original_sequence_size)

    # Start timing here.
    timing_start = datetime.now()

    # Sliding-window over packed sequence.
    for i in range(n, packed_sequence_size - n + 1):
        forward_window = counts_packed[i: i + n]
        backward_window = counts_packed[i - n: i]
        forward_density_obj = densratio(backward_window,
                                        forward_window,
                                        alpha=alpha,
                                        sigma_range=sigma_forward_range,
                                        lambda_range=lambda_forward_range,
                                        verbose=False)
        forward_divergence = forward_density_obj.alpha_PE
        backward_density_obj = densratio(forward_window, backward_window,
                                         alpha=alpha,
                                         sigma_range=sigma_backward_range,
                                         lambda_range=lambda_backward_range,
                                         verbose=False)
        backward_divergence = backward_density_obj.alpha_PE
        change_point_score = forward_divergence + backward_divergence

        # Use larger range of hyperparameters if we can't get a good fit with the smaller one.
        if change_point_score < 0:
            print 'Bad fit with forward sigma = %0.2f, backward sigma = %0.2f.' % (forward_density_obj.kernel_info.sigma, backward_density_obj.kernel_info.sigma)
            sigma_range = np.array([
                0.7 * dmed,
                0.8 * dmed,
                0.9 * dmed,
                dmed,
                1.1 * dmed,
                1.2 * dmed,
                1.3 * dmed])

            forward_density_obj = densratio(backward_window, forward_window,
                                            alpha=alpha,
                                            sigma_range=sigma_range,
                                            verbose=False)
            forward_divergence = forward_density_obj.alpha_PE
            backward_density_obj = densratio(forward_window, backward_window,
                                             alpha=alpha,
                                             sigma_range=sigma_range,
                                             verbose=False)
            backward_divergence = backward_density_obj.alpha_PE

            change_point_score = forward_divergence + backward_divergence

            print 'Tried again with forward sigma = %0.2f, backward sigma = %0.2f.' % (forward_density_obj.kernel_info.sigma, backward_density_obj.kernel_info.sigma)

        scores[i + k//2] = change_point_score
        print 'Change-point score at time %s computed as %0.4f.' % (datetime.strftime(mdates.num2date(times[i]), '%d-%m-%Y/%H:%M'), scores[i])

    # End time.
    timing_end = datetime.now()

    # Compute average time taken.
    total_time = (timing_end - timing_start).total_seconds()
    num_evals = packed_sequence_size - 2 * n + 1
    print '%0.2f seconds taken for %d change-point score evaluations. Average is %0.2f evals/sec, with k = %d, and n = %d.' % \
          (total_time, num_evals, num_evals/total_time, k, n)

    # Mask negative change-point scores.
    scores = np.ma.masked_less(scores, 0)

    # Plot change-point scores over windows as well as the original data.
    print 'Plotting...'

    fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True)

    plot_raw_ELS_data(fig, ax0, els_data_file, quantity,
                      start_time, end_time,
                      colorbar_range='subset', colorbar_orientation='horizontal')

    ax1.plot(times, scores)
    ax1.set_ylabel('Change-point Score')
    ax1.xaxis.set_tick_params(labelsize=8)

    # Place title below.
    fig.text(s='Change-point Scores for ELS Data \n k = %d, n = %d' % (k, n), x=0.5, y=0.03,
             horizontalalignment='center', fontsize=13)

    plt.subplots_adjust(bottom=0.3, left=0.2)

    # Save plot.
    if output_file is None:
        plt.show()
    else:
        plt.savefig(output_file, bbox_inches='tight')

    # Save scores.
    if save_to_file:
        rulsif_output_file = os.path.splitext(els_data_file)[0] + '_RuLSIF_output'
        arrays_with_names = {'scores': scores, 'times': times}
        np.savez(rulsif_output_file, **arrays_with_names)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)

    parser.add_argument('els_data_file',
                        help='ELS DAT file.')
    parser.add_argument('-o', '--output_file', default=None,
                        help='File to store output plot.')
    parser.add_argument('-q', '--quantity', default='anode5', choices=(
        ['iso'] +
        [('anode%d' % a) for a in range(1, 9)] +
        [('def%d'   % a) for a in range(1, 9)] +
        [('dnf%d'   % a) for a in range(1, 9)] +
        [('psd%d'   % a) for a in range(1, 9)]
    ))
    parser.add_argument('-k', dest='k', default=2, type=int,
                        help='RuLSIF packing parameter k.')
    parser.add_argument('-n', dest='n', default=10, type=int,
                        help='RuLSIF window size n.')
    parser.add_argument('-est', '--estimate', dest='perform_hyperparameter_estimation', default=False, action='store_true',
                        help='Estimate and restrict RuLSIF hyperparameter range by random sampling.')
    parser.add_argument('-l', '--load_from_file', default=False, action='store_true',
                        help='Load values saved for this file previously, if saved.')
    parser.add_argument('-s', '--save_to_file', default=False, action='store_true',
                        help='Save values for this file when computed. Files are stored in the ELS data file\'s directory, as {els_data_file}_RuLSIF_{name of value}.npz.')
    parser.add_argument('-st', '--start_time', default=None,
                        help='Start time in dd-mm-yyyy/HH:MM. Restricts data to those recorded on or after this time.')
    parser.add_argument('-et', '--end_time', default=None,
                        help='End time in dd-mm-yyyy/HH:MM. Restricts data to those recorded upto and including this time.')
    parser.add_argument('-t', '--test', dest='run_tests', default=False, action='store_true',
                        help='Run doctests before performing RuLSIF analysis.')
    parser.add_argument('-pps', '--plot_processed_sequence', default=False, action='store_true',
                        help='Show a plot of the processed sequence after extracting from the ELS data. This is the sequence passed on to RuLSIF.')

    args = parser.parse_args()
    main(**vars(args))
