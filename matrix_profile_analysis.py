#!/usr/bin/env python
"""
Applies the Multidimensional Matrix Profile to a time-series.

Author: Ameya Daigavane
"""

# External dependencies
from __future__ import division
from datetime import datetime
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matrixprofile import matrixProfile as mp
from sklearn.decomposition import PCA

# Internal dependencies
from plot_els import plot_raw_ELS_data
from data_utils import get_ELS_data, ternary_search, simulated_annealing, binary_search
from transform import Transformation

def matrix_profile(input_data, window_size, discord_dimensions=None,
                   std_noise=None, ignored_dimensions=None,
                   required_dimensions=None, num_pca_components=None,
                   pca_components_folder=None, plot_processed_sequence=False,
                   anomaly_type='change_points', transform='no_transform',
                   verbose=False):

    if anomaly_type != 'change_points':
        raise NotImplementedError

    if ignored_dimensions is None:
        ignored_dimensions = []

    if required_dimensions is None:
        required_dimensions = []

    # If verbose, print to console.
    def verbose_print(arg):
        if verbose:
            print arg

    # Unpack input.
    counts, energy_range, times = input_data

    # Transform counts if required.
    counts = Transformation(transform).transform(counts)

    # Time factor indicating how long each timestep is approximately in seconds.
    time_factor = round(np.min(np.diff(times)) * 60 * 60 * 24)
    window_size = int(round(window_size/time_factor))

    # Save original counts.
    original_counts = counts

    # Ignore dimensions if required.
    if len(ignored_dimensions) > 0:

        # Check for any intersection between required dimensions and ignored dimensions.
        if np.intersect1d(ignored_dimensions, required_dimensions).size > 0:
            raise ValueError('Cannot have any common dimensions between ignored and required dimensions.')

        ignored_dimensions.sort()
        required_dimensions.sort()

        if ignored_dimensions[-1] >= counts.shape[1]:
            raise ValueError('Dimension to be ignored out of bounds.')

        if len(required_dimensions) > 0 and required_dimensions[-1] >= counts.shape[1]:
            raise ValueError('Required dimension out of bounds.')

        # Delete ignored dimensions.
        counts = np.delete(counts, ignored_dimensions, axis=1)

        # Update required dimensions as they corresponded to dimensions in the original time-series.
        for index, dim in enumerate(required_dimensions):
            num_deleted, _ = binary_search(dim, ignored_dimensions)

            if num_deleted is None:
                num_deleted = len(ignored_dimensions)

            required_dimensions[index] -= num_deleted

    # Smoothen counts and convert to log-space.
    # counts = gaussian_blur(counts, sigma=0.5)
    # counts = np.ma.log(counts)

    # If a unidimensional time series, just reshape.
    if len(counts.shape) == 1:
        counts = counts.reshape((counts.shape[0], 1))

    # Dimensionality reduction with PCA.
    if num_pca_components is not None:
        verbose_print('Applying PCA to reduce dimensionality to %d.' % num_pca_components)

        if pca_components_folder is None:
            verbose_print('Computing PCA components...')

            pca = PCA(n_components=num_pca_components)
            sequence = pca.fit_transform(counts)

            verbose_print('%0.2f%% of the variance explained by first %d components of PCA.'
                          % ((np.sum(pca.fit(counts).explained_variance_ratio_[:num_pca_components]) * 100), num_pca_components))
        else:
            verbose_print('Loading components from folder %s...' % pca_components_folder)
            pca_components_file = pca_components_folder + 'pca%d_components.npy' % num_pca_components

            try:
                pca_components = np.load(pca_components_file)
                sequence = np.matmul(counts - counts.mean(axis=0), pca_components.T)
            except IOError:
                print 'PCA components file %s not found.' % pca_components_file

    else:
        sequence = counts

    # Number of dimensions in the time-series.
    num_dimensions = sequence.shape[1]

    # If no dimensions asked for, return the matrix profile over all dimensions.
    if discord_dimensions is None:
        discord_dimensions = num_dimensions
    else:
        if discord_dimensions > num_dimensions or discord_dimensions <= 0:
            raise ValueError('Discord dimensions must lie between 1 and the number of dimensions in the time-series.')

    # Compute matrix profile along each dimension.
    individual_profiles = np.zeros((num_dimensions, sequence.shape[0] - window_size + 1))
    for dim in range(num_dimensions):
        # Estimate noise threshold if none passed.
        if std_noise is None:
            std_noise_dimension = noise_threshold_optimizer(sequence[:, dim], window_size)
            print '%0.03f chosen as noise standard deviation after optimization.' % std_noise_dimension
        else:
            std_noise_dimension = std_noise

        individual_profiles[dim] = mp.scrimp_plus_plus(sequence[:, dim], window_size, std_noise=std_noise_dimension, exclusion_zone_fraction=1)[0]

    # These are the required profiles. We will push these to the top.
    required_profiles = individual_profiles[required_dimensions, :]

    # Now, sort the remaining profiles in descending order (distance-wise).
    other_dimensions = np.setdiff1d(range(sequence.shape[1]), required_dimensions)
    other_profiles = -np.sort(-individual_profiles[other_dimensions, :], axis=0)

    # Stack the required profiles on top, then the sorted profiles.
    individual_profiles = np.vstack((required_profiles, other_profiles))

    # Compute overall matrix profile, by summing up individual profiles.
    profile = np.zeros((num_dimensions, sequence.shape[0] - window_size + 1))

    # Now, sum up over sorted profiles.
    profile[0] = individual_profiles[0]
    for dim in range(1, num_dimensions):
        profile[dim] = profile[dim - 1] + individual_profiles[dim]
    profile = profile.T

    # Pad profile with zeros to get a length equalling the original sequence's length.
    padded_profile = np.zeros((sequence.shape[0], num_dimensions), dtype=float)
    padded_profile[window_size//2: profile.shape[0] + window_size//2] = profile

    # Plot multidimensional matrix profile.
    if plot_processed_sequence:
        fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, sharex=True)
        ax0.set_ylabel('Original \n Sequence')
        ax0.plot(original_counts)
        ax1.set_ylabel('Sequence After \n Removing Ignored \n Dimensions')
        ax1.plot(counts)
        ax2.plot(padded_profile)
        plt.title('The Multidimensional Matrix Profile', y=-0.45)
        plt.show()

    # Return only the discord dimension.
    return times, padded_profile[:, discord_dimensions - 1]


# Sets the noise standard deviation by optimizing for a suitable objective function.
def noise_threshold_optimizer(sequence, window_size, optimization_method='ternary_search'):

    # Try to maximize ratio of maximum value of matrix profile to mean value of matrix profile.
    # Since we have to write it as a minimization problem, we flip the ratio around.
    def objective(std_noise):
        profile = mp.stomp(sequence, window_size, std_noise=std_noise)[0]

        if np.max(profile) == 0:
            return np.inf

        return np.mean(profile) / np.max(profile)

    if optimization_method == 'simulated_annealing':
        return simulated_annealing(objective, init_state=0.5)

    elif optimization_method == 'ternary_search':
        return ternary_search(objective, low=0, high=1)


def main(els_data_file, output_file, quantity, start_time, end_time, run_tests, plot_processed_sequence, window_size, discord_dimensions, num_pca_components, ignored_dimensions, required_dimensions, std_noise):

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
        doctest.testmod(data_utils, verbose=True, optionflags=doctest.NORMALIZE_WHITESPACE)
        doctest.testmod(verbose=True, optionflags=doctest.NORMALIZE_WHITESPACE)

    # Set random seed for reproducibility.
    random_seed = 7
    np.random.seed(random_seed)

    # Load data from ELS DAT file.
    ELS_data = get_ELS_data(els_data_file, quantity, start_time, end_time)

    # Get matrix profile, padded to match length of the original sequence.
    times, profile = matrix_profile(ELS_data, window_size, discord_dimensions, std_noise, num_pca_components=num_pca_components, ignored_dimensions=ignored_dimensions, required_dimensions=required_dimensions, plot_processed_sequence=plot_processed_sequence, verbose=True)

    # Plot change-point scores over windows as well as the original data.
    print 'Plotting...'

    fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True)

    plot_raw_ELS_data(fig, ax0, els_data_file, quantity, start_time, end_time,
                      colorbar_range='subset', colorbar_orientation='horizontal')

    ax1.plot(times, profile)
    ax1.set_ylabel('Non-Self NN-Distance')
    ax1.xaxis.set_tick_params(labelsize=8)

    # Add only supplied parameters to title.
    parameter_strings = ['Window Size = %d', 'Dimensions = %d', 'PCA Components = %d', 'Noise Correction = %0.2f']
    parameters = [window_size, discord_dimensions, num_pca_components, std_noise]
    parameter_string = ', '.join([parameter_string % parameter for parameter, parameter_string in zip(parameters, parameter_strings) if parameter is not None])
    title = 'Matrix Profile on CAPS ELS \n %s' % parameter_string

    # Place title below.
    fig.text(s=title,
             y=0.03, x=0.5, horizontalalignment='center', fontsize=13)

    plt.subplots_adjust(bottom=0.3, left=0.2)

    # Save plot.
    if output_file is None:
        plt.show()
    else:
        plt.savefig(output_file, bbox_inches='tight')


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
    parser.add_argument('-st', '--start_time', default=None,
                        help='Start time in dd-mm-yyyy/HH:MM. Restricts data to those recorded on or after this time.')
    parser.add_argument('-et', '--end_time', default=None,
                        help='End time in dd-mm-yyyy/HH:MM. Restricts data to those recorded upto and including this time.')
    parser.add_argument('-w', '--window_size', default=10, type=int,
                        help='Window size parameter w. This is the size of the subsequences used in the matrix profile.')
    parser.add_argument('-d', '--discord_dimensions', default=None, type=int,
                        help='The number of dimensions the discords you want to identify span over.')
    parser.add_argument('-t', '--test', dest='run_tests', default=False, action='store_true',
                        help='Run doctests before performing Matrix Profile analysis.')
    parser.add_argument('-pps', '--plot_processed_sequence', default=False, action='store_true',
                        help='Show a plot of the processed sequence after extracting from the ELS data. This is the sequence passed on to the Matrix Profile.')
    parser.add_argument('-num_pca', '--num_pca_components', default=None, type=int,
                        help='Apply PCA to the multidimensional ELS data, and retain these many PCA components.')
    parser.add_argument('-noise', '--std_noise', default=None, type=float,
                        help='Standard deviation for noise correction.')
    parser.add_argument('-ign', '--ignored_dimensions', nargs='+', default=[], type=int,
                        help='Ignore these dimensions in the data.')
    parser.add_argument('-req', '--required_dimensions', nargs='+', default=[], type=int,
                        help='Require these dimensions when constructing the matrix profile.')

    args = parser.parse_args()
    main(**vars(args))
