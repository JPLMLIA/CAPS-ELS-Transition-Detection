#!/usr/bin/env python

# External dependencies.
from __future__ import division
import os
import numpy as np
import argparse
from datetime import datetime
from funcsigs import signature, Parameter
import matplotlib.dates as mdates
import timeit
import h5py
from pathlib2 import Path

# Internal dependencies.
from data_utils import get_ELS_data, get_Intel_data, plot_Intel_data, float_to_datestring, format_dict

# Functions being evaluated.
# RuLSIF.
from rulsif_analysis import rulsif_analysis
# HOT-SAX.
from sax_analysis import sax_analysis
# find_bimodal.
from find_bimodal import find_bimodal
# Matrix Profile.
from matrix_profile_analysis import matrix_profile
# HMMs.
from hmm_analysis import hmm_analysis
# Baseline.
from baseline_analysis import baseline_analysis

"""
To evaluate your own algorithm:
  * Import it here.
  * Add a subparser as shown at the very bottom to handle parameters, with a name for the algorithm.
  * Add the name chosen and the function to be called to algorithm_name_map.
  * Run this module from the command-line passing the name chosen and the required parameters.

The following points are important for the evaluation to proceed correctly:
  * The function that will serve as your algorithm's entry point must take the input as the first argument.
  * The input data is a tuple of (counts, energy_range, times) - which your function will have to unpack.
  * Further, your function must also accept a parameter anomaly_type, described below. 
  * Depending on the anomaly type, your function must return different things. 
    * If the anomaly_type is 'change_points', your function must return the times and scores for each timestep it scores.
    * If the anomaly_type is 'bimodality'/'negative_ions', your function must return a list of intervals indicating where the anomalies are, and their respective scores.
"""


# Wrapper function to add timing to function calls.
def timed(func):
    def timed_func(*args, **kwargs):
        print 'Parameters passed: %s.' % kwargs
        timing_start = timeit.default_timer()
        result = func(*args, **kwargs)
        timing_end = timeit.default_timer()
        total_time = (timing_end - timing_start)
        print '%0.2f seconds taken by function.' % total_time
        return result, total_time
    return timed_func


def main(data_file, results_file, algorithm_name, dataset, quantity, start_time, end_time, anomaly_type, blur_sigma, bin_selection, filter, filter_size, **kwargs):

    # Seed for reproducibility.
    np.random.seed(7)

    # Create the directory for the results file.
    results_directory = os.path.dirname(results_file)
    if results_directory != '' and not os.path.exists(results_directory):
        Path(results_directory).mkdir(parents=True, exist_ok=True)

    # Check input arguments - start and end times should be valid.
    if start_time is not None:
        try:
            start_time_dt = datetime.strptime(start_time, '%d-%m-%Y/%H:%M')
        except ValueError:
            raise
    else:
        start_time_dt = datetime.min

    if end_time is not None:
        try:
            end_time_dt = datetime.strptime(end_time, '%d-%m-%Y/%H:%M').replace(second=59, microsecond=999999)
        except ValueError:
            raise
    else:
        end_time_dt = datetime.max

    # The time-series data loaded.
    if dataset == 'els':
        print 'ELS Bin Selection:', bin_selection
        data = get_ELS_data(data_file, quantity, start_time_dt, end_time_dt, blur_sigma, bin_selection, filter, filter_size)
    elif dataset == 'intel':
        data = get_Intel_data(data_file, 'temperature', start_time_dt, end_time_dt, downsample_rate='20min', drop_sensors=[5, 15, 18])
    else:
        raise ValueError('Invalid dataset.')

    # Update start and end times, to match the data.
    _, _, datatimes = data
    start_time = datatimes[0]
    end_time = datatimes[-1]

    print 'Data start time:', float_to_datestring(start_time)
    print 'Data end time:',  float_to_datestring(end_time)

    # The function representing the entry point for the algorithm.
    func = algorithm_name_map[algorithm_name]

    # The list of arguments this function expects.
    # We exclude the first argument, which will be the input data.
    args = signature(func).parameters.items()[1:]

    # Arguments (parameters) being passed from the command-line.
    # Each argument should have a default value in the original function, or be set from the command-line here.
    for arg, val in args:
        if val.default is Parameter.empty and kwargs.get(arg) is None:
            raise ValueError('Please pass a value for argument %s.' % arg)

    parameters = {arg: kwargs.get(arg) for arg, val in args if kwargs.get(arg) is not None}

    # Add the anomaly type as a parameter.
    # parameters['anomaly_type'] = anomaly_type

    # Call function with arguments!
    print 'Evaluating function with given parameters...'
    results, time_taken = timed(func)(data, **parameters)
    times, scores = results
    print 'Function evaluation complete!'

    # Save results (and metadata) to file.
    data_dict = {
        'times': times, 
        'scores': scores, 
        'time_taken': time_taken, 
        'anomaly_type': anomaly_type,
        'quantity': quantity, 
        'start_time': start_time,
        'end_time': end_time,
        'data_file': data_file,
        'dataset': dataset,
        'algorithm_name': formatted_algorithm_names[algorithm_name],
        'parameters': format_dict(parameters),
        'blur_sigma': blur_sigma,
        'bin_selection': bin_selection,
        'filter': filter,
        'filter_size': filter_size,
    }

    with h5py.File(results_file, 'w') as f:
        for key, val in data_dict.items():
            f.create_dataset(key, data=val)

    print('Results saved to %s. Use evaluate_methods.py/evaluate_methods_time_tolerance.py to analyze results.' % results_file)


if __name__ == '__main__':

    # Map the algorithm's name to the function indicating the entry point for your algorithm.
    algorithm_name_map = {
        'rulsif': rulsif_analysis,
        'hotsax': sax_analysis,
        'find_bimodal': find_bimodal,
        'matrix_profile': matrix_profile,
        'hmm': hmm_analysis,
        'baseline': baseline_analysis,
    }

    formatted_algorithm_names = {
        'rulsif': 'RuLSIF',
        'hotsax': 'HOT-SAX',
        'find_bimodal': 'Find-Bimodal',
        'matrix_profile': 'Matrix Profile',
        'hmm': 'HMM',
        'baseline': 'L2-Diff',
    }

    # Create parsers.
    complete_parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, add_help=False)

    # Common arguments for all algorithms - data related.
    complete_parser.add_argument('data_file',
                                 help='Data file. Example: a ELS .DAT file.')
    complete_parser.add_argument('results_file',
                                 help='File to save results to.')
    complete_parser.add_argument('-at', '--anomaly_type', default='change_points', choices=['change_points', 'bimodality', 'negative_ions'],
                                 help='The type of anomaly to load and evaluate against.')
    complete_parser.add_argument('-data', '--dataset', default='els', choices=['els', 'intel'],
                                 help='Dataset name. This is required to call the right function to load the data.')
    complete_parser.add_argument('-q', '--quantity', default='anode5', choices=(
                                        ['iso'] +
                                        ['anodes_all'] +
                                        [('anode%d' % a) for a in range(1, 9)] +
                                        [('def%d' % a) for a in range(1, 9)] +
                                        [('dnf%d' % a) for a in range(1, 9)] +
                                        [('psd%d' % a) for a in range(1, 9)]
                                ))
    complete_parser.add_argument('-st', '--start_time', default=None,
                                 help='Start time in dd-mm-yyyy/HH:MM. Restricts data to those recorded on or after this time.')
    complete_parser.add_argument('-et', '--end_time', default=None,
                                 help='End time in dd-mm-yyyy/HH:MM. Restricts data to those recorded upto and including this time.')
    complete_parser.add_argument('--transform', default='no_transform', choices=['no_transform', 'anscombe_transform', 'log_transform'],
                                 help='Choice of transformation applied to counts.')
    complete_parser.add_argument('-b', '--blur_sigma', type=int, default=0,
                                help='Parameter sigma of the Gaussian blur applied to ELS data.')
    complete_parser.add_argument('--bin_selection', choices=('all', 'center', 'ignore_unpaired'), default='all',
                                help='Selection of ELS bins.')
    complete_parser.add_argument('-f', '--filter', choices=('min_filter', 'median_filter', 'max_filter', 'no_filter'), default='no_filter',
                                help='Filter to pass ELS data through, after the Gaussian blur.')
    complete_parser.add_argument('-fsize', '--filter_size', type=int, default=1,
                                help='Size of filter to pass ELS data through, after the Gaussian blur.')

    # Subparsers for individual algorithms.
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    subparsers = parser.add_subparsers(help='Algorithm to evaluate.', dest='algorithm_name')

    # Parser for RuLSIF.
    parser_rulsif = subparsers.add_parser('rulsif', help='Parameters for RuLSIF.', parents=[complete_parser])
    parser_rulsif.add_argument('-k', dest='k', default=2, type=int,
                        help='RuLSIF packing parameter k.')
    parser_rulsif.add_argument('-n', dest='n', default=10, type=int,
                        help='RuLSIF window size n.')
    parser_rulsif.add_argument('-al', '--alpha', default=0.1, type=float,
                        help='RuLSIF \'mixture\' parameter alpha.')
    parser_rulsif.add_argument('-est', '--estimate', dest='perform_hyperparameter_estimation', default=False, action='store_true',
                        help='Estimate and restrict RuLSIF hyperparameter range by random sampling.')
    
    # Parser for HOT-SAX.
    parser_sax = subparsers.add_parser('hotsax', help='Parameters for HOT-SAX.', parents=[complete_parser])
    parser_sax.add_argument('--sax_type', default='zscore', choices=['pca', 'ica', 'repeat', 'zscore', 'energy', 'independent'],
                        help='Choice of method to get a SAX representation for multidimensional data.')
    parser_sax.add_argument('--no_heuristics', default=False, action='store_true',
                        help='Do not use SAX heuristics for discord search.')
    parser_sax.add_argument('-w', '--window_size', default=10, type=int,
                        help='SAX window size parameter w.')
    parser_sax.add_argument('-a', '--alphabet_size', default=3, type=int,
                        help='SAX alphabet size parameter n.')
    parser_sax.add_argument('-z', '--znorm_threshold', default=None, type=float,
                        help='Threshold for variance - do not normalize sequences with lesser variance.')
    parser_sax.add_argument('-n', '--num_discords', default=np.inf, type=int,
                        help='Number of discords to identify.')
    parser_sax.add_argument('-num_pca', '--num_pca_components', default=None, type=int,
                        help='Apply PCA to the multidimensional ELS data, and retain these many PCA components. Note, this is distinct from SAX-PCA, which uses necessarily uses only the first principal PCA component.')
    parser_sax.add_argument('--pca_components_folder', type=str, default=None,
                        help='Load computed PCA components from this folder.')
    
    # Parser for find_bimodal.
    parser_find_bimodal = subparsers.add_parser('find_bimodal', help='Parameters for find_bimodal.', parents=[complete_parser])
    parser_find_bimodal.add_argument('-m', '--min_dur', default=60, type=int,
                        help='Minimum duration for a run in seconds. Note that runs cut-off at the end of the interval can be shorter than this duration.')

    # Parser for the matrix profile.
    parser_matrix_profile = subparsers.add_parser('matrix_profile', help='Parameters for Matrix Profile.', parents=[complete_parser])
    parser_matrix_profile.add_argument('-w', '--window_size', default=10, type=int,
                        help='Window size parameter w. This is the size of the subsequences used in the matrix profile.')
    parser_matrix_profile.add_argument('-d', '--discord_dimensions', default=None, type=int,
                        help='The number of dimensions the discords you want to identify span over.')
    parser_matrix_profile.add_argument('-num_pca', '--num_pca_components', default=None, type=int,
                        help='Apply PCA to the multidimensional ELS data, and retain these many PCA components.')
    parser_matrix_profile.add_argument('-ign', '--ignored_dimensions', nargs='+', default=[], type=int,
                        help='Ignore these dimensions in the data.')
    parser_matrix_profile.add_argument('-req', '--required_dimensions', nargs='+', default=[], type=int,
                        help='Require these dimensions when constructing the matrix profile.')
    parser_matrix_profile.add_argument('-noise', '--std_noise', default=0, type=float,
                        help='Standard deviation for noise correction.')
    parser_matrix_profile.add_argument('--pca_components_folder', type=str, default=None,
                        help='Load computed PCA components from this folder.')
    
    # Parser for the HMM.
    parser_hmm = subparsers.add_parser('hmm', help='Parameters for the Hidden Markov Models.', parents=[complete_parser])
    parser_hmm.add_argument('-n', '--num_states', default=3, type=int,
                        help='Number of hidden states in the HMM. This is the maximum number of states if the HDP-HMMs are used.')
    parser_hmm.add_argument('-num_pca', '--num_pca_components', default=None, type=int,
                        help='Number of PCA components to keep, if applied.')
    parser_hmm.add_argument('--type', '--hmm_type', dest='hmm_type', default='vanilla', choices=['vanilla', 'hdp', 'stickyhdp'],
                        help='HMM type to model as.')
    parser_hmm.add_argument('-k', '--kappa', dest='stickiness', type=float, default=10,
                        help='Stickiness parameter for the sticky HDP-HMM.')
    parser_hmm.add_argument('-a', '--alpha', type=float, default=1.0,
                        help='Dirichlet parameter alpha for the HDP-HMM.')
    parser_hmm.add_argument('-g', '--gamma', type=float, default=1.0,
                        help='Dirichlet parameter gamma for the HDP-HMM.')
    parser_hmm.add_argument('--mixture', dest='mixture_model', default=False, action='store_true',
                        help='Model as mixture of Gaussians with 2 components.')
    parser_hmm.add_argument('--pca_components_folder', type=str, default=None,
                        help='Load computed PCA components from this folder.')
    
    # Parser for baseline algorithm (L2-diff).
    parser_baseline = subparsers.add_parser('baseline', help='Parameters for the baseline L2-diff algorithm.', parents=[complete_parser])

    # Parse arguments and pass on to main().
    args = parser.parse_args()
    main(**vars(args))
