#!/usr/bin/env python
"""
Segments time-series with HMM models.

Author: Ameya Daigavane
"""

# External dependencies
from __future__ import division
from datetime import datetime
import argparse
import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from pyhsmm.basic.distributions import Gaussian
from pybasicbayes.models import MixtureDistribution

# Internal dependencies
from plot_els import plot_interpolated_ELS_data
from data_utils import get_ELS_data, array_to_intervals, reconstruct_from_PCA, kl_divergence_normals
from hmm_models import VanillaGaussianHMM, VanillaGaussianMixtureHMM, HDPHMM, StickyHDPHMM
from transform import Transformation

# Creates an HMM to segment the time-series.
def hmm_analysis(data, num_states=3, num_pca_components=None, verbose=False, evaluation=True, hmm_type='vanilla', anomaly_type='change_points', stickiness=10, alpha=None, gamma=None, mixture_model=False, pca_components_folder=None, transform='no_transform'):

    # If verbose, print to console.
    def verbose_print(arg):
        if verbose:
            print arg

    if anomaly_type != 'change_points':
        raise NotImplementedError

    # Unpack input data.
    counts, energy_range, times = data

    # Transform counts if required.
    counts = Transformation(transform).transform(counts)

    # Apply PCA if we have to.
    apply_pca = num_pca_components is not None and num_pca_components > 0
    if apply_pca:
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
                print('PCA components file %s not found.' % pca_components_file)

    else:
        sequence = counts

    if hmm_type == 'vanilla':
        # Mixture or singlular Gaussian emissions.
        if mixture_model:
            model = VanillaGaussianMixtureHMM(n_components=num_states, n_mix=2)
        else:
            model = VanillaGaussianHMM(n_components=num_states)

    elif hmm_type in ['hdp', 'stickyhdp']:

        resample_over_Dirichlet = False

        # Maximum number of states.
        max_states = num_states

        # Dimensionality of the time-series.
        data_dimensions = sequence.shape[1]

        # Hyperparameters for the observations.
        if mixture_model:
            # Modelling emission probabilities as a mixture of two Gaussians, instead.
            observation_hyperparameters = [{'mu_0':  np.zeros(data_dimensions),
                                            'sigma_0': np.eye(data_dimensions),
                                            'kappa_0': 0.25,
                                            'nu_0': data_dimensions + 2},
                                           {'mu_0': np.zeros(data_dimensions),
                                            'sigma_0': np.eye(data_dimensions),
                                            'kappa_0': 0.25,
                                            'nu_0': data_dimensions + 2}]
            observation_dist = [MixtureDistribution(alpha_0=1., components=[Gaussian(**observation_hyperparameters[group])
                                                                            for group in range(2)]) for state in range(max_states)]
        else:
            # Modelling emission probabilities as a Gaussian with a conjugate Normal/Inverse-Wishart prior.
            observation_hyperparameters = {'mu_0': np.zeros(data_dimensions),
                                           'sigma_0': np.eye(data_dimensions),
                                           'kappa_0': 0.25,
                                           'nu_0': data_dimensions + 2}
            observation_dist = [Gaussian(**observation_hyperparameters) for state in range(max_states)]

        if resample_over_Dirichlet:
            params = {'alpha_a_0': 1., 'alpha_b_0': 1./4,
                      'gamma_a_0': 1., 'gamma_b_0': 1./4,
                      'init_state_concentration': 1, 'obs_distns': observation_dist}
        else:
            params = {'alpha': alpha, 'gamma': gamma,
                      'init_state_concentration': 1, 'obs_distns': observation_dist}

        # Create model.
        if hmm_type == 'hdp':
            model = HDPHMM(**params)
        else:
            model = StickyHDPHMM(kappa=stickiness, **params)

    else:
        raise ValueError('Invalid HMM type.')

    # Learn HMM parameters.
    model.fit(sequence)

    # Estimate log-likelihood of sequence conditional on learned parameters.
    log_likelihood = model.log_likelihood()

    # Get posterior distribution over states, conditional on the input.
    states_dist = model.state_distribution()

    # Assign to state with maximum probability at each timestep.
    states = np.argmax(states_dist, axis=1)
   
    states_list, states_count = np.unique(states, return_counts=True)
    # for state in states_list:
    #     mu_peaks = peaks(model.emission_params(state)['mu'])
    #     num_mu_peaks = len(mu_peaks)
    #     if num_mu_peaks == 0:
    #         peak_distance = 0
    #     elif 0 < num_mu_peaks <= 2:
    #         peak_distance = mu_peaks[-1] - mu_peaks[0]
    #     else:
    #         peak_distance = mu_peaks[num_mu_peaks//2 + 1] - mu_peaks[num_mu_peaks//2]

    #     print 'State %d has peaks at indices %s giving it a score of %d.' % (state, mu_peaks, peak_distance)

    if len(states) != len(sequence):
        raise AssertionError

    # reconstructions = {}
    # for state, observation in zip(states, sequence):
    #     sample = model.generate_samples(state, 1)

    #     if state not in reconstructions:
    #         reconstructions[state] = 0

    #     reconstructions[state] += np.sqrt(np.sum(np.square(sample - observation)))

    # for state, count in zip(reconstructions, states_count):
    #     reconstructions[state] /= count

    # print 'State-wise mean reconstructions:', reconstructions

    # # Remove states that have high reconstruction.
    # median_reconstruction = np.median(reconstructions.values())
    # for state in reconstructions:
    #     if reconstructions[state] > 2 * median_reconstruction:
    #         states_dist[:, state] = 0

    # for timestep, dist in enumerate(states_dist):
    #     if np.sum(dist) == 0:
    #         states_dist[timestep] = states_dist[timestep - 1]

    # # Recompute after removing.
    # states_dist /= np.sum(states_dist, axis=1, keepdims=True)
    # states = np.argmax(states_dist, axis=1)

    # Time factor indicating how long each timestep is approximately in seconds.
    time_factor = round(np.min(np.diff(times)) * 60 * 60 * 24)

    # Remove state subsequences that are very short.
    while True:
        state_intervals = array_to_intervals(states)

        if len(state_intervals) <= 2:
            break

        found_short_subsequence = False
        for state in state_intervals:
            for state_start, state_end in state_intervals[state]:
                if state_end - state_start < 300 / time_factor:
                    found_short_subsequence = True
                    states_dist[state_start: state_end, state] = 0

        if not found_short_subsequence:
            break

        # Fill in missing distributions.
        for timestep, dist in enumerate(states_dist):
            if np.sum(dist) == 0:
                if timestep == 0:
                    states_dist[timestep] = np.ones(len(dist)) / len(dist)
                else:
                    states_dist[timestep] = states_dist[timestep - 1]

        # Recompute after removing.
        states_dist /= np.sum(states_dist, axis=1, keepdims=True)
        states = np.argmax(states_dist, axis=1)

    # Compute 'distances' between states.
    print(states_dist.shape)
    all_emission_params = {state: model.emission_params(state) for state in range(num_states)}
    dissimilarities = np.zeros((num_states, num_states))
    for (index1, params1), (index2, params2) in itertools.product(all_emission_params.iteritems(), all_emission_params.iteritems()):
        dissimilarities[index1][index2] = kl_divergence_normals(params1['mu'], params1['sigma'], params2['mu'], params2['sigma']) \
                                        + kl_divergence_normals(params2['mu'], params2['sigma'], params1['mu'], params1['sigma'])

    # Normalize dissimilarities such that the entire matrix sums up to 'num_states', just like the identity matrix would.
    dissimilarities *= num_states/np.sum(dissimilarities)

    # Score as difference in distributions.
    dist_after = states_dist[1:]
    dist_before = states_dist[:-1]
    dist_diff = np.abs(dist_after - dist_before)
    dist_diff_scaled = np.matmul(dist_diff, dissimilarities)

    scores = np.sum(dist_diff * dist_diff_scaled, axis=1)
    scores = np.append([0], scores)

    if evaluation:
        if anomaly_type == 'change_points':
            return times, scores
        else:
            raise NotImplementedError

    return model, states, log_likelihood, states_dist, scores


def main(els_data_file, output_file, quantity, start_time, end_time, hmm_type, num_states, num_pca_components, show_information_curves, visualize_states, stickiness, alpha, gamma, mixture_model, **kwargs):

    # Random seed for reproducibility.
    random_seed = 7
    np.random.seed(random_seed)

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

    # Get data, and unpack.
    counts, energy_ranges, times = get_ELS_data(els_data_file, quantity, start_time, end_time, **kwargs)

    # Segment with an HMM.
    model, states, log_likelihood, states_dist, scores = \
        hmm_analysis((counts, energy_ranges, times),
                     hmm_type=hmm_type,
                     num_states=num_states,
                     num_pca_components=num_pca_components,
                     stickiness=stickiness,
                     alpha=alpha,
                     gamma=gamma,
                     mixture_model=mixture_model,
                     verbose=True, evaluation=False)

    # Intervals of states to plot.
    states_dict = array_to_intervals(states)
    print '%d HMM states used to segment the time-series.' % len(states_dict)

    # Print individual state durations.
    for state, intervals in states_dict.iteritems():
        print 'State %d durations: %s timesteps.' % (state, np.sum([interval[1] - interval[0] for interval in intervals]))

    # Plot ELS data on top.
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, sharex=True)
    plot_interpolated_ELS_data(fig, ax0, els_data_file, quantity, start_time, end_time, 
                                colorbar_range='subset', colorbar_orientation='horizontal', **kwargs)
    ax0.set_xlabel('')

    # Plot segments in different colours. We don't want to repeat colors across states.
    colors = cm.get_cmap('Set3')
    labelled_states = set()
    for index, (state, intervals) in enumerate(states_dict.iteritems()):
        for interval in intervals:
            if state in labelled_states:
                ax1.axvspan(times[interval[0]], times[interval[1]], color=colors(index))
            else:
                ax1.axvspan(times[interval[0]], times[interval[1]], color=colors(index), label=state)
                labelled_states.add(state)

    ax1.margins(x=0)
    ax1.set_yticks([])
    ax1.set_ylabel('HMM States', labelpad=10)
    ax1.legend(title='HMM States', bbox_to_anchor=(1.28, 0.5), loc='center right', fontsize=8)

    # Plot scores.
    ax2.plot(times, scores)
    ax2.set_ylabel('Scores')
    ax2.xaxis.set_tick_params(labelsize=8)
    ax2.set_xlabel('Datetime')
    plt.setp(ax2.get_xticklabels(), rotation=30, ha='right')

    # Place title below.
    formatted_name = {
        'vanilla': 'Vanilla',
        'hdp': 'HDP',
        'stickyhdp': 'Sticky HDP',
    }

    hmm_parameters = {
        'vanilla': ['num_states', 'num_pca_components'],
        'hdp': ['num_states', 'num_pca_components', 'alpha', 'gamma'],
        'stickyhdp': ['num_states', 'num_pca_components', 'alpha', 'gamma', 'stickiness'],
    }

    parameter_strings = {
        'num_states': 'HMM States = %d',
        'num_pca_components': 'PCA Components = %d',
        'stickiness': 'Stickiness = %0.2f',
        'alpha': 'Dirichlet Alpha = %0.2f',
        'gamma': 'Dirichlet Gamma = %0.2f',
    }

    # Add only supplied parameters to title.
    parameters = hmm_parameters[hmm_type]
    parameter_string_all = ', '.join([parameter_strings[parameter] % eval(parameter) for parameter in parameters])
    title = 'CAPS ELS Segmentation with %s-HMM \n %s \n Log-Likelihood = %0.2f' % (formatted_name[hmm_type], parameter_string_all, log_likelihood)
    fig.text(s=title, x=0.5, y=0.03, horizontalalignment='center', fontsize=13)

    plt.subplots_adjust(bottom=0.4, left=0.1, right=0.8)

    # Save plot.
    if output_file is None:
        plt.show()
    else:
        plt.savefig(output_file, bbox_inches='tight')

    # Visualization of individual states.
    if visualize_states:

        # Get state-wise emission parameters, to compute distances between states.
        states_map = {index: state for index, state in enumerate(states_dict)}
        num_actual_states = len(states_map)
        all_emission_params = {index: model.emission_params(state) for index, state in enumerate(states_dict)}
        dissimilarities = np.zeros((num_actual_states, num_actual_states))
        for (index1, params1), (index2, params2) in itertools.product(all_emission_params.iteritems(),
                                                                      all_emission_params.iteritems()):
            dissimilarities[index1][index2] = kl_divergence_normals(params1['mu'], params1['sigma'], params2['mu'], params2['sigma']) \
                                              + kl_divergence_normals(params2['mu'], params2['sigma'], params1['mu'], params1['sigma'])

        transformed_states = MDS(dissimilarity='precomputed').fit_transform(dissimilarities)
        spacing = 5 / np.max(transformed_states)
        colors = cm.get_cmap('Set3')
        for index, _ in enumerate(transformed_states):
            plt.scatter(transformed_states[index, 0], transformed_states[index, 1], label=states_map[index],
                        color=colors(index))
            plt.text(transformed_states[index, 0], transformed_states[index, 1] + spacing, states_map[index],
                     fontsize=8)
        plt.title('MDS Plot of HMM States')
        plt.show()

        # Recreate PCA object.
        apply_pca = num_pca_components is not None and num_pca_components > 0
        if apply_pca:
            pca = PCA(n_components=num_pca_components)
            pca.fit(counts)
            data_mean = np.mean(counts, axis=0)

        # Samples hidden states to visualize them.
        num_samples = 50

        samples_dict = {}
        for state in states_dict:
            # Sample from the learned observation distribution for each state.
            samples_dict[state] = model.generate_samples(state, num_samples)

            # If we applied PCA, project back into the original number of dimensions.
            if apply_pca:
                samples_dict[state] = reconstruct_from_PCA(samples_dict[state], data_mean, pca)

        # Set minimum and maximum values across plots for consistency.
        samples_list = list(samples_dict.itervalues())
        vmin = np.percentile(samples_list, 5)
        vmax = np.percentile(samples_list, 95)

        # Plot the projected samples!
        for state in states_dict:
            plt.ylabel('Energies')
            plt.imshow(samples_dict[state].T, extent=[0, num_samples, 0, len(energy_ranges[0])], origin='upper',
                       interpolation='none', vmin=vmin, vmax=vmax)
            plt.yticks(np.arange(0, len(energy_ranges[0]), 5), energy_ranges[0][::5])
            plt.title('HMM State %d \n Transformed Samples' % state)
            cbar = plt.colorbar(orientation='vertical')
            cbar.set_label('Counts')
            plt.show()

        # Bunch up all the samples for all the states.
        all_samples = np.vstack([samples_dict[state].flatten() for state in states_dict])
        # statewise_means = np.expand_dims(np.mean(all_samples, axis=1), axis=1)
        # statewise_devs = np.expand_dims(np.std(all_samples, axis=1), axis=1)
        normalized_samples = (all_samples - np.mean(all_samples))

        # Compute the SVD - Singular Value Decomposition.
        U, S, VH = np.linalg.svd(normalized_samples, full_matrices=False)

        # Compute reconstruction errors from an SVD.
        for num_svd_components in [1, 2, 5, 8, 10]:

            # Don't have these many components!
            if num_svd_components > num_actual_states:
                break

            # Compute reconstruction using these many SVD components.
            reconstruction = np.matmul(U[:, :num_svd_components] * S[:num_svd_components],
                                       VH[:num_svd_components, :])
            reconstruction_errors = np.sqrt(np.mean(np.square(reconstruction - normalized_samples), axis=1))

            # Plot mean reconstruction error as a function of the states.
            used_states = list(states_dict.iterkeys())
            plt.scatter(used_states, reconstruction_errors)
            plt.ylabel('Mean Reconstruction Error Over All Samples')
            plt.xlabel('HMM State Number')
            plt.title('Reconstruction Error after Retaining %d SVD Components' % num_svd_components)
            plt.ylim(bottom=0)
            plt.xticks(used_states)
            plt.show()

    # Plots the AIC and BIC values, as the number of states is varied.
    if show_information_curves:

        print 'Plotting AIC and BIC curves...'

        bics = []
        aics = []
        num_states_range = np.arange(1, 21)
        for num_states in num_states_range:
            model, _, _, _, _ = hmm_analysis((counts, energy_ranges, times), hmm_type=hmm_type, num_states=num_states, num_pca_components=num_pca_components, evaluation=False)

            # Compute AIC and BIC criteria.
            bics.append(model.bic_value())
            aics.append(model.aic_value())

        fig, ax = plt.subplots(nrows=1)
        ax.plot(num_states_range, bics, label='BIC')
        ax.plot(num_states_range, aics, label='AIC')
        ax.legend()
        ax.set_xticks(num_states_range)
        ax.set_ylabel('Information Criteria Value')
        ax.set_xlabel('Number of States')

        # Add only supplied parameters to title.
        parameter_strings = ['PCA Components = %d', 'Stickiness = %0.2f']
        parameters = [num_states, num_pca_components, stickiness]
        parameter_string = ', '.join([parameter_string % parameter
                                      for parameter, parameter_string
                                      in zip(parameters, parameter_strings)
                                      if parameter is not None])
        title = 'Selecting the Number of States via Information Criteria \n %s-HMM \n %s' % (formatted_name[hmm_type], parameter_string)

        ax.set_title(title)

        plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument('els_data_file',
                        help='ELS DAT file.')
    parser.add_argument('-o', '--output_file', default=None,
                        help='Save the change-point scores plot to this file.')
    parser.add_argument('-q', '--quantity', default='anode5', choices=(
        ['iso'] +
        [('anode%d' % a) for a in range(1, 9)] +
        [('def%d' % a) for a in range(1, 9)] +
        [('dnf%d' % a) for a in range(1, 9)] +
        [('psd%d' % a) for a in range(1, 9)]
    ))
    parser.add_argument('-n', '--num_states', default=3, type=int,
                        help='Number of hidden states in the HMM. This is the maximum number of states if the HDP-HMMs are used.')
    parser.add_argument('-num_pca', '--num_pca_components', default=0, type=int,
                        help='Number of PCA components to keep, if applied.')
    parser.add_argument('-st', '--start_time', default=None,
                        help='Start time in dd-mm-yyyy/HH:MM. Restricts data to those recorded on or after this time.')
    parser.add_argument('-et', '--end_time', default=None,
                        help='End time in dd-mm-yyyy/HH:MM. Restricts data to those recorded upto and including this time.')
    parser.add_argument('-i', '--information', dest='show_information_curves', default=False, action='store_true',
                        help='Plot AIC and BIC curves, varying the number of states.')
    parser.add_argument('--type', '--hmm_type', dest='hmm_type', default='vanilla', choices=['vanilla', 'hdp', 'stickyhdp'],
                        help='HMM type to model as.')
    parser.add_argument('-v', '--visualize', dest='visualize_states', default=False, action='store_true',
                        help='Show visualization plots for individual HMM states.')
    parser.add_argument('-k', '--kappa', dest='stickiness', type=float, default=10.,
                        help='Stickiness parameter for the sticky HDP-HMM.')
    parser.add_argument('-a', '--alpha', default=1.,
                        help='Dirichlet parameter alpha for the HDP-HMM.')
    parser.add_argument('-g', '--gamma', default=1.,
                        help='Dirichlet parameter gamma for the HDP-HMM.')
    parser.add_argument('--mixture', dest='mixture_model', default=False, action='store_true',
                        help='Model as mixture of Gaussians with 2 components.')
    parser.add_argument('-b', '--blur_sigma', type=int, default=0,
                        help='Parameter sigma of the Gaussian blur applied to ELS data.')
    parser.add_argument('--bin_selection', choices=('all', 'center', 'ignore_unpaired'), default='all',
                        help='Selection of ELS bins.')
    parser.add_argument('-f', '--filter', choices=('min_filter', 'median_filter', 'max_filter', 'no_filter'), default='no_filter',
                        help='Filter to pass ELS data through, after the Gaussian blur.')
    parser.add_argument('-fsize', '--filter_size', type=int, default=1,
                        help='Size of filter to pass ELS data through, after the Gaussian blur.')
    args = parser.parse_args()

    main(**vars(args))
