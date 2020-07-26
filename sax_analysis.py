#!/usr/bin/env python
"""
Applies multidimensional SAX to a time-series.

Author: Ameya Daigavane
"""

# External dependencies
from __future__ import division
from datetime import datetime
import argparse
from logging import warning
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from saxpy.hotsax import find_discords_hotsax
from saxpy.discord import find_discords_brute_force
from sklearn.decomposition import PCA, FastICA

# Internal dependencies
from plot_els import plot_raw_ELS_data
from data_utils import get_ELS_data, uniform_blur
from transform import Transformation


# Function called within this script, and can be imported from other modules.
def sax_analysis(input_data, alphabet_size, window_size, num_discords, sax_type,
                 znorm_threshold=None, num_pca_components=None,
                 pca_components_folder=None, plot_processed_sequence=False,
                 no_heuristics=False, verbose=False, return_sequence=False,
                 anomaly_type='change_points', transform='no_transform'):
    """
        * SAX-PCA:
          Reduce dimensionality to 1 using PCA, and then use ordinary SAX.

        * SAX-ICA:
          Reduce dimensionality to 1 using ICA, and then use ordinary SAX.

        * SAX-REPEAT:
          Apply standard SAX to dimensions separately, then concatenate and
          cluster to get a string in the original alphabet size.

        * SAX-ZSCORE:
          Modify the z-normalization to multi-dimensional sequences and the
          PAA aggregation to the average along data-dimensions.

        * SAX-ENERGY:
          Apply SAX on the energy counts curve, and concatenate.

        * SAX-INDEPENDENT:
          Apply standard SAX to dimensions separately, then concatenate but do
          not cluster. The alphabets for each dimensions are identical.
    """
    if anomaly_type != 'change_points':
        raise NotImplementedError

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

    if sax_type == 'pca':
        # PCA to reduce dimensionality to 1.
        pca = PCA(n_components=1)
        sequence = pca.fit_transform(counts).flatten()

        # Plot first three components of PCA.
        pca_fit = PCA(n_components=3).fit(counts)
        if plot_processed_sequence:
            verbose_print('Plotting PCA components...')
            fig, ax = plt.subplots(1, 1)
            ax.plot(energy_range, pca_fit.components_[0], '-ro', label='First PCA Component')
            ax.plot(energy_range, pca_fit.components_[1], '-bo', label='Second PCA Component')
            ax.plot(energy_range, pca_fit.components_[2], '-go', label='Third PCA Component')
            ax.set_title('PCA Components for ELS Data')
            ax.legend(loc='upper right')
            plt.show()

        verbose_print('%0.2f%% of the variance explained by first component of PCA.' % (pca_fit.explained_variance_ratio_[0] * 100))

        if num_pca_components is not None:
            warning('SAX type chosen as SAX-PCA. num_pca_components has been ignored.')

    elif sax_type == 'ica':
        # ICA to identify a independent source for bimodality.
        ica = FastICA(n_components=1)
        sequence = ica.fit_transform(counts).flatten()

        # Plot first 3 components of ICA.
        if plot_processed_sequence:
            ica_fit = FastICA(n_components=3).fit(counts)
            verbose_print('Plotting ICA components...')
            fig, ax = plt.subplots(1, 1)
            ax.plot(energy_range, ica_fit.components_[0], '-ro', label='First ICA Component')
            ax.plot(energy_range, ica_fit.components_[1], '-bo', label='Second ICA Component')
            ax.plot(energy_range, ica_fit.components_[2], '-go', label='Third ICA Component')
            ax.set_title('ICA Components for ELS Data')
            ax.legend(loc='upper right')
            plt.show()

        if num_pca_components is not None:
            warning('SAX type chosen as SAX-ICA. num_pca_components has been ignored.')

    else:
        # Apply PCA if we have to.
        if num_pca_components is not None and num_pca_components > 0:
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

    # See the sequence plotted (lineplot for 1D data, colourplot for 2D data).
    if plot_processed_sequence:
        verbose_print('Plotting processed sequence...')
        fig, ax = plt.subplots(1, 1)
        ax.set_title('Processed Sequence')

        if len(sequence.shape) == 1:
            ax.xaxis_date()
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y/%H:%M'))
            fig.autofmt_xdate()
            ax.plot(times, sequence)

        elif len(sequence.shape) == 2:
            plt.imshow(sequence.T, origin='lower', interpolation='none')
            ax.set_aspect('auto')
            plt.colorbar(ax=ax, orientation='vertical')

        plt.show()

    # Start timing here.
    timing_start = datetime.now()

    # Estimate znorm_threshold, if not passed explicitly.
    if znorm_threshold is None:
        num_random_samples = 20
        random_subsequences_indices = np.random.randint(low=0, high=sequence.shape[0] - window_size,
                                                        size=num_random_samples)
        random_subsequences_std = [np.std(sequence[index: index + window_size])
                                   for index in random_subsequences_indices]
        znorm_threshold = sorted(random_subsequences_std)[num_random_samples//10]

    # Apply brute-force if explicitly asked for. Otherwise, apply HOT-SAX.
    if no_heuristics:
        verbose_print('Applying brute-force to find discords...')
        discords = find_discords_brute_force(sequence, window_size,
                                             num_discords=num_discords,
                                             znorm_threshold=znorm_threshold)
    else:
        verbose_print('Applying HOT-SAX to find discords...')
        discords = find_discords_hotsax(sequence, win_size=window_size,
                                        num_discords=num_discords,
                                        alphabet_size=alphabet_size,
                                        paa_size=window_size,
                                        znorm_threshold=znorm_threshold,
                                        sax_type=sax_type)

    # End timing here.
    timing_end = datetime.now()

    # Log time taken.
    total_time = (timing_end - timing_start).total_seconds()
    verbose_print('%0.2f seconds taken to find %d discords. On average, it took %0.2f seconds per discord on a time-series of length %d.'
                  % (total_time, len(discords), total_time / len(discords), counts.shape[0]))

    # Return sequence and discords if required.
    if return_sequence:
        return sequence, discords, znorm_threshold
    else:
        # Sort according to start-time. Each discord is a 2-tuple of (discord_start_time, discord_score).
        discords = sorted(discords)

        # Separate indexes and scores.
        discord_indexes = [discord[0] + window_size//2 for discord in discords]
        discord_scores = [discord[1] for discord in discords]

        # Updates scores at all discord start points.
        original_sequence_size = counts.shape[0]
        scores = np.zeros(original_sequence_size)
        scores[discord_indexes] = discord_scores

        # Uniformly blur scores.
        scores = uniform_blur(scores, 2)

        # Return times and the discord scores.
        return times, scores


def main(els_data_file, output_file, quantity, start_time, end_time, run_tests,
         plot_processed_sequence, alphabet_size, window_size, num_discords,
         sax_type, no_heuristics, znorm_threshold, num_pca_components):

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

    # Get ELS data.
    counts, energy_range, times = get_ELS_data(els_data_file, quantity, start_time, end_time)

    # Find discords.
    sequence, discords, znorm_threshold = sax_analysis(
        (counts, energy_range, times), alphabet_size, window_size, num_discords,
        znorm_threshold=znorm_threshold,
        sax_type=sax_type,
        num_pca_components=num_pca_components,
        plot_processed_sequence=plot_processed_sequence,
        no_heuristics=no_heuristics,
        verbose=True, return_sequence=True
    )

    for discord in discords:
        print 'Discord found starting at time %s (index %3d) with nearest-neighbour distance as %0.4f.' \
              % (datetime.strftime(mdates.num2date(times[discord[0]]), '%d-%m-%Y/%H:%M'), discord[0], discord[1])

    # Plot discords as well as the original data.
    print 'Plotting...'

    fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True)

    plot_raw_ELS_data(fig, ax0, els_data_file, quantity, start_time, end_time,
                      colorbar_range='subset', colorbar_orientation='horizontal')

    ax1.plot(times, sequence)
    ax1.set_ylabel('ELS Data \n after SAX-%s' % (sax_type.upper()))
    ax1.xaxis.set_tick_params(labelsize=8)

    for discord in discords:
        ax1.axvspan(xmin=times[discord[0]], xmax=times[discord[0] + window_size - 1], color='r', alpha=0.2)

    # Place title below.
    fig.text(s='Discords in ELS Data \n znorm_threshold = %0.3f, window_size = %d' % (znorm_threshold, window_size),
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
    parser.add_argument('--sax_type', default='zscore', choices=['pca', 'ica', 'repeat', 'zscore', 'energy', 'independent'],
                        help='Choice of method to get a SAX representation for multidimensional data.')
    parser.add_argument('--no_heuristics', default=False, action='store_true',
                        help='Do not use SAX heuristics for discord search.')
    parser.add_argument('-w', '--window_size', default=10, type=int,
                        help='SAX window size parameter w.')
    parser.add_argument('-a', '--alphabet_size', default=3, type=int,
                        help='SAX alphabet size parameter n.')
    parser.add_argument('-z', '--znorm_threshold', default=None, type=float,
                        help='Threshold for variance - do not normalize sequences with lesser variance.')
    parser.add_argument('-n', '--num_discords', default=5, type=int,
                        help='Number of discords to identify.')
    parser.add_argument('-num_pca', '--num_pca_components', default=None, type=int,
                        help='Apply PCA to the multidimensional ELS data, and retain these many PCA components. Note, this is distinct from SAX-PCA, which uses necessarily uses only the first principal PCA component.')
    parser.add_argument('-st', '--start_time', default=None,
                        help='Start time in dd-mm-yyyy/HH:MM. Restricts data to those recorded on or after this time.')
    parser.add_argument('-et', '--end_time', default=None,
                        help='End time in dd-mm-yyyy/HH:MM. Restricts data to those recorded upto and including this time.')
    parser.add_argument('-t', '--test', dest='run_tests', default=False, action='store_true',
                        help='Run doctests before performing SAX analysis.')
    parser.add_argument('-pps', '--plot_processed_sequence', default=False, action='store_true',
                        help='Show a plot of the processed sequence after extracting from the ELS data. This is the sequence passed on to SAX.')

    args = parser.parse_args()
    main(**vars(args))
