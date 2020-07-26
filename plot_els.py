#!/usr/bin/env python

# External dependencies.
from __future__ import division
import os
import re
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys

# Hack to import 'compute_labelled_events.py'.
sys.path.append('secondary_scripts')

# Internal dependencies.
from els_data import ELS

class DataGroup(object):

    def __init__(self, x, y, z):
        """
        x: scalar time
        y: energy levels
        z: counts (same size as y)
        """
        self.x = np.array([x])
        self.y = y
        self.z = z

    def matches(self, y):
        """
        Are there the same energy levels y as previously seen in this group?
        """
        return (y.size == self.y.size) and np.allclose(y, self.y)

    def add(self, x, z):
        """
        Append x to list of times and z to array of values
        """
        self.x = np.hstack([self.x, x])
        self.z = np.column_stack([self.z, z])

    def get_data(self, x):
        """
        Set x as the last time in the data group (right-hand side of last
        time bin), and return all data within the group
        """
        return np.hstack([self.x, x]), self.y, self.z


def parse_quantity(data, quantity):
    """
    Grabs the appropriate quantity from the ELS data object, and construct the
    appropriate label for the color scalebar label.

    :param data: ELS data object
    :param quantity: string name of quantity to extract

    :return: data matrix wth shape (records, energy levels),
             scalebar label string
    """
    if quantity == 'iso':
        D = data.iso_data
        scalelabel = 'Counts / s'
    else:
        m = re.match(r'([^\d]*)(\d*)', quantity)
        if m is None:
            raise ValueError('Unknown quantity "%s"' % quantity)
        qtype = m.group(1)
        anode = int(m.group(2))

        # (index = ordinal - 1)
        anode_idx = anode - 1

        if qtype == 'anode':
            D = data.data
            scalelabel = ('Anode %d Counts / s' % anode)
        elif qtype == 'def':
            D = data.def_data
            scalelabel = (
                'Anode %d DEF (m$^{-2}$ sr$^{-1}$ s$^{-1}$)' % anode
            )
        elif qtype == 'dnf':
            D = data.dnf_data
            scalelabel = (
                'Anode %d DNF (m$^{-2}$ sr$^{-1}$ s$^{-1}$ J$^{-1}$)'
                % anode
            )
        elif qtype == 'psd':
            D = data.psd_data
            scalelabel = ('Anode %d PSD (m$^{-6}$ s$^{-3}$)' % anode)
        else:
            raise ValueError('Unknown quantity "%s"' % quantity)
        D = np.squeeze(D[:, :, anode_idx])
    return D, scalelabel


# Adds a colorbar to a colormapped plot. Here, a plot is the returned object on calling 'imshow'/'pcolormesh'.
def add_colorbar(plot, figure, axes, colorbar_orientation):
    if colorbar_orientation == 'vertical':
        cbar = figure.colorbar(plot, ax=axes, orientation=colorbar_orientation)
    elif colorbar_orientation == 'horizontal':
        divider = make_axes_locatable(axes)
        cax = divider.append_axes('top', size='5%', pad=0.15)
        cbar = figure.colorbar(plot, cax=cax, orientation=colorbar_orientation)
        cbar.ax.xaxis.set_label_position('top')
        cbar.ax.xaxis.set_ticks_position('top')
    else:
        raise ValueError('Invalid colorbar_orientation passed. Use \'vertical\' or \'horizontal\'.')
    return cbar


def plot_raw_ELS_data(figure, axes, els_data_file, quantity, start_time=datetime.min, end_time=datetime.max, colorbar_range='subset', colorbar_orientation='vertical', verbose=True, **kwargs):
    """
    Plots raw ELS data in a suitable time range on the given figure and axes.

    :param figure: figure matplotlib object
    :param axes: axes matplotlib object
    :param els_data_file: path of ELS .DAT file
    :param quantity: string indicating the quantity to be extracted from the ELS object
    :param start_time: datetime object indicating the start time of data to plot
    :param end_time: datetime object indicating the end time of data to plot
    :param colorbar_range: string indicating whether to use the entire data ('full') or only the subset being plotted ('subset') for setting colorbar range.
    :param colorbar_orientation: string indicating the orientation of the colorbar.
    :param verbose: boolean indicating whether to print logging lines
    :param blur_sigma: IGNORED. Parameter sigma (in timesteps) for the Gaussian kernel.
    :param bin_selection: IGNORED. Selection of ELS bins.
    :param filter: IGNORED. Filter to be applied bin-wise after the Gaussian blur.
    :param filter_size: IGNORED. Size of the filter to be applied after the Gaussian blur.
    """

    # Validate times.
    if start_time > end_time:
        raise ValueError('Start time larger than end time.')

    # Remind the user about unused parameters.
    if len(kwargs):
        print('Parameters %s have no effect, and are only for consistency with plot_interpolated_ELS_data\'s API.' % ' '.join(['\'' + kwarg + '\'' for kwarg in kwargs]))

    # Load ELS object.
    data = ELS(els_data_file)

    # Convert dates to floats for matplotlib
    mds = mdates.date2num(data.start_date)

    # If all anodes, just use 'iso' data, for now.
    if quantity == 'anodes_all':
        quantity = 'iso'

    # D for data.
    D, scalelabel = parse_quantity(data, quantity)

    # If a datetime object, convert to a matplotlib float date.
    try:
        xmin = mdates.date2num(start_time)
        xmax = mdates.date2num(end_time)
    except AttributeError:
        xmin = start_time
        xmax = end_time

    # Check if our time range has atleast some overlap with data.
    if xmin > np.max(mds):
        raise ValueError('Start time too large - no data to plot.')

    if xmax < np.min(mds):
        raise ValueError('End time too small - no data to plot.')

    # Good (non-NaN) data from the complete ELS file.
    good_complete = ~np.isnan(D)
    gdata_complete = D[good_complete]

    # Prune to desired date range
    keep = np.where((mds >= xmin) & (mds <= xmax))[0]
    mds  = mds[keep]
    D    = D[keep,:]
    dim1_e       = data.dim1_e[keep,:]
    dim1_e_lower = data.dim1_e_lower[keep,:]
    dim1_e_upper = data.dim1_e_upper[keep,:]

    # Get x/y/z ranges based on non-NaN data ('good' data)
    good = ~np.isnan(D)
    gdata = D[good]
    xmin = np.min(mds)
    xmax = np.max(mds)
    ymin = np.min(dim1_e_lower[good])
    ymax = np.max(dim1_e_upper[good])

    # Colorbar range.
    if colorbar_range == 'full':

        # Set colorbar max and min based on the entire ELS data in this file.
        vmin = np.min(gdata_complete[gdata_complete > 0])
        vmax = np.max(gdata_complete)

    elif colorbar_range == 'subset':

        # Set colorbar max and min based on the subset being plotted.
        vmin = np.min(gdata[gdata > 0])
        vmax = np.max(gdata)

    else:
        raise ValueError('Invalid value for \'colorbar_range\'.')

    if verbose:
        print 'Data start time:', datetime.strftime(mdates.num2date(np.min(mds)), '%d-%m-%Y/%H:%M')
        print 'Data end time:', datetime.strftime(mdates.num2date(np.max(mds)), '%d-%m-%Y/%H:%M')
        
        print('Colorbar Range:')
        print('- vmin = %0.2f' % vmin)
        print('- vmax = %0.2f' % vmax)
    
    last_group = None
    rows = zip(
        mds, data.record_dur, D,
        dim1_e, dim1_e_lower, dim1_e_upper
    )
    for x, dur, drec, e, elo, ehi in rows:
        good = ~np.isnan(drec)
        if np.sum(good) == 0:
            # Skip entries with no good data
            continue

        # Sort ascending by energy level
        idx = np.argsort(e)
        good = good[idx]
        ylo = elo[idx][good]
        yhi = ehi[idx][good]

        # Pick y bin edges to be midway between
        # adjacent low/high bins (maybe this should
        # be a geometric mean instead for log-space)?
        y = np.hstack([
            [ylo[0]],
            (ylo[1:] + yhi[:-1]) / 2.0,
            [yhi[-1]],
            ])
        z = drec[idx][good]

        # Replace zero counts with small number to avoid logarithm issues
        z[z == 0] = 1e-9
        z = z.reshape((-1, 1))

        if last_group is None:
            # Start the first group
            last_group = DataGroup(x, y, z)

        elif last_group.matches(y):
            # We have the same energy levels as the last group, so
            # just add this record to it
            last_group.add(x, z)
        else:
            # We need to start a new group; first finish and plot
            # the previous group of data
            xg, yg, zg = last_group.get_data(x)
            pmax = axes.pcolormesh(xg, yg, zg, cmap='viridis',
                                   norm=LogNorm(vmin=vmin, vmax=vmax))
            last_group = DataGroup(x, y, z)

    if last_group is not None:
        # Finish the last group of data if there is one
        xg, yg, zg = last_group.get_data(mds[-1] + data.record_dur[-1])
        pmax = axes.pcolormesh(xg, yg, zg, cmap='viridis',
                               norm=LogNorm(vmin=vmin, vmax=vmax))

    axes.xaxis_date()
    axes.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y/%H:%M'))
    axes.xaxis.set_tick_params(labelsize=8)
    axes.set_xlim(xmin, xmax)
    axes.set_ylim(ymin, ymax)
    axes.set_yscale('log')
    axes.set_xlabel('Date/Time')
    axes.set_ylabel('Energy (eV/q)')

    # Tilts dates to the left for easier reading.
    plt.setp(axes.get_xticklabels(), rotation=30, ha='right')

    # Add colorbar with label.
    cbar = add_colorbar(pmax, figure, axes, colorbar_orientation)
    cbar.set_label(scalelabel)

    # Aspect ratio.
    axes.set_aspect('auto')


# Plots interpolated data from an ELS file, as obtained from get_ELS_data().
def plot_interpolated_ELS_data(figure, axes, els_data_file, quantity, start_time=datetime.min, end_time=datetime.max, colorbar_range='subset', colorbar_orientation='vertical', verbose=True, **kwargs):
    """
    Plots interpolated ELS data in a suitable time range on the given figure and axes.

    :param figure: figure matplotlib object
    :param axes: axes matplotlib object
    :param els_data_file: path of ELS .DAT file
    :param quantity: string indicating the quantity to be extracted from the ELS object
    :param start_time: datetime object indicating the start time of data to plot
    :param end_time: datetime object indicating the end time of data to plot
    :param colorbar_range: string indicating whether to use the entire data ('full') or only the subset being plotted ('subset') for setting colorbar range.
    :param colorbar_orientation: string indicating the orientation of the colorbar
    :param verbose: boolean indicating whether to print logging lines.
    :param blur_sigma: Parameter sigma (in timesteps) for the Gaussian kernel.
    :param bin_selection: Selection of ELS bins.
    :param filter: Filter to be applied bin-wise after the Gaussian blur.
    :param filter_size: Size of the filter to be applied after the Gaussian blur.
    """

    # We have to import here, because of an ImportError due to cyclic dependencies.
    from data_utils import get_ELS_data

    # Extract data.
    counts, energy_ranges, times = get_ELS_data(els_data_file, quantity, start_time, end_time, **kwargs)

    # Colorbar range.
    if colorbar_range == 'full':

        # Set colorbar max and min based on the entire *raw* ELS data in this file.
        els_object = ELS(els_data_file)
        raw_counts = parse_quantity(els_object, quantity)[0]
        raw_counts = raw_counts[~np.isnan(raw_counts)]
        vmin = np.min(raw_counts[raw_counts > 0])
        vmax = np.max(raw_counts)

    elif colorbar_range == 'subset':

        # Set colorbar max and min based on the *raw* subset being plotted.
        els_object = ELS(els_data_file)
        raw_counts = parse_quantity(els_object, quantity)[0]
    
        # If a datetime object, convert to a matplotlib float date.
        try:
            xmin = mdates.date2num(start_time)
            xmax = mdates.date2num(end_time)
        except AttributeError:
            xmin = start_time
            xmax = end_time
    
        mds = mdates.date2num(els_object.start_date)
        keep = np.where((mds >= xmin) & (mds <= xmax))[0]
        raw_counts = raw_counts[keep, :]
        raw_counts = raw_counts[~np.isnan(raw_counts)]
        vmin = np.min(raw_counts[raw_counts > 0])
        vmax = np.max(raw_counts)

    elif colorbar_range == 'interpolated_full':

        # Set colorbar max and min based on the entire *interpolated* ELS data in this file.
        all_counts = get_ELS_data(els_data_file, quantity, datetime.min, datetime.max)[0]
        vmin = np.min(all_counts[all_counts > 0])
        vmax = np.max(all_counts)

    elif colorbar_range == 'interpolated_subset':

        # Set colorbar max and min based on the *interpolated* subset being plotted.
        vmin = np.min(counts[counts > 0])
        vmax = np.max(counts)

    else:
        raise ValueError('Invalid value for \'colorbar_range\'.')

    if verbose:
        print('Colorbar Range:')
        print('- vmin = %0.2f' % vmin)
        print('- vmax = %0.2f' % vmax)

    # Plot.
    mesh = axes.pcolormesh(times, energy_ranges[0], counts.T, norm=LogNorm(vmin=vmin, vmax=vmax))

    # Add labels and ticks.
    axes.set_aspect('auto')
    axes.set_yscale('log')
    axes.set_xlabel('Date/Time')
    axes.set_ylabel('Energy (eV/q)')
    axes.xaxis_date()
    axes.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y/%H:%M'))
    axes.xaxis.set_tick_params(labelsize=8)

    # Tilts dates to the left for easier reading.
    plt.setp(axes.get_xticklabels(), rotation=30, ha='right')

    # Add colorbar with label.
    cbar = add_colorbar(mesh, figure, axes, colorbar_orientation)
    cbar.set_label('Interpolated Counts / s')


# Colours for labelled events.
def crossing_color(event_type):
    if 'BS' in event_type:
        return 'black'
    elif 'MP' in event_type:
        return 'red'
    else:
        return 'gray'


def main(els_data_file, outputfile, quantity, start_time, end_time, colorbar_range, colorbar_orientation, title, interpolated, show_labels, **kwargs):

    # Check input arguments - data file should exist.
    if not os.path.exists(els_data_file):
        raise OSError('Could not find %s.' % els_data_file)

    # Create figure and axes.
    fig, ax = plt.subplots()

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

    # Pass all parameters and plot.
    if interpolated:
        plot_interpolated_ELS_data(fig, ax, els_data_file, quantity, start_time, end_time, colorbar_range, verbose=True, **kwargs)
    else:
        plot_raw_ELS_data(fig, ax, els_data_file, quantity, start_time, end_time, colorbar_range, colorbar_orientation, verbose=True, **kwargs)

    # Add title.
    if title is not None:
        ax.set_title(title)

    # Plot the events occurring in this file.
    if show_labels:
        from compute_labelled_events import list_of_events
        from data_utils import datestring_to_float

        labels = list_of_events(os.path.basename(os.path.splitext(els_data_file)[0]), './')

        # How large is the width of the rectangle around each labelled event?
        days_per_minute = 1/(24 * 60)
        window_size = 1*days_per_minute

        # Annotate plot with labelled events.
        print 'Labelled events:'
        for label_type, crossing_timestring in labels:
            print '- Event at %s of type %s.' % (crossing_timestring, label_type)
            crossing_time = datestring_to_float(crossing_timestring)
            ax.axvspan(crossing_time - window_size/2, crossing_time + window_size/2, facecolor=crossing_color(label_type), alpha=1)

    # Save to file if given.
    if outputfile is None:
        plt.show()
    else:
        plt.savefig(outputfile, bbox_inches='tight')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)

    parser.add_argument('els_data_file', help='ELS DAT file.')
    parser.add_argument('-o', '--outputfile', default=None)
    parser.add_argument('-q', '--quantity', default='anode5', choices=(
        ['iso'] +
        [('anode%d' % a) for a in range(1, 9)] +
        [('def%d'   % a) for a in range(1, 9)] +
        [('dnf%d'   % a) for a in range(1, 9)] +
        [('psd%d'   % a) for a in range(1, 9)]
    ))
    parser.add_argument('-cbr', '--colorbar_range', default='subset', choices=['full', 'subset', 'interpolated_full', 'interpolated_subset'],
                        help='Whether to use the entire raw data (\'full\'), entire interpolated data (\'interpolated_full\'), only the raw subset being plotted (\'subset\'), or only the interpolated subset being plotted (\'interpolated_subset\') for setting the colorbar range. To use the \'interpolated_subset\' and \'interpolated_full\' options, \'--interpolated\' must be passed.')
    parser.add_argument('-cbo', '--colorbar_orientation', default='vertical', choices=['vertical', 'horizontal'],
                        help='Orientation of the colorbar for the plot.')
    parser.add_argument('-st', '--start_time', default=None,
                        help='Start time in dd-mm-yyyy/HH:MM. Restricts data to those recorded on or after this time.')
    parser.add_argument('-et', '--end_time', default=None,
                        help='End time in dd-mm-yyyy/HH:MM. Restricts data to those recorded upto and including this time.')
    parser.add_argument('--show_labels', action='store_true', default=False,
                    help='Whether to show labels on the plot.')
    parser.add_argument('--title', default=None, type=str,
                        help='Title for plot.')
    parser.add_argument('--interpolated', action='store_true', default=False,
                        help='Whether to use interpolated or raw ELS data.')
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
