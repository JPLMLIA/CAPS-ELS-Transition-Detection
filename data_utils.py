"""
A bunch of utility functions used by the algorithmic pipelines in this directory.
"""

from __future__ import division
import os
from datetime import datetime
import warnings
import math
import dateutil.parser
import numpy as np
import matplotlib.dates as mdates
import scipy
from scipy.ndimage import minimum_filter
from scipy.linalg import hankel
from scipy.ndimage.filters import gaussian_filter, uniform_filter, median_filter
from scipy.spatial.distance import pdist
from simanneal import Annealer
from joblib import Memory
import pandas as pd
from pandas.plotting import register_matplotlib_converters
from mpl_toolkits.axes_grid1 import make_axes_locatable

from els_data import ELS
from plot_els import parse_quantity
from transform import Filter

# Required for pandas to be compatible with matplotlib when formatting dates.
register_matplotlib_converters()


# Caches the return values of a function in a specific directory.
def cache(directory):
    def cache_in_dir(func):
        memory = Memory(directory, verbose=0)
        return memory.cache(func)
    return cache_in_dir


# Returns the total time contained within the list of intervals.
def get_total_time(intervals):
    """
    >>> get_total_time([])
    0.0
    >>> get_total_time([(1, 3)])
    2.0
    >>> get_total_time([(1, 3), (4, 5)])
    3.0
    """

    return math.fsum([(interval_end - interval_start) for (interval_start, interval_end) in intervals])


# Returns the list of overlap regions between two sorted (according to start time) lists of intervals (as 2-tuples).
# Note that the lists should not contain any overlaps within themselves.
def get_overlaps(first_intervals, second_intervals):
    """
    >>> get_overlaps([(1, 2), (3, 4), (8, 9)], [(1, 4), (7, 8.5)])
    [(1, 2), (3, 4), (8, 8.5)]
    >>> get_overlaps([(1, 4), (7, 8.5)], [(1, 2), (3, 4), (8, 9)])
    [(1, 2), (3, 4), (8, 8.5)]
    >>> get_overlaps([(1, 8), (9, 10)], [(2, 3), (5, 6), (7, 9.5)])
    [(2, 3), (5, 6), (7, 8), (9, 9.5)]
    >>> get_overlaps([(2, 3), (5, 6), (7, 9.5)], [(1, 8), (9, 10)])
    [(2, 3), (5, 6), (7, 8), (9, 9.5)]
    >>> get_overlaps([(1, 10)], [(0, 5)])
    [(1, 5)]
    >>> get_overlaps([(0, 5)], [(1, 10)])
    [(1, 5)]
    >>> get_overlaps([(1, 6), (7, 9)], [(5.5, 7.5)])
    [(5.5, 6), (7, 7.5)]
    >>> get_overlaps([(5.5, 7.5)], [(1, 6), (7, 9)])
    [(5.5, 6), (7, 7.5)]
    """

    overlaps = []
    for first_interval in first_intervals:
        # Find the index of the first interval in the second list starting after this interval ends.
        # We do not need to search beyond this interval.
        end_index = None
        for index, second_interval in enumerate(second_intervals):
            if second_interval[0] >= first_interval[1]:
                end_index = index
                break

        # Go through all of these intervals and compute the overlaps.
        for second_interval in second_intervals[:end_index]:
            if second_interval[1] > first_interval[0]:
                uncovered_region = (max(first_interval[0], second_interval[0]), min(first_interval[1], second_interval[1]))
                overlaps.append(uncovered_region)

    return overlaps


# Returns the intervals within the interval [start_time, end_time] which are not covered by the list of intervals given.
# The list of intervals should be sorted according to their start times.
def get_uncovered(intervals, start_time, end_time):
    """
    >>> get_uncovered([(1, 3)], 0, 10)
    [(0, 1), (3, 10)]
    >>> get_uncovered([(1, 8), (9, 10)], 0, 20)
    [(0, 1), (8, 9), (10, 20)]
    >>> get_uncovered([], 0, 20)
    [(0, 20)]
    >>> get_uncovered([(1, 3), (3, 6)], 0, 10)
    [(0, 1), (6, 10)]
    """

    uncovered_intervals = []
    curr_start = start_time

    # Go through the list.
    for interval in intervals:
        curr_end = interval[0]

        # We don't add degenerate intervals.
        if curr_start < curr_end:
            uncovered_intervals.append((curr_start, curr_end))

        curr_start = interval[1]

    # If there's still time left!
    if curr_start < end_time:
        uncovered_intervals.append((curr_start, end_time))

    return uncovered_intervals


# Merges adjacent intervals passed as a list of 2-tuples.
# The list of intervals must be sorted according to start-time and disjoint.
def merge_adjacent_intervals(intervals):
    """
    >>> merge_adjacent_intervals([(1, 3), (4, 5)])
    [(1, 3), (4, 5)]
    >>> merge_adjacent_intervals([(1, 4), (4, 5)])
    [(1, 5)]
    >>> merge_adjacent_intervals([(1, 2), (2, 5), (5, 7)])
    [(1, 7)]
    >>> merge_adjacent_intervals([(1, 2), (2, 5), (5, 7), (8, 9)])
    [(1, 7), (8, 9)]
    >>> merge_adjacent_intervals([(1, 2), (2, 5), (6, 7), (7, 9), (10, 11), (13, 14)])
    [(1, 5), (6, 9), (10, 11), (13, 14)]
    >>> merge_adjacent_intervals([])
    []
    >>> merge_adjacent_intervals([(0, 1)])
    [(0, 1)]
    >>> merge_adjacent_intervals([(0, 1), (1, 8)])
    [(0, 8)]
    """

    merged_intervals = []

    # Iterate once through list, and merge greedily.
    index = 0
    while index < len(intervals):
        curr_interval = intervals[index]
        curr_start = curr_interval[0]
        curr_end = curr_interval[1]

        # See how far we can go on merging intervals.
        next_index = index + 1
        while next_index < len(intervals):
            next_interval = intervals[next_index]

            if next_interval[0] == curr_end:
                curr_end = next_interval[1]
            else:
                break

            next_index += 1

        merged_intervals.append((curr_start, curr_end))
        index = next_index

    return merged_intervals


# Returns non-overlapping intervals as a list of 2-tuples indicating start and end times for each interval.
# The list of interval centers must be sorted.
def mark_intervals(interval_centers, window, start_time, end_time):
    """
    >>> mark_intervals([1, 4, 6], 2, 0, 10)
    [(0.0, 2.0), (3.0, 7.0)]
    >>> mark_intervals([1, 4, 6], 1, 0, 10)
    [(0.5, 1.5), (3.5, 4.5), (5.5, 6.5)]
    >>> mark_intervals([1, 4, 6], 3, 0, 10)
    [(0, 7.5)]
    >>> mark_intervals([], 3, 0, 10)
    []
    >>> mark_intervals([2], 5, 0, 2)
    [(0, 2)]
    >>> mark_intervals([4, 5], 3, 0, 10)
    [(2.5, 6.5)]
    >>> mark_intervals([4, 5, 8], 3, 0, 9)
    [(2.5, 9)]
    """

    # Variables indicating the start and end times for the next interval to be added.
    interval_start = -np.inf
    interval_end = -np.inf

    interval_time = 0
    intervals = []
    for center in interval_centers:
        interval_start = max(center - window / 2, interval_end, start_time)
        interval_end = min(center + window / 2, end_time)
        interval_time += interval_end - interval_start
        intervals.append((interval_start, interval_end))

    # Merge adjacent intervals.
    return merge_adjacent_intervals(intervals)


# Converts an array to a dictionary of intervals, with key as the value of the array.
def array_to_intervals(array):
    """
    >>> array_to_intervals([2, 1, 1, 1, 3, 3, 4, 4, 1, 1, 2, 2])[1]
    [(1, 4), (8, 10)]
    >>> array_to_intervals([2, 1, 1, 1, 3, 3, 4, 4, 1, 1, 2, 2])[2]
    [(0, 1), (10, 11)]
    >>> array_to_intervals([2, 1, 1, 1, 3, 3, 4, 4, 1, 1, 2, 2])[3]
    [(4, 6)]
    >>> array_to_intervals([2, 1, 1, 1, 3, 3, 4, 4, 1, 1, 2, 2])[4]
    [(6, 8)]
    >>> array_to_intervals([])
    {}
    """

    if len(array) == 0:
        return {}

    interval_dict = {}
    last_val = array[0]
    interval_start = 0
    for index, val in enumerate(array):

        # Assign a list of intervals to this value.
        if val not in interval_dict:
            interval_dict[val] = []

        # Check if we have finished an interval.
        if val != last_val:
            interval_dict[last_val].append((interval_start, index))
            interval_start = index
            last_val = val

    # Assign the last interval.
    if interval_start != len(array) - 1:
        interval_dict[last_val].append((interval_start, len(array) - 1))

    return interval_dict


# Returns the leftmost index in a sorted 1D array of the closest value to the given value.
def closest_index(val, array):

    # Restrict array via binary search.
    low = 0
    high = array.size

    while high - low > 2:
        mid = (high + low) // 2

        if array[mid] < val:
            low = mid
        else:
            high = mid

    # Now, search within the restricted array.
    return low + np.argmin(np.abs(array[low: high] - val))


# Returns the indices of the peaks (local maxima) in the array, in a neighbourhood.
def peaks(array, neighbourhood=1):
    """
    >>> peaks([1, 3, 2])
    [1]
    >>> peaks([1])
    [0]
    >>> peaks([3, 2, 1])
    [0]
    >>> peaks([])
    []
    >>> peaks([1, 0, 2, 1])
    [0, 2]
    """
    # Trivial cases.
    if len(array) == 0:
        return []

    if len(array) == 1:
        return [0]

    peak_indices = []

    # Check each.
    for index, val in enumerate(array):

        # Check if values masked.
        if np.ma.is_masked(val):
            continue

        if index == 0:
            prev_vals = -np.inf
        else:
            prev_vals = np.max(array[max(0, index - neighbourhood): index])

        if index == len(array) - 1:
            next_vals = -np.inf
        else:
            next_vals = np.max(array[index + 1: index + neighbourhood + 1])

        if prev_vals <= val and next_vals < val:
            peak_indices.append(index)

    return peak_indices

# Returns an array, indicating for each element in the first array, the L1-distance to the closest element in the second array.
# If 'return_closest' is true, returns a boolean array indicating for each element in the second array, whether it is the closest element to atleast one element in the first
def closest_distance(array1, array2, return_closest=False):
    """
    >>> '%s' % closest_distance([1, 3, 4], [1, 3, 4])
    '[0. 0. 0.]'
    >>> '%s, %s' % closest_distance([1, 3, 4], [1, 3, 4], return_closest=True)
    '[0. 0. 0.], [ True  True  True]'
    >>> '%s' % closest_distance([1, 3, 4], [1, 3, 5])
    '[0. 0. 1.]'
    >>> '%s, %s' % closest_distance([1, 3, 4], [1, 3, 5], return_closest=True)
    '[0. 0. 1.], [ True  True  True]'
    >>> '%s' % closest_distance([5, 3, 4], [1, 3, 5])
    '[0. 0. 1.]'
    >>> '%s, %s' % closest_distance([5, 3, 4], [1, 3, 5], return_closest=True)
    '[0. 0. 1.], [False  True  True]'
    >>> '%s' % closest_distance([10, 3, 4], [1, 3, 5])
    '[5. 0. 1.]'
    >>> '%s, %s' % closest_distance([10, 3, 4], [1, 3, 5], return_closest=True)
    '[5. 0. 1.], [False  True  True]'
    >>> '%s' % closest_distance([10, -2, 4], [1, 3, 5])
    '[5. 3. 1.]'
    >>> '%s, %s' % closest_distance([10, -2, 4], [1, 3, 5], return_closest=True)
    '[5. 3. 1.], [ True  True  True]'
    >>> '%s' % closest_distance([10, -2, 4], [-100, 3])
    '[7. 5. 1.]'
    >>> '%s, %s' % closest_distance([10, -2, 4], [-100, 3], return_closest=True)
    '[7. 5. 1.], [False  True]'
    """

    distances = np.full(len(array1), np.inf)
    for index1, element1 in enumerate(array1):
        for element2 in array2:
            distances[index1] = np.minimum(distances[index1], np.sum(np.abs(element1 - element2)))

    is_closest = np.full(len(array2), False)
    if return_closest:
        for index2, element2 in enumerate(array2):
            for index1, element1 in enumerate(array1):
                if distances[index1] == np.sum(np.abs(element1 - element2)):
                    is_closest[index2] = True
        return distances, is_closest
    else:
        return distances


# Returns the index and the value of the first value in the sorted array which is greater than or equal to val.
def binary_search(val, array):
    """
    >>> binary_search(1, [1, 2, 3])
    (0, 1)
    >>> binary_search(1.5, [1, 2, 3])
    (1, 2)
    >>> binary_search(8, [1, 2, 3])
    (None, None)
    >>> binary_search(3, [1, 2, 4])
    (2, 4)
    """

    # Trivial cases - out of array range.
    if len(array) == 0:
        return None, None

    if val > array[-1]:
        return None, None

    if val <= array[0]:
        return 0, array[0]

    low = 0
    high = len(array) - 1

    while high - low >= 2:
        mid = (high + low) // 2

        if array[mid] >= val:
            high = mid
        else:
            low = mid + 1

    if low == high:
        return low, array[low]

    if val > array[low]:
        return high, array[high]
    else:
        return low, array[low]


# Artificial one-dimensional time-series with 2 discords.
def generate_1D_timeseries(size):
    return np.hstack((np.sin(np.arange(0, (size + 1)/3)), 3 + np.sin(np.arange(0, (size + 1)/3)), np.sin(np.arange(0, (size + 1)/3))))[:size]


# Artificial n-dimensional time-series with 2 discords.
def generate_nD_timeseries(shape, discord_dimensions=None):
    series = np.zeros(shape)

    sublength = shape[0]//3
    width = shape[1]

    series[:sublength] = np.ones((sublength, width)) + np.random.rand(sublength, width)/10
    series[sublength: 2*sublength, :discord_dimensions] = 2 * np.ones((sublength, discord_dimensions)) + np.random.rand(sublength, discord_dimensions)/10
    series[sublength: 2*sublength, discord_dimensions:] = np.ones((sublength, width - discord_dimensions)) + np.random.rand(sublength, width - discord_dimensions)/10
    series[2*sublength:] = np.ones((sublength, width)) + np.random.rand(sublength, width)/10

    return series


# Artificial 2-dimensional time-series to be segmented.
def generate_2D_timeseries(size):
    series1 = np.random.rand(size)
    series2 = np.hstack((np.random.rand((size + 1)//3), 2 + np.random.rand((size + 1)//3), 4 + np.random.rand((size + 1)//3)))[:size]
    return np.vstack((series1, series2)).T


# Makes a stack of sequences (called a subsequence) from the original sequence.
def pack(sequence, k=10):
    """
    >>> pack(np.array([1, 2, 3]), k=2)
    array([[1, 2],
           [2, 3]])
    >>> pack(np.array([1, 2, 3, 4]), k=2)
    array([[1, 2],
           [2, 3],
           [3, 4]])
    >>> a = np.array([[1,2,3], [4, 5, 6], [7, 8, 9], [11, 12, 13], [14, 15, 16]])
    >>> a
    array([[ 1,  2,  3],
           [ 4,  5,  6],
           [ 7,  8,  9],
           [11, 12, 13],
           [14, 15, 16]])
    >>> pack(a, k=2)
    array([[ 1,  2,  3,  4,  5,  6],
           [ 4,  5,  6,  7,  8,  9],
           [ 7,  8,  9, 11, 12, 13],
           [11, 12, 13, 14, 15, 16]])
    """

    if len(sequence.shape) == 1:
        # Reference: https://stackoverflow.com/questions/22970716/get-all-subsequences-of-a-numpy-array
        # Pass first column, and last row of Hankel matrix to be constructed.
        return hankel(sequence[:k], sequence[k - 1:]).T[:sequence.shape[0] - k + 1]
    else:
        # Not as efficient as above, but works correctly for multi-dimensional sequences.
        subsequences = np.array([sequence[i: i + k].flatten() for i in range(0, sequence.shape[0] - k + 1)])
        return np.stack(arrays=subsequences)


# Returns the unique values with the highest frequency, ie, from ranks 1 to num_rank according to frequency.
def get_top_counts(sequence, num_rank=2):
    """
    >>> '%s' % get_top_counts([1, 2, 1, 1, 1, 3, 3, 4, 4, 4], num_rank=2)
    '[1 4]'
    >>> '%s' % get_top_counts([1, 2, 1, 1, 1, 3, 3, 4, 4, 4], num_rank=3)
    '[1 4 3]'
    """
    unique, counts = np.unique(sequence, return_counts=True)
    array_with_counts = np.asarray((unique, counts)).T
    return np.array(sorted(array_with_counts, key=lambda x: -x[1]))[:num_rank, 0]


# Returns a numpy array of all pair-wise Euclidean squared distances between points in this finite sequence, after flattening if required.
def squared_distances(sequence):
    """
    >>> squared_distances(np.array([[1, 2], [2, 3], [3, 4]]))
    array([2., 8., 2.])
    """
    return pdist(sequence.reshape(sequence.shape[0], -1), metric='sqeuclidean')


# Returns the median of all pair-wise Euclidean squared distances between points in this finite sequence.
def get_median_pairwise_distance(sequence):
    """
    >>> '%.2f' % get_median_pairwise_distance(np.array([[1, 2], [2, 3], [3, 4]]))
    '1.41'
    """
    return np.sqrt(np.ma.median(squared_distances(sequence)))


# Extracts the required quantity from the Intel dataset. Results are cached.
@cache('temp/cachedir/')
def get_Intel_data(intel_data_file, quantity, start_time, end_time, drop_sensors=None, downsample_rate='20min'):
    """
    :param intel_data_file: The path of the ELS data file. Note that headers must be present in the same directory as well.
    :param quantity: The quantity to be extracted.
    :param start_time: Start time (as a datetime.datetime object) for readings.
    :param end_time: End time (as a datetime.datetime object) for readings.
    :param drop_sensors: List of sensors to ignore.
    :param downsample_rate: The size of the time bins to downsample into, taking averages.
    :return: 3-tuple of (quantities, sensor_ids, times)
    """

    # Check file paths.
    if not os.path.exists(intel_data_file):
        raise OSError('Could not find %s.' % intel_data_file)

    # Column names!
    names = ['date', 'time', 'epoch', 'sensor_id', 'temperature', 'humidity', 'light', 'voltage']

    if quantity not in names:
        raise ValueError('Invalid quantity passed. Choose from \'temperature\', \'humidity\', \'light\', \'voltage\'.')

    # To parse the date and time columns into a single datetime object.
    def dateparse(date, time):
        try:
            return datetime.strptime(date + " " + time, '%Y-%m-%d %H:%M:%S.%f')
        except ValueError:
            return datetime.strptime(date + " " + time, '%Y-%m-%d %H:%M:%S')

    # Construct a dataframe from the input CSV file.
    df = pd.read_csv(intel_data_file, sep=' ', parse_dates={'datetime': ['date', 'time']}, date_parser=dateparse, names=names)

    # Restrict to the column we want, restricting to valid sensors.
    df = df[df['sensor_id'] <= 54][['sensor_id', 'datetime', quantity]]

    # Filter values to those between start and end times.
    df = df[(start_time <= df['datetime']) & (df['datetime'] <= end_time)]

    # Remove sensors if required.
    if drop_sensors is not None:
        df = df[~df['sensor_id'].isin(drop_sensors)]

    # Create an index on datetime to downsample.
    df.set_index(['datetime'], inplace=True)

    # Downsample!
    df = df.groupby('sensor_id').resample(downsample_rate)[quantity].mean().unstack('sensor_id', fill_value=0)

    # Interpolate linearly across sensors independently.
    df.interpolate(method='linear', inplace=True)

    # Extract the relevant quantities.
    quantities = df.to_numpy()
    sensor_ids = df.columns.to_numpy()
    times = df.index.to_series().apply(mdates.date2num).to_numpy()

    return quantities, sensor_ids, times


# Plots Intel data.
def plot_Intel_data(fig, ax, intel_data, colorbar_orientation='vertical', tick_spacing=5):

    # Unpack data.
    quantities, sensor_ids, times = intel_data

    # Set labels.
    ax.set_ylabel('Sensor ID')
    ax.set_xlabel('Time')

    # Plot as a colourmap.
    im = ax.imshow(quantities.T, aspect='auto', origin='lower',
                   interpolation='none', extent=[times[0], times[-1], 0, len(sensor_ids)])

    # Set ticks on the y-axis.
    ax.set_yticks(np.arange(len(sensor_ids), step=tick_spacing))
    ax.set_yticklabels(sensor_ids[::tick_spacing])

    # Time on the x-axis has to be formatted correctly.
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y/%H:%M'))
    fig.autofmt_xdate()

    # Label size to keep the font not too big,
    ax.tick_params(axis='both', which='major', labelsize=8)

    # Add a colorbar for the temperature.
    if colorbar_orientation == 'horizontal':
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('top', size='5%', pad=0.15)
        cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
        cbar.ax.xaxis.set_label_position('top')
        cbar.ax.xaxis.set_ticks_position('top')
    else:
        cbar = fig.colorbar(im, ax=ax, orientation=colorbar_orientation)

    cbar.set_label('Temperature')


# Returns the anomalies (labelled as 1) in the Intel dataset.
def get_Intel_anomalies(quantities, times):
    sensor_means = np.mean(quantities, axis=0)
    sensor_devs = np.std(quantities, axis=0)

    lower_threshold = np.tile(sensor_means - 3 * sensor_devs, (len(quantities), 1))
    upper_threshold = np.tile(sensor_means + 3 * sensor_devs, (len(quantities), 1))

    labels = np.ones(len(quantities))
    labels[np.all(np.logical_and(lower_threshold <= quantities, quantities <= upper_threshold), axis=1)] = 0

    return times[labels == 1]


# Extracts the data from a single file (ie, one sensor) from the Yahoo dataset. Results are cached.
def get_Yahoo_record(yahoo_data_file, quantities):

    # Check file paths.
    if not os.path.exists(yahoo_data_file):
        raise OSError('Could not find %s.' % yahoo_data_file)

    # Construct a DataFrame from the input CSV file.
    df = pd.read_csv(yahoo_data_file, sep=',')

    # Convert UNIX timestamps to actual dates.
    df['datetime'] = pd.to_datetime(df.pop('timestamps'), unit='s')

    # Return only the required columns with datetimes.
    return df[['datetime'] + quantities]


# Loads the Yahoo dataset from combining all files in a directory into a single DataFrame.
def load_Yahoo_as_df(directory):

    # Check file paths.
    if not os.path.exists(directory):
        raise OSError('Could not find %s.' % directory)

    # Combine all sensor readings into one big Dataframe.
    df = None
    for index, filename in enumerate(os.listdir(directory)):

        # Actual path of the file.
        filepath = directory + filename

        # Check whether this file has the required columns.
        columns = pd.read_csv(filepath, sep=',').columns
        if 'value' in columns and 'anomaly' in columns:

            sensor_id = filename[14:-4]

            # Get one sensor's reading.
            df_sensor = get_Yahoo_record(filepath, ['value', 'anomaly'])

            # Create first Dataframe, if nothing so far.
            if df is None:
                df = df_sensor.rename(columns={'value': sensor_id})
            else:
                if np.any(df['datetime'] != df_sensor['datetime']):
                    raise ValueError('Join between non-matching dates.')

                df[sensor_id] = df_sensor['value']
                df['anomaly'] = df['anomaly'] | df_sensor['anomaly']

    return df


# Extracts the data present in this directory as the Yahoo dataset. Results are cached.
@cache('temp/cachedir/')
def get_Yahoo_data(directory):
    df = load_Yahoo_as_df(directory)
    times = df['datetime'].apply(mdates.date2num).to_numpy()
    df.drop(['datetime', 'anomaly'], axis=1, inplace=True)
    values = df.to_numpy()
    sensor_ids = df.columns.to_numpy()
    return values, sensor_ids, times


# Returns the times marked as anomalies (label 1), by taking the logical OR over individual sensor anomaly flags.
def get_Yahoo_anomalies(directory):
    df = load_Yahoo_as_df(directory)
    labels = df['anomaly'].to_numpy()
    times = df['datetime'].apply(mdates.date2num).to_numpy()
    return times[labels == 1]


# Cannot handle missing data! Use get_ELS_data() below instead.
# Extracts the quantity with no interpolation across any dimensions from the given ELS .DAT file. Results are cached.
@cache('temp/cachedir/')
def get_ELS_data_no_interpolation(els_data_file, quantity, start_time, end_time):
    """
    :param els_data_file: The path of the ELS data file. Note that headers must be present in the same directory as well.
    :param quantity: The quantity to be extracted.
    :param start_time: Start time (as a datetime.datetime object) for readings.
    :param end_time: End time (as a datetime.datetime object) for readings.
    :return: 3-tuple of (counts, energy_range, times)
    """
    # Check input arguments - data file should exist.
    if not os.path.exists(els_data_file):
        raise OSError('Could not find %s.' % els_data_file)

    if start_time > end_time:
        raise ValueError('Start time larger than end time.')

    # If we have to get all anodes, we will have to jump in and out to get all anodes.
    if quantity == 'anodes_all':
        return get_all_anode_ELS_data(els_data_file, start_time, end_time)

    data = ELS(els_data_file)

    # Convert dates to floats for matplotlib
    mds = mdates.date2num(data.start_date)

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
        raise ValueError('Start time after any data.')

    if xmax < np.min(mds):
        raise ValueError('End time before any data.')

    # Prune to desired date range.
    keep = np.where((mds >= xmin) & (mds <= xmax))[0]

    mds = mds[keep]
    D = D[keep[:, None], :]
    data.dim1_e = data.dim1_e[keep[:, None], :]
    data.dim1_e_upper = data.dim1_e_upper[keep[:, None], :]
    data.dim1_e_lower = data.dim1_e_lower[keep[:, None], :]

    print 'Data start time:', datetime.strftime(mdates.num2date(np.min(mds)), '%d-%m-%Y/%H:%M')
    print 'Data end time:', datetime.strftime(mdates.num2date(np.max(mds)), '%d-%m-%Y/%H:%M')

    if not (len(mds) == len(data.dim1_e) == len(D)):
        raise ValueError('Invalid number of columns.')

    counts = D
    energy_ranges = data.dim1_e
    times = mds

    # Squeeze out superfluous dimensions.
    counts, energy_ranges, times = np.squeeze(counts), np.squeeze(energy_ranges), np.squeeze(times)

    return counts, energy_ranges, times

# Extracts ELS data from all anodes.
def get_all_anode_ELS_data(els_data_file, start_time, end_time):
    samples, energy_ranges, times = zip(*[get_ELS_data(els_data_file, 'anode' + str(anode_number), start_time, end_time) for anode_number in range(1, 9)])

    # Validate times.
    times = np.array(times)
    if not np.allclose(np.tile(times[0], (8, 1)), times, 0.00000001):
        raise ValueError('Invalid times.')

    # Validate energy ranges.
    energy_ranges = np.array(energy_ranges)
    if not np.allclose(np.tile(energy_ranges[0], (8, 1)), energy_ranges):
        raise ValueError('Invalid energy ranges.')

    # Stack up counts.
    samples = np.hstack(samples)

    # Return just one of the energy ranges and times, since they're all the same.
    return samples, energy_ranges[0], times[0]


# Extracts the quantity from the given ELS .DAT file. Results are cached.
@cache('temp/cachedir/')
def get_ELS_data(els_data_file, quantity, start_time, end_time, blur_sigma=0, bin_selection='all', filter='no_filter', filter_size=1):
    """

    :param els_data_file: The path of the ELS data file. Note that the .LBL file must be present in the same directory.
    :param quantity: The quantity to be extracted.
    :param start_time: Start time (as a datetime.datetime object/matplotlib float date) for readings.
    :param end_time: End time (as a datetime.datetime object/matplotlib float date) for readings.
    :param blur_sigma: Parameter sigma (in timesteps) for the Gaussian kernel.
    :param bin_selection: Selection of ELS bins.
    :param filter: Filter to be applied bin-wise after the Gaussian blur.
    :param filter_size: Size of the filter to be applied after the Gaussian blur.
    :return: 3-tuple of (counts, energy_ranges, times)
    """
    # Obtain the raw counts.
    counts, energy_ranges, times = get_ELS_data_no_interpolation(els_data_file, quantity, start_time, end_time)

    # The common energy range, fixed across all files. These are the bin centres of the 32-bin timesteps in the original CAPS ELS data.
    common_energy_range = np.array([
        2.39710098e+04, 1.75067754e+04, 1.27858037e+04, 9.34583984e+03,
        6.82949463e+03, 4.98947949e+03, 3.64505884e+03, 2.66262939e+03,
        1.94540930e+03, 1.42190784e+03, 1.03906091e+03, 7.59045593e+02,
        5.54588379e+02, 4.04940857e+02, 2.96158539e+02, 2.16495728e+02,
        1.58241898e+02, 1.15493149e+02, 8.43917389e+01, 6.18861465e+01,
        4.50986481e+01, 3.29373093e+01, 2.40994759e+01, 1.76704102e+01,
        1.27102909e+01, 9.25298405e+00, 6.92527056e+00, 4.90834713e+00,
        3.74522614e+00, 2.58445168e+00, 1.41556251e+00, 5.79999983e-01,
    ])

    # Rebin counts at each time-step.
    new_counts = rebin_to_common_range(counts, energy_ranges, common_energy_range)

    # Interpolate (and resample) counts across each bin independently as a function of time.
    new_counts, new_times = interpolate_timesteps(new_counts, times, time_resolution_s=2)

    # We might have negative values after interpolation. Clip these to 0, so that they make physical sense.
    new_counts[new_counts < 0] = 0

    # Smooth along time dimension.
    new_counts = gaussian_blur(new_counts, blur_sigma)

    # If we have to ignore the unpaired bin, remove it now.
    if bin_selection == 'ignore_unpaired':
        new_counts = new_counts[:, :-1]
        common_energy_range = common_energy_range[:-1]
    elif bin_selection == 'center':
        new_counts = new_counts[:, 15:-7]
        common_energy_range = common_energy_range[15:-7]

    # Apply the filter.
    new_counts = Filter(filter, filter_size).filter(new_counts)

    # Since we have the same energy ranges at each timestep, we repeat the common energy range.
    new_energy_ranges = np.repeat(common_energy_range[:, np.newaxis], new_counts.shape[0], axis=1).T

    # Print bin-wise statistics.
    for bin_dimension in range(new_counts.shape[1]):
        bin_counts = new_counts[:, bin_dimension]
        valid_indices = ~np.isnan(bin_counts)

        bin_mean = np.mean(bin_counts[valid_indices])
        bin_std = np.std(bin_counts[valid_indices])
        print 'Bin %d: Mean = %0.2f, Standard Dev. = %0.2f' % (bin_dimension, bin_mean, bin_std)

    # The new time-series has a common energy range of 32 bins across all timesteps, and is regularly sampled.
    return new_counts, new_energy_ranges, new_times


# Interpolate the counts as a function of the energy range at each time-step, with linear spline interpolation.
def rebin_to_common_range(counts, energy_ranges, common_energy_range):
    
    # Rebin at each time-step using valid (not 'NaN') entries.
    new_counts = np.full((counts.shape[0], common_energy_range.shape[0]), np.nan)
    for index, (timestep_counts, timestep_energy_range) in enumerate(zip(counts, energy_ranges)):
        valid_bins = ~np.isnan(timestep_energy_range)
        
        # How many valid counts do we have?
        num_valid_bins = np.sum(valid_bins)
        corresponding_counts = timestep_counts[valid_bins]

        # If we have 32 bins, keep them as is.
        if num_valid_bins == 32:
            new_counts[index] = corresponding_counts

        # If we have 63 bins, combine all the adjacent bins, except the last, which is as is.
        # Note that the last bin is the one with the lowest mean energy.
        elif num_valid_bins == 63:
            new_counts[index, :-1] = (corresponding_counts[0:-1:2] + corresponding_counts[1:-1:2])/2
            new_counts[index, -1] = corresponding_counts[-1]

        # Otherwise, we'll fill this timestep in later.
        else:
            pass

    return new_counts


# Interpolate (and resample) counts across each bin independently as a function of time, with linear spline interpolation.
def interpolate_timesteps(counts, times, time_resolution_s=2):
    time_resolution_days = time_resolution_s / (60 * 60 * 24)
    resampled_times = np.arange(times[0], times[-1], time_resolution_days)
    resampled_counts = np.zeros((resampled_times.shape[0], counts.shape[1]))

    # Rebin along each dimension using valid (not 'NaN') entries.
    for bin_dimension in range(counts.shape[1]):
        bin_counts = counts[:, bin_dimension]
        valid_indices = ~np.isnan(bin_counts)
        valid_counts = bin_counts[valid_indices]
        valid_times = times[valid_indices]

        interpolated_counts_function = scipy.interpolate.interp1d(valid_times, valid_counts, kind='slinear', fill_value='extrapolate', assume_sorted=True)
        resampled_counts[:, bin_dimension] = interpolated_counts_function(resampled_times)
   
    return resampled_counts, resampled_times


# Outdated. No longer used in get_ELS_data().
# Fill in missing timesteps with extra entries, so that the time-series appears regularly sampled.
def interpolate_timesteps_duplication(counts, energy_ranges, times):

    # Obtain the time-resolution of sampling.
    time_differences = np.diff(times)
    time_resolution = np.min(time_differences)

    new_times = times
    new_counts = counts
    new_energy_ranges = energy_ranges

    inserted = 0

    # Add in extra entries wherever we have undersampled - that is, whenever we have a timestamp difference >= 2 * minimum timestamp difference (time resolution).
    for index in np.where(time_differences >= 2 * time_resolution)[0]:

        # Fill in samples between this timestamp and the next timestamp at steps with size equal to the time resolution.
        for new_index, timestep in enumerate(np.arange(times[index] + time_resolution, times[index + 1] - time_resolution, time_resolution), start=index + inserted + 1):
            new_times = np.insert(new_times, new_index, timestep)
            new_counts = np.insert(new_counts, new_index, counts[index], axis=0)
            new_energy_ranges = np.insert(new_energy_ranges, new_index, energy_ranges[index], axis=0)
            inserted += 1

    return new_counts, new_energy_ranges, new_times


# Takes the average of the energies with the num_rank highest counts at each timestep.
def ranked_average_energy(counts, energies, num_rank=5):
    """
    :param counts: counts corresponding to energies
    :param energies: energy values
    :return: numpy array of average of energies chosen

    >>> counts = np.array([[1, 4, 1, 5, 2], [5, 4, 1, 2, 1]])
    >>> energies = np.array([1, 2, 3, 4, 5])
    >>> ranked_average_energy(counts, energies, 2)
    array([3. , 1.5])
    >>> ranked_average_energy(counts, energies, 4)
    array([3.5, 3. ])
    """
    # Select the indexes with the 'num_rank' highest counts.
    indices = np.argsort(counts, axis=1)[:, -num_rank:]

    # Select the energies corresponding to these indices.
    energies_full = np.vstack([energies] * counts.shape[0])
    energies_selected = energies_full[np.arange(energies_full.shape[0])[:, None], indices]

    # Take the average of these energies.
    return np.average(energies_selected, axis=1)


# Takes the average energy weighted according to counts.
def weighted_average_energy(counts, energies):
    """
    :param counts: counts corresponding to energies
    :param energies: energy values
    :return: numpy array of average of energies chosen

    >>> counts = np.array([[1, 4, 1, 5, 2], [5, 4, 1, 2, 1]])
    >>> energies = np.array([1, 2, 3, 4, 5])
    >>> weighted_average_energy(counts, energies)
    array([3.23076923, 2.23076923])
    """

    # Multiply energies by counts to weight.
    energies_counts_product = np.multiply(energies, counts)

    # Take the average of these energies.
    return np.sum(energies_counts_product, axis=1) / np.sum(counts, axis=1)


# Takes the total energy - energies multiplied by counts and summed.
def total_energy(counts, energies):
    """
    :param counts: counts corresponding to energies
    :param energies: energy values
    :return: numpy array of average of energies chosen

    >>> counts = np.array([[1, 4, 1, 5, 2], [5, 4, 1, 2, 1]])
    >>> energies = np.array([1, 2, 3, 4, 5])
    >>> total_energy(counts, energies)
    array([42, 29])
    """

    # Multiply energies by counts to weight.
    energies_counts_product = np.multiply(energies, counts)

    # Take the average of these energies.
    return np.sum(energies_counts_product, axis=1)


# Applies a Gaussian blur to the 2D sequence, along the given dimensions only.
def gaussian_blur(sequence, sigma, dims=[0]):
    sigmas = np.zeros(len(sequence.shape))
    sigmas[np.asarray(dims)] = sigma
    return gaussian_filter(np.array(sequence).astype(float), sigma=sigmas)


# Applies a uniform filter to a sequence - elements are replaced by the average of their neighbours.
def uniform_blur(sequence, filter_size):
    return uniform_filter(np.array(sequence).astype(float), filter_size, mode='constant')


# Applies a median filter to the sequence.
def median_blur(sequence, filter_size):
    return median_filter(sequence, filter_size)


# Assuming the objective function is unimodal, we can run ternary search to find the minima.
def ternary_search(objective, low, high, eps=0.0001):
    if high < low:
        raise ValueError('Invalid parameters: high must be greater than low.')

    if objective(low) == np.inf:
        return ValueError('Objective function takes value infinity.')

    def restrict_domain(objective, low, high):
        if objective(high) == np.inf:
            for jump in 2 ** np.linspace(0, 10, 11) / 1000:
                if objective(low + jump) < np.inf:
                    high = low + jump
                    break

        return low, high

    # Ensure that we do not have any infinity values within this domain by restricting it!
    low, high = restrict_domain(objective, low, high)

    while high - low > eps:
        m1 = low + (high - low)/3
        m2 = low + 2*(high - low)/3

        if objective(m1) < objective(m2):
            high = m2
        else:
            low = m1

    return low


# Runs one run of simulated annealing to minimize the objective function.
def simulated_annealing(objective, init_state):
    class NoiseThresholdOptimization(Annealer):

        # Move state randomly, ensuring always positive.
        def move(self):
            self.state += np.random.randn() * self.state
            self.state = max(self.state, 0)

        # Energy of a state is just the objective function evaluated in that state.
        def energy(self):
            return objective(self.state)

    best_state, best_ratio = NoiseThresholdOptimization(init_state).anneal()
    return best_state


# Reconstructs original data from PCA.
def reconstruct_from_PCA(data, mean, pca_obj):
    reconstructed_data = np.dot(data, pca_obj.components_)
    reconstructed_data += mean
    return reconstructed_data


# Gets the first index from the start such that the prefix sum is greater than or equal to the given fraction of the total.
# Similarly, gets the first index from the end. Both of these are returned as a 2-tuple.
def fraction_sum_indices(arr, fraction=0.5):
    """
    >>> fraction_sum_indices([1, 2, 3])
    (1, 2)
    >>> fraction_sum_indices([1, 8, 1])
    (1, 1)
    >>> fraction_sum_indices([0, 1, 0])
    (1, 1)
    >>> fraction_sum_indices([3, 2, 1])
    (0, 1)
    >>> fraction_sum_indices([3, 2, 1, 6, 8])
    (3, 3)
    >>> fraction_sum_indices([])
    (None, None)
    """

    # The indices we will return.
    left, right = None, None

    # Sum of the entire array.
    sum = np.sum(arr)

    # Prefixes from the left.
    currprefix = 0
    for index, val in enumerate(arr):
        currprefix += val
        if currprefix >= fraction * sum:
            left = index
            break

    # Suffixes from the right.
    currsuffix = 0
    for index, val in reversed(list(enumerate(arr))):
        currsuffix += val
        if currsuffix >= fraction * sum:
            right = index
            break

    return left, right


# KL-divergence between two multivariate normals.
def kl_divergence_normals(mean_1, covar_1, mean_2, covar_2):
    """
    >>> '%0.3f' % kl_divergence_normals([2], np.eye(1), [1], np.eye(1))
    '0.500'
    >>> '%0.3f' % kl_divergence_normals([1], np.eye(1), [1], 2 * np.eye(1))
    '0.097'
    >>> '%0.3f' % kl_divergence_normals(np.ones(3), np.eye(3), np.ones(3), np.eye(3))
    '0.000'
    """

    # Reshaping for numpy to multiply correctly.
    mean_1 = np.reshape(mean_1, (len(mean_1), 1))
    mean_2 = np.reshape(mean_2, (len(mean_2), 1))

    # Plug into the big formula.
    kl_divergence = 0.5 * (np.log(np.linalg.det(covar_2)/np.linalg.det(covar_1)) - len(mean_1) + np.trace(np.matmul(np.linalg.inv(covar_2), covar_1)) + np.squeeze(np.matmul((mean_2 - mean_1).T, np.matmul(np.linalg.inv(covar_2), (mean_2 - mean_1)))))

    # Due to numerical precision, we sometimes end up negative but very close to 0.
    if kl_divergence < 0:
        warnings.warn('Computed KL-divergence %0.3f less than 0.' % kl_divergence)
        kl_divergence = 0

    return kl_divergence


# Wrapper for vectorized functions to handle empty arrays.
def check_if_empty(func):
    def wrapped_func(array):
        if np.asarray(array).size == 0:
            return []
        else:
            return func(array)
    return wrapped_func


# Convert to floats (unit days). We vectorize this to work over arrays nicely.
@check_if_empty
@np.vectorize
def datestring_to_float(timestep, format_string='%d-%m-%Y/%H:%M:%S'):
    return mdates.date2num(datetime.strptime(timestep, format_string))


# Convert to strings. We vectorize this to work over arrays nicely.
@check_if_empty
@np.vectorize
def float_to_datestring(timestep, format_string='%d-%m-%Y/%H:%M:%S'):
    return datetime.strftime(mdates.num2date(timestep), format_string)


# Convert a string to a datetime object.
def convert_to_dt(timestep):
    return dateutil.parser.parse(timestep, dayfirst=True)


# Gets the day of the year from a string/datetime object.
def day_of_year(dt):
    try:
        return dt.timetuple().tm_yday
    except AttributeError:
        return day_of_year(convert_to_dt(dt))


# Gets the year from a string/datetime object.
def year(dt):
    try:
        return dt.year
    except AttributeError:
        return year(convert_to_dt(dt))


# Gets the hour from a string/datetime object.
def hour(dt):
    try:
        return dt.hour
    except AttributeError:
        return hour(convert_to_dt(dt))


# Returns the ELS .DAT file corresponding to a string/datetime object.
def get_ELS_file_name(dt, remove_extension=False):
    """
    >>> get_ELS_file_name('28-06-2004/22:00')
    'ELS_200418018_V01.DAT'
    >>> get_ELS_file_name('28-06-2004/09:00')
    'ELS_200418006_V01.DAT'
    >>> get_ELS_file_name('29-06-2004/09:00')
    'ELS_200418106_V01.DAT'
    >>> get_ELS_file_name('29-06-2005/09:00')
    'ELS_200518006_V01.DAT'
    >>> get_ELS_file_name('30-06-2005/09:00')
    'ELS_200518106_V01.DAT'
    >>> get_ELS_file_name(datetime(year=2004, month=1, day=1))
    'ELS_200400100_V01.DAT'
    >>> get_ELS_file_name(datetime(year=2004, month=1, day=2))
    'ELS_200400200_V01.DAT'
    """

    try:
        dt = convert_to_dt(dt)
    except TypeError:
        pass

    def doy_map(doy):
        return '0' * (3 - len(str(doy))) + str(doy)

    def hour_map(hour):
        def expand(num):
            if num == 0:
                return '00'
            elif num == 1:
                return '06'
            elif num == 2:
                return '12'
            elif num == 3:
                return '18'
        return expand(hour // 6)

    if remove_extension:
        basestring = 'ELS_%d%s%s_V01'
    else:
        basestring = 'ELS_%d%s%s_V01.DAT'

    return basestring % (year(dt), doy_map(day_of_year(dt)), hour_map(hour(dt)))


# Returns a nicer string representation of a dictionary.
def format_dict(d):
    return ',  '.join(['%s: %s' % (k, v) for k, v in d.iteritems()])
