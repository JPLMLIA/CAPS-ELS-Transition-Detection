"""
Contains code to transform and filter ELS counts.

Author: Ameya Daigavane.
"""

# External dependencies.
from __future__ import division
import numpy as np
from scipy.ndimage.filters import minimum_filter, median_filter, maximum_filter

class Transformation(object):
    """
    Applies a fixed transformation to ELS counts.
    Transformations are applied point-wise to every single count entry,
    independent of the other counts.
    """

    def __init__(self, transform):
        if transform not in [
                'anscombe_transform',
                'log_transform',
                'no_transform',
            ]:
            raise ValueError('Invalid transform specified.')
        self.transformation_func = getattr(Transformation, transform)

    def transform(self, counts):
        """
        Applies the selected transformation to the passed counts.
        """
        return self.transformation_func(counts)

    @staticmethod
    def anscombe_transform(counts):
        """
        Applies the Anscombe transform to each dimension independently.
        """
        return np.sqrt(counts + 3/8)

    @staticmethod
    def log_transform(counts):
        """
        Applies the log transform to each dimension independently.
        """
        return np.log(counts + 1e-9)

    @staticmethod
    def no_transform(counts):
        """
        Does nothing, just returns the counts passed.
        """
        return counts


class Filter(object):
    """
    Applies a fixed filter to ELS counts.
    Filters modify individual count entries, possibly dependending
    on the entries in its neighbourhood.
    """

    def __init__(self, filter, filter_size):
        if filter not in [
                'min_filter',
                'median_filter',
                'max_filter',
                'no_filter'
            ]:
            raise ValueError('Invalid filter specified.')
        self.filteration_func = getattr(Filter, filter)
        self.filter_size = filter_size

    def filter(self, counts):
        """
        Applies the selected filter to the passed counts.
        """
        return self.filteration_func(counts, self.filter_size)

    @staticmethod
    def min_filter(counts, filter_size):
        """
        A minimum filter centered around the current timestep.
        """
        return minimum_filter(counts, size=(filter_size, 1),
                              mode='reflect', origin=(filter_size - 1)//2)

    @staticmethod
    def median_filter(counts, filter_size):
        """
        A median filter centered around the current timestep.
        """
        return median_filter(counts, size=(filter_size, 1),
                             mode='reflect', origin=((filter_size - 1)//2, 0))

    @staticmethod
    def max_filter(counts, filter_size):
        """
        A maximum filter centered around the current timestep.
        """
        return maximum_filter(counts, size=(filter_size, 1),
                              mode='reflect', origin=(filter_size - 1)//2)

    @staticmethod
    def no_filter(counts, filter_size):
        """
        Does nothing, just returns the counts passed.
        """
        return counts
