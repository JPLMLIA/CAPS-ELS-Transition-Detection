#!/usr/bin/env python

from __future__ import division
from data_utils import get_label_file_names
from datetime import datetime
from dateutil.parser import parse
import os
import argparse
import yaml
import pandas as pd


# Save labels to a file.
def save_labels(labels_file, anomalies, rewrite=False, mode='prompt'):

    # Check if the labels file exists previously.
    print 'Opening file %s.' % labels_file
    file_data = {}
    if os.path.exists(labels_file):
        with open(labels_file, 'r') as labels_file_obj:
            file_data = yaml.safe_load(labels_file_obj)

    previous_change_points = file_data.get('change_points', [])
    previous_bimodality = file_data.get('bimodality', [])
    previous_negative_ions = file_data.get('negative_ions', [])

    print 'Labels previously present in labels file:'
    print '\t Change-points: %s.' % previous_change_points
    print '\t Bimodality intervals: %s.' % previous_bimodality
    print '\t Negative ions intervals: %s. \n' % previous_negative_ions

    # Unpack dictionary, and parse given strings.
    change_points = parse_into_strings(anomalies.get('change_points', []))
    bimodality = list_of_tuples(parse_into_strings(anomalies.get('bimodality', [])))
    negative_ions = list_of_tuples(parse_into_strings(anomalies.get('negative_ions', [])))

    print 'Labels to be added into labels file:'
    print '\t Change-points: %s.' % change_points
    print '\t Bimodality intervals: %s.' % bimodality
    print '\t Negative ions intervals: %s. \n' % negative_ions

    # If we're not rewriting, add the previous entries.
    if not rewrite:
        print 'Adding labels in append mode...'
        change_points.extend(previous_change_points)
        bimodality.extend(previous_bimodality)
        negative_ions.extend(previous_negative_ions)
    else:
        print 'Overwriting previous labels...'

    # The names (keys) are important!
    arrays_with_names = {
        'change_points': change_points,
        'bimodality': bimodality,
        'negative_ions': negative_ions,
    }

    print 'After this operation, the labels file will contain:'
    print yaml.safe_dump(arrays_with_names)

    # Prompt just before we do any file operations.
    if mode == 'prompt':
        to_save = user_prompt()
    elif mode == 'no_prompt':
        to_save = True
    else:
        raise ValueError('Invalid mode - choose from \'prompt\', \'no_prompt\'.')

    # Save to the labels file.
    if to_save:
        with open(labels_file, 'w') as labels_file_obj:
            yaml.safe_dump(arrays_with_names, labels_file_obj)
        print 'Saved to file %s. \n' % labels_file
    else:
        print 'Labels file has not been modified. Quitting...'

    return to_save


# Prompt for user input.
def user_prompt():
    yes_options = {'y', 'ye', 'yes'}
    no_options = {'n', 'no'}

    while True:
        choice = raw_input('Please enter \'y\' to confirm, \'n\' to quit: ').lower()
        if choice in yes_options:
            return True
        elif choice in no_options:
            return False


# Parse into a common format.
def parse_into_strings(flat_list):
    return [datetime.strftime(parse(timestring, dayfirst=True), '%d-%m-%Y/%H:%M:%S') for timestring in flat_list]


# Create a list of tuples from a flat list.
def list_of_tuples(flat_list):

    # Combine every 2 timestrings into a tuple.
    it = iter(flat_list)
    return zip(it, it)


# Parses year, date and time together. Required to store in the CSV file.
def parse_after_combine(year, date, time):
    return datetime.strftime(parse(' '.join([time, date, year])), '%d-%m-%Y/%H:%M:%S')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument('--labels_file',
                        help='File to save labels to (in YAML (.yaml) format).')
    parser.add_argument('-c', '--change_points', nargs='+', default=[],
                        help='Change point times in a valid datetime format, eg. 19:08, 28th Jun 2004.')
    parser.add_argument('-b', '--bimodality', nargs='+', default=[],
                        help='Tuples of (start time, end time) for bimodality regions in a valid datetime format.')
    parser.add_argument('-n', '--negative_ions', nargs='+', default=[],
                        help='Tuples of (start time, end time) for negative ion regions in a valid datetime format.')
    parser.add_argument('-rw', '--rewrite', default=False, action='store_true',
                        help='Overwrite labels file, if it already exists.')
    parser.add_argument('-ip', '--input', default=None,
                        help='Use this CSV file (in the right format) with labels as an input. \
                              Note that the other options are ignored if this option is selected. \
                              The appropriate labels file are created according to the CAPS ELS format based on the times.')
    parser.add_argument('-iptype', '--input_type', default='change_points', choices=['change_points', 'bimodality', 'negative_ions'],
                        help='When used with the \'-ip\' option, select the type of anomalies to be saved as.')
    parser.add_argument('-od', '--output_dir', default='temp/labels/',
                        help='Output directory for all the generated labels.')

    args = parser.parse_args()

    # If we have a labels file given to us, use that.
    if args.input is None:
        anomalies = {
            'change_points': args.change_points,
            'bimodality': args.bimodality,
            'negative_ions': args.negative_ions,
        }

        save_labels(args.labels_file, anomalies, args.rewrite)

    # Otherwise, read in the CSV file.
    else:

        # Construct a dataframe from the input CSV file.
        df = pd.read_csv(args.input, sep=',', parse_dates={'datetime': ['Year', 'Date', 'Time']}, date_parser=parse_after_combine, names=['Year', 'Date', 'Time', 'in/out'], skiprows=1)

        # We'll fill this in later.
        anomalies = {
            args.input_type: []
        }

        # Get the label files to save individual anomalies to.
        labels_files = get_label_file_names(df['datetime'])
        anomaly_times = df['datetime']

        # Iterate, and keep checking if we ever quit.
        for anomaly_time, labels_file in zip(anomaly_times, labels_files):
            anomalies[args.input_type] = [anomaly_time]

            keep_saving = save_labels(args.output_dir + labels_file, anomalies, mode='no_prompt')
            if not keep_saving:
                break
