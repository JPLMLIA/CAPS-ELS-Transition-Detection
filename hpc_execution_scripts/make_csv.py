#!/usr/bin/env python

# External dependencies.
import argparse
from datetime import datetime
import pandas as pd

# Internal dependencies.
from data_utils import get_ELS_data


def create_csv(els_data_file, output_csv, quantity, start_time, end_time, no_index, no_headers):

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

    # Get the data!
    data = get_ELS_data(els_data_file, quantity, start_time, end_time)

    # Convert to a DataFrame, with the right fields.
    df = pd.DataFrame(data=data[0], columns=data[1], index=data[2])

    # Save as a CSV, with the right options.
    df.to_csv(output_csv, index=not no_index, header=not no_headers)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument('els_data_file', help='ELS DAT file.')
    parser.add_argument('output_csv', help='Name of the .csv file to save as.')
    parser.add_argument('-q', '--quantity', default='anode5', choices=(
            ['iso'] +
            [('anode%d' % a) for a in range(1, 9)] +
            [('def%d' % a) for a in range(1, 9)] +
            [('dnf%d' % a) for a in range(1, 9)] +
            [('psd%d' % a) for a in range(1, 9)]
    ))
    parser.add_argument('-st', '--start_time', default=None,
                        help='Start time in dd-mm-yyyy/HH:MM. Restricts data to those recorded on or after this time.')
    parser.add_argument('-et', '--end_time', default=None,
                        help='End time in dd-mm-yyyy/HH:MM. Restricts data to those recorded upto and including this time.')
    parser.add_argument('--no_index', default=False, action='store_true',
                        help='Do not store index (of times) as the first column in the .csv file.')
    parser.add_argument('--no_headers', default=False, action='store_true',
                        help='Do not store headers (of times) as the first row in the .csv file.')
    args = parser.parse_args()
    create_csv(**vars(args))