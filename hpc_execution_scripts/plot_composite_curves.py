#!/usr/bin/env python

import pandas
import matplotlib.pyplot as plt
import argparse
from sklearn.metrics import auc


def create_composite_plot(csv_files, names, title, output_file, plot_type):

    type_to_columns_map = {
        'roc': {
            'x': 'fpr',
            'y': 'tpr',
        },
        'pr': {
            'x': 'tpr',
            'y': 'precision',
        },
    }

    columns_to_name_map = {
        'fpr': 'False Positive Rate (FPR)',
        'tpr': 'True Positive Rate (TPR)',
        'precision': 'Precision',
        'recall': 'Recall',
        'f1': 'F1-Score',
        'thresholds': 'Thresholds',
    }

    # Get the right columns for the plot!
    x_col = type_to_columns_map[plot_type]['x']
    y_col = type_to_columns_map[plot_type]['y']

    # Get the right labels for these columns.
    x_label = columns_to_name_map[x_col]
    y_label = columns_to_name_map[y_col]

    # Read CSV files into dataframes.
    dfs = [pandas.read_csv(filename) for filename in csv_files]

    # Extract TPR and FPR lists.
    x_lists = [df[x_col].to_numpy() for df in dfs]
    y_lists = [df[y_col].to_numpy() for df in dfs]

    # Compute area under curve.
    auc_list = [auc(x_list, y_list) for x_list, y_list in zip(x_lists, y_lists)]

    # Default titles.
    if names is None:
        names = map(str, range(len(dfs)))

    # Plot everything.
    labels = [name + ' (AUC: %0.3f)' % auroc for name, auroc in zip(names, auc_list)]
    fig, ax = plt.subplots(1, 1)
    for x_list, y_list, label in zip(x_lists, y_lists, labels):
        ax.plot(x_list, y_list, '-o', label=label)
    ax.legend()
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_xlim(0, 1.005)
    ax.set_ylabel(y_label)
    ax.set_ylim(0, 1.005)
    ax.set_aspect('equal', adjustable='box')

    # Baseline for ROC.
    if plot_type == 'roc':
        ax.plot([0, 1], [0, 1], 'gray', dashes=[6, 4])

    # Save to file, if we have to.
    if output_file is None:
        plt.show()
    else:
        plt.savefig(output_file)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)

    parser.add_argument('csv_files', nargs='+',
                        help='CSV files containing individual algorithm values.')
    parser.add_argument('-pt', '--plot_type', default='roc', choices=['roc', 'pr'],
                        help='The type of composite curve to plot. Only the Receiver Operating Characteristic and the Precision Recall curves are currently supported.')
    parser.add_argument('-n', '--names', nargs='+', default=None,
                        help='Names (for the plot) for each of the algorithms.')
    parser.add_argument('-t', '--title', default='Composite Curve',
                        help='Titles for the composite plot.')
    parser.add_argument('-o', '--output_file', default=None,
                        help='Save the composite plot to this file.')

    args = parser.parse_args()
    create_composite_plot(**vars(args))

