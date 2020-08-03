#!/usr/bin/env python

# This script plots PR curves for each algorithm's combined confusion matrix.
# Author: Ameya Daigavane

# External dependencies.
import numpy as np
import os
import sys
import re
import os
import matplotlib
import matplotlib.pyplot as plt
# plt.switch_backend('agg')
from collections import defaultdict
import yaml

# Internal dependencies.
sys.path.append('../')
from evaluate_methods_time_tolerance import Metrics
from stats_plotters import StatsPlotter

# Helper functions specific to the way our files are named.
def get_algorithm_class(algorithm_name):
    return algorithm_name[:3]

def get_parameters(algorithm_name):
    try:
        for prefix in algorithm_file_prefixes:
            if prefix in algorithm_name:
                return parse(algorithm_name[len(prefix):])
    except ValueError:
        return ''

def parse_num(num_str):
    if num_str[0] == '0':
        return num_str[0] + '.' + num_str[1:]
    else:
        return num_str

def parse(substr):
    split = list(re.split(r'(\d+)', substr))
    res = ''
    for param, val in zip(split[0::2], split[1::2]):
        res += param + ' = ' + parse_num(val) + ', '
    return res[:-2]

def get_key(algorithm):
    try:
        params = get_parameters(algorithm)
        start_index = 4
        end_index = params.find(',')
        if end_index == -1:
            end_index = len(params)
        return int(params[start_index: end_index])
    except ValueError:
        return ''

def parse_year(filename):
    basename = os.path.basename(filename)
    start_index = basename.find('20')
    return basename[start_index: start_index + 4]

def parse_type(filename):
    if 'mp/in/' in filename:
        return 'MP - IN'
    if 'mp/out/' in filename:
        return 'MP - OUT'
    if 'bs/in/' in filename:
        return 'BS - IN'
    if 'bs/out/' in filename:
        return 'BS - OUT'
    if 'valid/' in filename:
        return 'VALID'

# Load paths from config.
# CONFIG_FILE = os.environ['CONFIG_FILE']
# with open('/scratch_lg/image-content/ameyasd/europa-onboard-science/src/caps_els/secondary_scripts/' + CONFIG_FILE, 'r') as config_file_object:
#     config = yaml.safe_load(config_file_object) 
#     TIME_TOLERANCE = config['TIME_TOLERANCE']
#     COMBINED_MATRICES_FILE = config['COMBINED_MATRICES_FILE']
#     TRANSFORM = config['TRANSFORM']
#     MODE = config['MODE']

# transform = TRANSFORM.split('_')[0].lower()
TIME_TOLERANCE = 20
transform = 'anscombe'
MODE = 'train'
# results = np.load(COMBINED_MATRICES_FILE)

algorithm_file_prefixes = ['rulsif', 'stickyhmm', 'vhmm', 'hotsax', 'mp', 'baseline']
algorithm_classes = ['rul', 'sti', 'vhm', 'hot', 'mpw', 'bas']
algorithm_classes_expanded = {
    'rul': 'RuLSIF',
    'sti': 'Sticky HDP-HMM',
    'vhm': 'Vanilla HMM',
    'hot': 'HOT SAX',
    'mpw': 'Matrix Profile',
    'bas': 'L2-Diff',  
}

if MODE == 'train':
    algorithm_classes_subplots = {
        'rul': 4,
        'sti': 3,
        'vhm': 3,
        'hot': 4,
        'mpw': 3,
        'bas': 1,
    }
elif MODE == 'test':
    algorithm_classes_subplots = {
        'rul': 1,
        'sti': 1,
        'vhm': 1, 
        'hot': 1,
        'mpw': 1,
        'bas': 1,
    }

# blurs = [2, 5]
blur = 5
bin_selection = 'ignore_unpaired'
filters = ['no_filter_120', 'min_filter_120', 'median_filter_120', 'max_filter_120']
years = ['all']
# years = ['all'] + range(2005, 2013)
# types = ['bs/in/', 'bs/out/', 'mp/in/', 'mp/out/', 'valid/']
types = ['bs/all/', 'mp/all/']
type = 'mp/all/'

# Create subplots for algorithm classes.
# subplots = {}
# for filter in filters:
#     subplots[filter] = {algorithm_class: plt.subplots(nrows=len(types), ncols=algorithm_classes_subplots[algorithm_class], figsize=(20, 15)) for algorithm_class in algorithm_classes}
subplots = {algorithm_class: plt.subplots(nrows=len(filters), ncols=algorithm_classes_subplots[algorithm_class], figsize=(20, 20)) for algorithm_class in algorithm_classes}
printed = defaultdict(lambda: defaultdict(lambda: False))

# Now compute summaries.
# files = ['../temp/crossings_test_results/20min/labels/' + f for f in os.listdir('../temp/crossings_test_results/20min/labels/') if '20' in f and '.npz' in f]
# files = ['../temp/crossings_test_results/20min/new_labels/' + dir + 'combined_results_.npz' for dir in ['mp/in/', 'mp/out/', 'bs/in/', 'bs/out/', 'valid/']]
# files = ['../temp/blur_sigma_experiments/blur_%d/' % dir + 'combined_results_.npz' for dir in [0, 1, 5, 10]]
# files = ['../temp/crossings_test_results/blur_%d/' % dir + 'combined_results_.npz' for dir in [0, 1, 5, 10]]
# for index, f in enumerate(sorted(files)):
# for blur in [0, 1, 5, 10, 20, 50, 100, 200]:
for subplot_row, filter in enumerate(filters):
    # for filter in filters:
    for index, year in enumerate(years):
        # f = ('../temp/crossings_test_results/blur_%d/new_labels/%s' % (blur, type)) + ('combined_matrices_%s.npz' % str(year))
        f = ('../temp/filter_experiments/%s/%s/%s' % (bin_selection, filter, type)) + ('combined_matrices_%s.npz' % str(year))
        # f = ('../temp/blur_sigma_experiments/anscombe_transform/blur_%d/%s' % (blur, type)) + ('combined_matrices_%s.npz' % str(year))
        results = np.load(f)
        
        # Identification of subplots.
        key_params = defaultdict(set)
        for algorithm in sorted(results):
            key_params[get_algorithm_class(algorithm)].add(get_key(algorithm))

        for algorithm_class in key_params:
            key_params[algorithm_class] = sorted(list(key_params[algorithm_class]))

        subplot_key = dict()
        for algorithm in sorted(results):
            subplot_key[algorithm] = key_params[get_algorithm_class(algorithm)].index(get_key(algorithm))

        for algorithm in sorted(results):
            print('Blur %d: %s' % (blur, algorithm))
            stats = Metrics()

            algorithm_max_recall = -1
            for confusion_matrix in results[algorithm]:
                (true_positives, true_negatives, false_positives, false_negatives) = confusion_matrix
                if true_positives + false_positives > 0:
                    stats.compute_precision(confusion_matrix)
                    stats.compute_recall(confusion_matrix)
                    stats.compute_f1(confusion_matrix)
            
            stats.compute_summaries()

            # Plot.
            algorithm_class = get_algorithm_class(algorithm)
            subplot_col = subplot_key[algorithm]
            # subplot_col = index
            # subplot_row = 0
            plotter = StatsPlotter(stats)
            title_params = {
                'algorithm_name': algorithm_classes_expanded[algorithm_class],
                'time_tolerance': TIME_TOLERANCE,
            }
            # fig, axs = subplots[filter][algorithm_class]
            fig, axs = subplots[algorithm_class]

            # Handle case when only one subplot present.
            try:
                ax = axs[subplot_row][subplot_col]
            except TypeError:
                try:
                    ax = axs[subplot_row]
                except TypeError:
                    try:
                        ax = axs[subplot_col]
                    except TypeError:
                        ax = axs

            legend_title = 'Parameters'
            cmap = matplotlib.cm.get_cmap('viridis')
            norm = matplotlib.colors.LogNorm(vmin=1e-1, vmax=200)
            plotter.plot_pr(fig, ax, title_params={}, legend=get_parameters(algorithm))
            
            if subplot_col == 0 and not printed[algorithm_class][filter]:
                ax.text(-0.8, 0.5, '%s' % filter, fontsize=12)
                printed[algorithm_class][filter] = True
            
            # if subplot_col == len(years)//2:
            if subplot_row == len(filters) - 1:
                ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), fancybox=True, shadow=True, title=legend_title, fontsize=9)
                # ax.legend(loc='best', fancybox=True, shadow=True, title=legend_title, fontsize=9)

            # if subplot_row == 0:
            #     ax.set_title(str(year))
            # ax.set_title(parse_type(f))
            # ax.set_title('Performance of L2-Diff with Varying Amounts of Blur\n Anscombe Transform')

for algorithm_class, (fig, axs) in subplots.items():
# for algorithm_class, (fig, axs) in subplots.items():
    fig.suptitle('%s\n%s' % (type, algorithm_classes_expanded[algorithm_class]), fontsize=16, y=1)
    fig.subplots_adjust(left=0.12, right=0.95, bottom=0.27, wspace=0.45, hspace=0.2, top=0.95)
    fig.savefig('../temp/filter_experiments/%s/combined/%s/%s.png' % (bin_selection, type, algorithm_class), dpi=fig.dpi, bbox_inches='tight')
    
    # fig.savefig('../temp/blur_sigma_experiments/anscombe_transform/blur_' + str(blur) + '/' + algorithm_class + '_typewise_all.png', dpi=fig.dpi, bbox_inches='tight')
    # fig.savefig('typewise_' + algorithm_class + '_' + transform + '_transform.png')
    # fig.savefig(COMBINED_MATRICES_FILE[:-4] + '_' + algorithm_class + '_' + transform + '_transform.png')
    # fig.savefig('../figs/' + algorithm_class + '_' + transform + '_transform.png')
    # fig.show()
# plt.show()