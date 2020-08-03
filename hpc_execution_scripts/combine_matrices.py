# Combines all the computed confusion matrices for all algorithms.


# External dependencies.
import os
import numpy as np
from pandas import DataFrame
import sys
from pathlib2 import Path

# Internal dependencies.
sys.path.append('/halo_nobackup/image-content/ameyasd/europa-onboard-science/src/caps_els/')
from evaluate_methods import Metrics

RESULTS_DIR = '/halo_nobackup/image-content/ameyasd/crossings_updated_results/'

# Combine confusion matrices and time differences for each algorithm.
algorithm_confusion_matrices = {}
algorithm_labelled_time_differences = {}
algorithm_predicted_time_differences = {}

for folder in os.listdir(RESULTS_DIR):
    
    if folder in ['stderr', 'stdout']:
        continue
    
    print 'Folder: %s.' % folder

    if len(os.listdir(RESULTS_DIR + folder)) <= 2:
        print 'Folder: %s is empty!' % folder
        continue

    confusion_matrices_path = RESULTS_DIR + folder + '/confusion_matrices/'
    for matrix_file in os.listdir(confusion_matrices_path):
        algorithm_name = '_'.join(str.split(matrix_file, '_')[:-2])
        if algorithm_name not in algorithm_confusion_matrices:
            algorithm_confusion_matrices[algorithm_name] = np.load(confusion_matrices_path + matrix_file)
        else:
            algorithm_confusion_matrices[algorithm_name] += np.load(confusion_matrices_path + matrix_file)

    time_differences_path = RESULTS_DIR + folder + '/time_differences/'
    for matrix_file in os.listdir(time_differences_path):
        algorithm_name = '_'.join(str.split(matrix_file, '_')[:-3])
        if 'labelled' in matrix_file:
            if algorithm_name not in algorithm_labelled_time_differences:
                algorithm_labelled_time_differences[algorithm_name] = np.load(time_differences_path + matrix_file).flatten()
            else:
                algorithm_labelled_time_differences[algorithm_name] = np.concatenate((algorithm_labelled_time_differences[algorithm_name], np.load(time_differences_path + matrix_file).flatten()))
        else:
            if algorithm_name not in algorithm_predicted_time_differences:
                algorithm_predicted_time_differences[algorithm_name] = np.load(time_differences_path + matrix_file).flatten()
            else:
                algorithm_predicted_time_differences[algorithm_name] = np.concatenate((algorithm_predicted_time_differences[algorithm_name], np.load(time_differences_path + matrix_file).flatten()))

# Compute statistics for each algorithm.
for algorithm, confusion_matrices in algorithm_confusion_matrices.items():

    stats = Metrics()

    # Scan over all thresholds.
    for index, confusion_matrix in enumerate(confusion_matrices):

        # Compute all metrics.
        stats.compute_tpr(confusion_matrix)
        stats.compute_fpr(confusion_matrix)
        stats.compute_precision(confusion_matrix)
        stats.compute_recall(confusion_matrix)
        stats.compute_f1(confusion_matrix)
        stats.compute_accuracy(confusion_matrix)
        stats.compute_auprc_baseline(confusion_matrix)

    print 'Algorithm  \'%s\' completed.' % algorithm

    # Save to CSV file - only add a header if this file is empty.
    all_data = {'tpr': stats.tpr_list, 'fpr': stats.fpr_list, 'precision': stats.precision_list, 'recall': stats.recall_list, 'accuracy': stats.accuracy_list, 'f1': stats.f1_score_list}
    df = DataFrame(data=all_data)

    combined_results_dir = RESULTS_DIR + 'combined/overall'
    if not os.path.exists(combined_results_dir):
        Path(combined_results_dir).mkdir(parents=True, exist_ok=True)

    with open(combined_results_dir + algorithm + '.csv', 'a') as f:
        df.to_csv(f, header=(f.tell() == 0), index=False)
    
    combined_results_dir = RESULTS_DIR + 'combined/confusion_matrices/'
    if not os.path.exists(combined_results_dir):
        Path(combined_results_dir).mkdir(parents=True, exist_ok=True)
     
    np.save(combined_results_dir + algorithm + '_confusion_matrices.npy', confusion_matrices)

# Log time differences.
for algorithm, time_differences in algorithm_labelled_time_differences.items():
    
    combined_results_dir = RESULTS_DIR + 'combined/labelled_time_differences/'                                                                                                                                                
    if not os.path.exists(combined_results_dir):
       Path(combined_results_dir).mkdir(parents=True, exist_ok=True)

    np.save(combined_results_dir + algorithm + '_labelled_time_differences.npy', time_differences)

# Log time differences. 
for algorithm, time_differences in algorithm_predicted_time_differences.items(): 

    combined_results_dir = RESULTS_DIR + 'combined/predicted_time_differences/'
    if not os.path.exists(combined_results_dir):    
        Path(combined_results_dir).mkdir(parents=True, exist_ok=True)
                                                                                                                                                                                         
    np.save(combined_results_dir + algorithm + '_predicted_time_differences.npy', time_differences)
                                                                                  
