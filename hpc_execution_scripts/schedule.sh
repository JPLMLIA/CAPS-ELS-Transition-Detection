#!/bin/bash

for file in *v2*.yaml; do
    echo $file
    setconf $file
    id_1=$(qsub -v CONFIG_FILE=$CONFIG_FILE -h compute_pca_components.py)
    id_2=$(qsub -v CONFIG_FILE=$CONFIG_FILE -W depend=afterok:$id_1 -J $JOB_ARRAY_RANGE parameter_sweep.py)
    id_3=$(qsub -v CONFIG_FILE=$CONFIG_FILE -W depend=afterok:$id_2 compute_threshold_range.py)
    id_4=$(qsub -v CONFIG_FILE=$CONFIG_FILE -W depend=afterok:$id_3 -J $JOB_ARRAY_RANGE collect_matrices_time_tolerance.py)
    id_5=$(qsub -v CONFIG_FILE=$CONFIG_FILE -W depend=afterok:$id_4 combine_matrices_time_tolerance.py)
    id_6=$(qsub -v CONFIG_FILE=$CONFIG_FILE -W depend=afterok:$id_5 compute_results.py)
    qrls $id_1
done
