#!/bin/bash

# Create the config file. Replace common_config.yaml by the config file name.
./create_config.py common_config.yaml

# Set this config file. This sets 2 environment variables: CONFIG_FILE and JOB_ARRAY_RANGE.
setconf common_config.yaml
# This is equivalent to:
# export CONFIG_FILE=$(pwd)/common_config.yaml
# export JOB_ARRAY_RANGE="0-$(( $(yq .MINIMUM_JOB_ARRAY_SIZE $CONFIG_FILE) - 1))"

# Using job chaining.
id_1=$(qsub -v CONFIG_FILE=$CONFIG_FILE -h compute_pca_components.py)
id_2=$(qsub -v CONFIG_FILE=$CONFIG_FILE -W depend=afterok:$id_1 -J $JOB_ARRAY_RANGE parameter_sweep.py)
id_3=$(qsub -v CONFIG_FILE=$CONFIG_FILE -W depend=afterok:$id_2 compute_threshold_range.py)
id_4=$(qsub -v CONFIG_FILE=$CONFIG_FILE -W depend=afterok:$id_3 -J $JOB_ARRAY_RANGE collect_matrices_time_tolerance.py)
id_5=$(qsub -v CONFIG_FILE=$CONFIG_FILE -W depend=afterok:$id_4 combine_matrices_time_tolerance.py)
id_6=$(qsub -v CONFIG_FILE=$CONFIG_FILE -W depend=afterok:$id_5 compute_results.py)
qrls $id_1

# If you don't need to recompute scores (such as when recomputing confusion matrices with different labels):
id_3=$(qsub -v CONFIG_FILE=$CONFIG_FILE -h compute_threshold_range.py)
id_4=$(qsub -v CONFIG_FILE=$CONFIG_FILE -W depend=afterok:$id_3 -J $JOB_ARRAY_RANGE collect_matrices_time_tolerance.py)
id_5=$(qsub -v CONFIG_FILE=$CONFIG_FILE -W depend=afterok:$id_4 combine_matrices_time_tolerance.py)
id_6=$(qsub -v CONFIG_FILE=$CONFIG_FILE -W depend=afterok:$id_5 compute_results.py)
qrls $id_3

