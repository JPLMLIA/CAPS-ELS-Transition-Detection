#!/bin/bash

# Create the config file. Replace common_config.yaml by the config file name.
./create_config.py common_config.yaml

# Set this config file. This sets 2 environment variables: CONFIG_FILE and JOB_ARRAY_RANGE.
setconf common_config.yaml
# This is equivalent to:
# export CONFIG_FILE=common_config.yaml
# export JOB_ARRAY_RANGE="0-$(( $(yq .MINIMUM_JOB_ARRAY_SIZE $CONFIG_FILE) - 1))"

# Using job chaining.
id_1=$(sbatch --partition=shared --export=CONFIG_FILE --hold compute_pca_components.py)
id_1=${id_1##* }
id_2=$(sbatch --partition=shared --export=CONFIG_FILE --depend=afterok:$id_1 --array=$JOB_ARRAY_RANGE parameter_sweep.py)
id_2=${id_2##* }
id_3=$(sbatch --partition=shared --export=CONFIG_FILE --depend=afterok:$id_2 compute_threshold_range.py)
id_3=${id_3##* }
id_4=$(sbatch --partition=shared --export=CONFIG_FILE --depend=afterok:$id_3 --array=$JOB_ARRAY_RANGE collect_matrices_time_tolerance.py)
id_4=${id_4##* }
id_5=$(sbatch --partition=shared --export=CONFIG_FILE --depend=afterok:$id_4 combine_matrices_time_tolerance.py)
id_5=${id_5##* }
id_6=$(sbatch --partition=shared --export=CONFIG_FILE --depend=afterok:$id_5 compute_results.py)
id_6=${id_6##* }
scontrol release $id_1

# If you don't need to recompute scores (such as when recomputing confusion matrices with different labels):
id_3=$(sbatch --partition=shared --export=CONFIG_FILE --hold compute_threshold_range.py)
id_3=${id_3##* }
id_4=$(sbatch --partition=shared --export=CONFIG_FILE --depend=afterok:$id_3 --array=$JOB_ARRAY_RANGE collect_matrices_time_tolerance.py)
id_4=${id_4##* }
id_5=$(sbatch --partition=shared --export=CONFIG_FILE --depend=afterok:$id_4 combine_matrices_time_tolerance.py)
id_5=${id_5##* }
id_6=$(sbatch --partition=shared --export=CONFIG_FILE --depend=afterok:$id_5 compute_results.py)
id_6=${id_6##* }
scontrol release $id_3
