### Instructions for Evaluating on HPC

This is a brief set of steps to evaluate on our PBS Pro cluster. For users outside NASA-JPL, you will need to modify paths in **create_config.py** and at
the top of most scripts for the Python interpreter to import correctly.

Type **hpchelp** in the shell (after updating your **.bashrc** as indicated below) to see the list of commands to run.

Some of these scripts spawn job arrays, so please use **setconf** to set the config file.
The config file is common to all scripts and supplies crucial information about the current evaluation.
Otherwise, you will have to be careful to set the size of the job array (#PBS -J option at the top of the script.)

This is a description of each of the scripts:
1. Compute PCA components of counts with **compute_pca_components.py**.
2. Sweep parameters with **parameter_sweep.py**, creating result files in the HDF5 format, by calling **../find_scores.py**.
3. Compute common threshold range with **compute_thresholds.py**.
4. Collect confusion matrices for each result file with **collect_matrices_time_tolerance.py** (which calls **../evaluate_methods_time_tolerance.py**).
5. Combine confusion matrices for each algorithm with **combine_matrices_time_tolerance.py**.
6. Compute and save precision-recall curves for each algorithm (and for each year) with **compute_results.py**.

There are some other scripts (**analyze_errors.py**, **plot_errors.py**) that are not essential for the HPC evaluation pipeline,
but can still be used to analyze CAPS ELS data.

### General Help
Use Bash aliases and functions to make life easier!
Here are some relevant parts of my **.bashrc**:
```
export HOME="/scratch_lg/image-content/ameyasd/"
export SCRIPTS="$HOME/europa-onboard-science/src/caps_els/"

alias hpchelp="cat $SCRIPTS/secondary_scripts/hpc_evaluate_helper.sh"
myjobs() { qstat "$1" | grep ameyasd ; }
dirs() { find . -maxdepth 1 -type d  | awk '{print "echo -n \""$0"  \";ls -l "$0" | grep -v total | wc -l" }' | sh ; }
setconf() { export CONFIG_FILE="$1" && export JOB_ARRAY_RANGE="0-$(( $(yq .MINIMUM_JOB_ARRAY_SIZE $CONFIG_FILE) - 1))";}
hpctest() { $SCRIPTS/secondary_scripts/hpc_evaluate_test.sh "$@"; }
```

Now, in the bash shell:
* **myjobs** to see the status of your jobs. You can pass it the same options as you would to **qstat**.
* **hpchelp** to show the contents of **hpc_evaluate_helper.sh**.
* **dirs** to see how many files are present in subdirectories in the current directory.
* **setconf config-file** to set the config file.
* **hpctest config-file script** to test a script, with $PBS_ARRAY_INDEX as 0.

Useful **qstat** options:
* **-x** to see all jobs (upto some history), even ones that have finished.
* **-W depend=afterok:$jobid** will hold the current scheduled job (make it wait) until job $jobid has terminated successfully.
* **-h** to put a hold on a job when scheduling it. This is very useful for jobs which are dependencies of other jobs.
* **-t** to expand job arrays.
