# Detecting Magnetic Field Transitions in CAPS ELS Data

This directory contains Python scripts that operate on time-series data from CAPS ELS.

#### Algorithms
* **baseline_analysis** - Applies the Baseline (L2-Diff) algorithm on ELS data.
* **hmm_analysis** - Applies the Hidden Markov Model segmentation on ELS data, using models defined in **hmm_models**.
* **rulsif_analysis** - Applies the RuLSIF change-point detection algorithm on ELS data.
* **sax_analysis** - Applies the HOT SAX time-series discord detection algorithm on ELS data.
* **matrix_profile_analysis** - Applies the Matrix Profile algorithm on ELS data.

#### Evaluation
* **find_scores** - Executes an anomaly detection method with given parameters over a given data file, and stores returned scores, times and other metadata in a HDF5 file.
* **evaluate_methods_time_tolerance** - Evaluates prediction performance of anomaly detection methods from the generated HDF5 file, and plots performance curves.
* **stats_plotters** - Contains useful plotting functions for intervals, ROC, PR and Time-Differences curves, for the evaluation framework.

#### Utilities
* **plot_els** - Plots ELS data and quantities, and provides a plotting interface to be called from other scripts.
* **els_data** - Defines the ELS class as an interface for scripts to access ELS data and relevant computed quantities (Phase Space Density,  Differential Number Flux, Differential Energy Flux).
* **data_utils** - Helper functions for algorithms to perform common useful operations, eg. load ELS data.
* **transform** - Helper classes for transforms (Anscombe and log) and filters (min, median and max) for ELS data.

Most scripts will indicate the supported options by calling them with the *-h* flag.

## Dependencies
To run the scripts here, first install dependencies with:
```
pip install -r requirements.txt
``` 

## Algorithms

#### Baseline L2-Diff (baseline_analysis)
This implements the Baseline L2-Diff pipeline, for CAPS ELS data. This algorithm computes the L2-distance between the vectors of counts between adjacent timesteps.

#### Hidden Markov Models (hmm_analysis and hmm_models)
This implements the HMM pipelines from [hmmlearn](https://hmmlearn.readthedocs.io/en/latest/) (for non-Bayesian HMMs with Gaussian emissions) and [pyhsmm](https://github.com/mattjj/pyhsmm) (for ordinary [[1]](https://www.stat.berkeley.edu/~aldous/206-Exch/Papers/hierarchical_dirichlet.pdf) and sticky HDP-HMMs [[2]](https://icml.cc/Conferences/2008/papers/305.pdf) with Gaussian emissions) for CAPS ELS data.

#### RuLSIF (rulsif_analysis)
This implements the RuLSIF pipeline, from [[3]](https://arxiv.org/pdf/1203.0453.pdf), using [densratio](https://github.com/hoxo-m/densratio_py), for CAPS ELS data.

#### HOT SAX (sax_analysis)
This implements the HOT SAX pipeline, from [[4]](https://www.cs.ucr.edu/~eamonn/HOT%20SAX%20%20long-ver.pdf) and [[5]](https://ieeexplore.ieee.org/document/6987960),  using [saxpy](https://github.com/ameya98/saxpy), for CAPS ELS data.

#### Matrix Profile (matrix_profile_analysis)
This implements the Multidimensional Matrix Profile pipeline, built on top of the original Matrix Profile pipeline from [[6]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7837992&isnumber=7837813), with [matrixprofile-ts](https://github.com/ameya98/matrixprofile-ts), for CAPS ELS data.

## References
[1] Teh, Y. W., Jordan, M. I., Beal, M. J., Blei, D. M.,  
**Hierarchical Dirichlet Processes**. Journal of the American Statistical Association, 2006.

[2] Fox, E. B., Sudderth, E. B., Jordan, M. I., Willsky, A. S.,  
**An HDP-HMM for Systems with State Persistence**. ICML, 2008.

[3] Liu, S., Yamada, M., Collier, N., Sugiyama, M.,   
**Change-Point Detection in Time-Series Data by Relative Density-Ratio Estimation**. Neural Networks, 2013.

[4] Keogh, E., Lin, J., Fu, A.,  
**HOT SAX: Efficiently finding the most unusual time series subsequence**. ICDM, 2005.

[5] Mohammad, Y., Nishida T.,  
**Robust learning from demonstrations using multidimensional SAX**. ICCAS, 2014.

[6] Yeh, C. M., Zhu, Y., Ulanova, L., Begum, N., Ding, Y., Dau, H. A., Silva, D. F., Mueen, A., Keogh, E.,  
**Matrix Profile I: All Pairs Similarity Joins for Time Series: A Unifying View That Includes Motifs, Discords and Shapelets**. ICDM, 2016.
