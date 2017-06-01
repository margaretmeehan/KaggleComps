import os

### PROBLEM INFO ###
NUMBER_OF_CLUSTERS = 10
NUMBER_OF_SEEDS = 3


REDUCTION_PCA_DIMENSIONS = 90
REDUCTION_SPECTRAL_DIMENSIONS = 103
REDUCTION_CCA_DIMENSIONS = 6


### PATHS ###
# All paths are based off of scripts being run from the /.../comp1/ directory
# All path constants should end in a "/
PATH_DATA = os.getcwd() + "/data/"


### SUBMISSION FORMAT ###
KAGGLE_SUBMISSION_HEADER = ["Id", "Category"]
LABEL_COMPARISON_HEADER = ["Id", "Feature_Label", "Adj_Label"]
