import csv
import constants
import datetime

import sklearn
from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize
from sklearn.manifold import SpectralEmbedding
from sklearn.cross_decomposition import CCA
from sklearn import mixture

import numpy as np
from numpy import loadtxt
from numpy import genfromtxt

import matplotlib.pyplot as plt
import pandas as pd
# import pyamg

import general_functions as gf

reload(gf)
reload(constants)

print constants.REDUCTION_PCA_DIMENSIONS
print constants.REDUCTION_SPECTRAL_DIMENSIONS
print constants.REDUCTION_CCA_DIMENSIONS


# ----------------------------------------------------------------------------------------------------------------


def main():
    X, adjacency_np, raw_seeds = gf.import_data()

    spectral_embedded_matrix = gf.do_spectral_embedding(adjacency_np, constants.REDUCTION_SPECTRAL_DIMENSIONS)

    cca_obj = CCA(n_components=constants.REDUCTION_CCA_DIMENSIONS)
    X_c, Y_c = cca_obj.fit_transform(X, spectral_embedded_matrix)

    # km = gf.get_kmeans_object(X_c, raw_seeds, num_clusters=30)
    # model_labels = km.labels_
    model_labels = gf.get_gmm_labels(X_c, raw_seeds)

    gf.plot_labels(X_c[:, 0], X_c[:, 1], list(model_labels))

    labels = map(lambda x: x / 3, model_labels)
    print list(model_labels)[:30]
    print labels[:30]

    gf.get_seed_labels(labels, raw_seeds)

    gf.make_submission(labels[:12000],
                       "../submissions/submission_%s" % datetime.datetime.now().strftime("%m-%d %H:%M:%S"))


if __name__ == '__main__':
    main()