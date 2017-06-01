import csv
from collections import defaultdict, Counter
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.manifold import SpectralEmbedding
from sklearn import mixture
import scipy.sparse.linalg as la
from sklearn.cross_decomposition import CCA

from constants import *

########################################################################################################
### DATA GETTERS ###
########################################################################################################
def import_data():
    """ Imports data. Must be called from inside /.../src/.

    Returns:
        X (numpy.ndarray): the matrix of feature vectors from features.csv
        adjacency_np (numpy.ndarray): the adjacency matrix from Adjacency.csv
        seeds (numpy.ndarray): the list of seed indices from seed.csv
    """
    X = pd.read_csv("../data/features.csv", header=None)
    X = normalize(X, axis=0)
    adjacency = pd.read_csv("../data/Adjacency.csv", header=None)
    adjacency_np = np.asarray(adjacency)
    seeds = pd.read_csv("../data/seed.csv", header=None)

    print type(X), type(adjacency_np), type(np.asarray(seeds))
    return X, adjacency_np, np.asarray(seeds)


########################################################################################################
### MACHINE LEARNING FUNCTIONS ###
########################################################################################################
def do_pca(X, num_dimensions=3):
    """ Returns the result of runnning sklearn's PCA on matrix X with k=num_dimensions.

    Args:
        X (numpy.ndarray): the matrix to be reduced
        num_dimensions (int): number of dimensions to reduce to

    Returns:
        (numpy.ndarray): see above
    """
    pca = PCA(n_components=num_dimensions)
    return pca.fit_transform(X)


def get_kmeans_object(X, raw_seeds, num_clusters=10):
    """ Returns a sklearn kmeans object fit on data X with num_clusters clusters.

    Args:
        X (numpy.ndarray): the matrix to be clustered
        num_clusters (int): the number of clusters

    Returns:
        (sklearn.cluster.KMeans): see above
    """
    print "clustering with %d clusters" % num_clusters
    # avg_seeds = get_averaged_seeds(X, raw_seeds)
    seeds_30 = get_seed_points_list(X, raw_seeds)
    print seeds_30
    print seeds_30.shape
    return KMeans(n_clusters=num_clusters, init=seeds_30, max_iter=10000).fit(X)


def do_spectral_embedding(adj, num_dimensions=10):
    """ Returns the result of running sklearn's SpectralEmbedding on adjacency matrix adj.

    Args:
        adj (numpy.ndarray): a n by n square adjacency matrix
        num_dimensions (int): the number of dimensions to reduce to. Must be less than n

    Returns:
        (numpy.ndarray): see above

    """
    se = SpectralEmbedding(n_components=num_dimensions, affinity="precomputed")
    return se.fit_transform(adj)


def get_gmm_labels(X_c, raw_seeds, slow=True):
    """ Returns the labels from the result of applying an optimized GMM to matrix X_c, with initial seeds raw_seeds.

    Args:
        X_c (numpy.ndarray): a m by n matrix to fit the data to, where n > NUMBER_OF_CLUSTERS
        raw_seeds (numpy.ndarray): a NUMBER_OF_CLUSTERS by NUMBER_OF_SEEDS size matrix, containing the indices of each seed point for each label
        slow (bool): if True (default), then get_best_gmm will maximize EM iterations and initializations for best scoring but worse efficiency
    Returns:
        (numpy.ndarray): see above
    """
    # avg_seeds = get_averaged_seeds(X_c, raw_seeds)
    seeds_30 = get_seed_points_list(X_c, raw_seeds)
    # gmm = get_best_gmm(X_c, avg_seeds, slow=slow)
    gmm = mixture.GaussianMixture(n_components=30, max_iter=10000000, means_init=seeds_30)
    gmm.fit(X_c)
    return gmm.predict(X_c)
    # gmm_fake_labels = gmm.predict(X_c)

    # gmm_label_mapping = get_label_mapping(X_c, gmm.means_, raw_seeds)
    # gmm_labels = map_labels(gmm_fake_labels, gmm_label_mapping)

    # pprint(gmm_label_mapping)
    # return gmm_labels, gmm_fake_labels, gmm_label_mapping


def get_best_gmm(X_c, avg_seeds, slow=True):
    """ Returns the best GMM for data X_c. Uses avg_seeds as initial means for the gmm.

    Args:
        X_c (numpy.ndarray): a m by n matrix to fit the data to, where n > NUMBER_OF_CLUSTERS
        avg_seeds (numpy.ndarray): a NUMBER_OF_CLUSTERS by n size matrix containing the average of the seed values for each cluster
        slow (bool): if True (default), then maximizes EM iterations and initializations for best scoring but worse efficiency

    Returns:
        (sklearn.mixture.GaussianMixture): see above
     """
    greatest_bic = -np.infty
    bic = []
    cv_types = ['spherical', 'tied', 'diag', 'full']
    for cv_type in cv_types:
        # Fit a Gaussian mixture with EM
        gmm = mixture.GaussianMixture(n_components=NUMBER_OF_CLUSTERS, max_iter=10000 if slow else 100, n_init=10 if slow else 1, means_init=avg_seeds, covariance_type=cv_type)
        gmm.fit(X_c)
        bic.append(gmm.bic(X_c))
        if bic[-1] > greatest_bic:
            greatest_bic = bic[-1]
            best_gmm = gmm

    print bic
    print "Using GMM with covar_type: %s" % best_gmm.covariance_type
    return best_gmm


########################################################################################################
### OUR ACCESSORIES ###
########################################################################################################
def get_label_mapping(X_c, cluster_centers, seeds):
    """ Returns a mapping"""
    avg = get_averaged_seeds(X_c, seeds)
    label_distance_cluster_index = get_distances_from_seed_cluster_averages(avg, cluster_centers)

    model_to_real_label = dict()
    unassigned_labels = set(range(10))

    for real_label in range(10):
        cluster_index = label_distance_cluster_index[real_label][2]
        if cluster_index not in model_to_real_label and real_label in unassigned_labels:
            model_to_real_label[cluster_index] = real_label
            unassigned_labels.remove(real_label)

    for cluster_index in range(10):
        if cluster_index not in model_to_real_label:
            model_to_real_label[cluster_index] = unassigned_labels.pop()

    return model_to_real_label

def get_seed_labels(labels, raw_seeds, do_print=True):
    """ Prints the cluster assignments for each seed.
    Args:
        labels (numpy.ndarray): all 12000 labels, 0-9
        raw_seeds (numpy.ndarray): a NUMBER_OF_CLUSTERS by NUMBER_OF_SEEDS matrix
        do_print (bool): if true, prints extra info
    Returns:
        None
    """
    total_correct = 0.0
    for i, row in enumerate(raw_seeds):
        if do_print:
            print "Cluster: %d" % i
            print map(lambda seed: labels[seed - 1], row) # seed: 1 - 12000, indices 0-11999
        total_correct += sum(map(lambda seed: labels[seed - 1] == i, row))

    print total_correct

def get_seed_labels(labels, raw_seeds, do_print=True):
    """ Prints the cluster assignments for each seed.

    Args:
        labels (numpy.ndarray): all 12000 labels, 0-9
        raw_seeds (numpy.ndarray): a NUMBER_OF_CLUSTERS by NUMBER_OF_SEEDS matrix
        do_print (bool): if true, prints extra info

    Returns:
        None
    """
    total_correct = 0.0
    for i, row in enumerate(raw_seeds):
        if do_print:
            print "Cluster: %d" % i
            print map(lambda seed: labels[seed - 1], row) # seed: 1 - 12000, indices 0-11999
        total_correct += sum(map(lambda seed: labels[seed - 1] == i, row))

    print total_correct




########################################################################################################
### EXTRA FUNCTIONS FOR GETTING AVERAGE SEEDS
########################################################################################################
def get_seed_points_by_label(X, seed_numbers):
    """ Return a mapping from seed label to a list of each feature vector of each seed value

        Args:
            X (numpy.ndarray): a matrix that contains seed_number id's
            seed_numbers (numpy.ndarray): the seed matrix

        Returns:
            matrix
    """
    given_points_by_label = defaultdict(list)
    for i in range(NUMBER_OF_CLUSTERS):
        for j in range(NUMBER_OF_SEEDS):
            data_index = int(seed_numbers[i][j])
            given_points_by_label[i].append(X[data_index - 1])

    return given_points_by_label


def get_seed_points_list(X, seed_numbers):
    seeds_list = []
    for i in range(NUMBER_OF_CLUSTERS):
        for j in range(NUMBER_OF_SEEDS):
            data_index = int(seed_numbers[i][j])
            # print data_index
            seeds_list.append(X[data_index - 1])

    return np.asarray(seeds_list)

def get_averaged_seeds(Y, seeds):
    """Returns the feature average of each seed in Y
        Args:
            Y (numpy.ndarray): feature matrix
            seeds (numpy.ndarray): seeds
    """
    seed_numbers = np.asarray(seeds)
    # print seed_numbers
    seed_points_by_label = get_seed_points_by_label(Y, seed_numbers)
    # print seed_points_by_label[0]
    # print np.average(seed_points_by_label[0],axis=0)

    ### get average of given pre-classified points ###
    averaged_seeds = np.zeros((NUMBER_OF_CLUSTERS, Y.shape[1]))
    for label in range(NUMBER_OF_CLUSTERS):
        averaged_seeds[label] = (np.average(seed_points_by_label[label], axis=0))
    return averaged_seeds


def get_distances_from_seed_cluster_averages(averaged_seeds, cluster_centers):
    """ Returns a list of of averaged_seeds paired with the cluster_center
    each average seed is closest to

        ARgs:
            averaged_seeds (numpy.ndarray): averaged seed values
            cluster_centers (numpy.ndarray): cluster centers
    """
    ### GET DISTANCES FROM SEED CLUSTER AVERAGES ###
    avg_seed_distance_from_centroid_pairs = []

    for seed_label in range(NUMBER_OF_CLUSTERS):
        # print 'Using array: %s' % averaged_seeds[seed_label]

        distances = []
        for i, cluster_center in enumerate(cluster_centers):
            distance = np.linalg.norm(averaged_seeds[seed_label] - cluster_center)
            distances.append(distance)
            # if VERBOSE_FOR_CLUSTERING:
            #     print "\tchecking cluster_center %d: %s" % (i, cluster_center)
            #     print "\tdistance: %s" % distance

        min_distance = min(distances)
        closest_index = distances.index(min(distances))

        avg_seed_distance_from_centroid_pairs.append((seed_label, min_distance, closest_index))

    return avg_seed_distance_from_centroid_pairs


def map_labels(model_labels, model_to_real_label):
    """ maps cluster_indices to real_labels

        Args:
            model_labels (numpy.ndarray): labels produced by clustering
            model_to_real_label (numpy.ndarray): maping of real labels to model labels
    """

    labels = model_labels[:]
    labels = map(lambda ci: model_to_real_label[ci], labels)

    pprint(Counter(labels).most_common())
    return labels


### HELPER FUNCTIONS ###
def make_submission(labels, output_filename):
    """ Formats a csv with name [output_filename] for submission given labels"""
    with open(output_filename + ".csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["Id", "Category"])

        for i, lab in enumerate(labels):
            writer.writerow([i + 1, lab])


def plot_labels(x_points, y_points, labels):
    """ Makes a scatter plot of x_points and y_points and colors them
    given labels """
    plot_height = 5
    x_points = list(x_points) + range(NUMBER_OF_CLUSTERS)
    y_points = list(y_points) + [plot_height] * NUMBER_OF_CLUSTERS
    new_labs = labels[:]
    new_labs.extend(range(NUMBER_OF_CLUSTERS))

    plt.figure(figsize=(20, 10))

    plt.scatter(x=x_points, y=y_points, c=new_labs, s=20)
    plt.show()


def plot_two_dimensional_data(X):
    pass

def plot_eigenvalues_of_matrix(X):
    """Plots the eigenvalues of a matrix X """
    max_dim = X.shape[1] - 2
    sigma = np.cov(X)
    vals, _ = la.eigs(sigma, k=max_dim)

    vals = sorted(vals.real, reverse=True)
    print vals

    eigs_sum = sum(vals)
    total = 0.0

    for i, val in enumerate(vals):
        total += val
        if total > 0.95 * eigs_sum:
            break



    plt.scatter(x=range(1, len(vals) + 1), y=vals)
    plt.plot([i, i], [0, max(vals) * 1.1])
    plt.show()
