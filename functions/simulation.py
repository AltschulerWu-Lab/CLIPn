"""
Function to generate simulation profiles with multiple datasets where we controlled sample categories, feature
dimensions, noise rates, and shared labels across datasets for quantitative assessment
Feng Bao @ UCSF, 2024
"""

import numpy as np
import random
from copy import deepcopy
import pandas as pd


def assay_simulator(n_sample, n_cluster, n_assay, cluster_observe_ratio=0.8,
                    p_max=1000, p_min=100,
                    sigma_max=0.5, sigma_min=0.1,
                    rho_max=0.6, rho_min=0.1,
                    dim_latent=100,
                    dim_intermediate=64, random_seed=2023
                    ):
    """
    Generate simulated data for n celllines

    :param n_sample: total number of samples in all screen assays
    :param n_cluster: number of drug categories in total
    :param n_assay: number of different assays
    :param cluster_observe_ratio: ratio of total drug categories observed from a single assay
    :param p_max: maximal feature dimensions in datasets
    :param p_min: minimal feature dimensions in datasets
    :param sigma_max: maximal noise level
    :param sigma_min: minimal noise level
    :param rho_max: maximal drop-out rate
    :param rho_min: mininal drop-out rate
    :param dim_latent: dimension of latent code to generate profiles
    :param dim_intermediate: dimension of neural network layer to generate cell features

    :return: data dictionary
    """

    random.seed(random_seed)

    data = {}

    # Generate ground truth latent features Z.
    """ generation of true cluster labels """
    cluster_ids = np.array([random.choice(range(n_cluster)) for i in range(n_sample)])
    data["true_cluster"] = cluster_ids

    # Generate latent code.
    Z = np.zeros([dim_latent, n_sample])
    for id in list(set(cluster_ids)):
        idxs = cluster_ids == id
        cluster_mu = np.random.random([dim_latent]) - 0.5
        Z[:, idxs] = np.random.multivariate_normal(
            mean=cluster_mu, cov=0.1 * np.eye(dim_latent), size=idxs.sum()
        ).transpose()
    data["Z"] = Z.T

    for i in range(n_assay):
        p = random.randint(p_min, p_max)
        sigma = random.uniform(sigma_min, sigma_max)
        rho = random.uniform(rho_min, rho_max)

        A_1 = np.random.random([dim_intermediate, dim_latent]) - 0.5

        noise = np.random.normal(0, sigma, size=[p, n_sample])

        X_1 = np.dot(A_1, Z)
        X_2 = 1 / (1 + np.exp(-X_1))

        A_2 = np.random.random([p, dim_intermediate]) - 0.5
        X_3 = (np.dot(A_2, X_2) + noise).transpose()
        X = 1 / (1 + np.exp(-X_3))

        Y = deepcopy(X)
        rand_matrix = np.random.random(Y.shape)
        zero_mask = rand_matrix < rho
        Y[zero_mask] = 0

        n_to_drop = int(n_cluster * (1 - cluster_observe_ratio))
        id_to_drop = random.sample(range(n_cluster), n_to_drop)

        label = pd.Series(cluster_ids)
        X = pd.DataFrame(X)
        Y = pd.DataFrame(Y)

        label_removed = label[~label.isin(id_to_drop)]
        Y_removed = Y.loc[label_removed.index, :]

        label_left = label[label.isin(id_to_drop)]
        Y_left = Y.loc[label_left.index, :]

        data["dataset_" + str(i) + "_feature"] = Y_removed.values
        data["dataset_" + str(i) + "_label"] = label_removed.values
        data["dataset_" + str(i) + "_feature_left"] = Y_left.values
        data["dataset_" + str(i) + "_label_left"] = label_left.values

        data["dataset_" + str(i) + "_feature_full"] = Y.values
        data["dataset_" + str(i) + "_label_full"] = label.values

        data["dataset_" + str(i) + "_noise"] = sigma
        data["dataset_" + str(i) + "_dropout"] = rho

    return data
