import umap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from mycolorpy import colorlist as mcp
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold


def gen_color(cmap="tab10", label_n=10):
    # Get the "tab10" colormap
    cmap = plt.get_cmap(cmap)

    colors = [cmap(i / (label_n - 1)) for i in range(label_n)]
    return colors


def recall_k(Z_train, Z_test, y_train, y_test, k=[1, 5, 10, 20, 50]):
    """
    Calculate recall@k for the given latent space and labels.
    Z_train: A dictionary of latent spaces for training datasets.
    Z_test: A dictionary of latent spaces for testing datasets.

    """

    Z_train_mat = np.array([])
    y_train_vec = np.array([])
    Z_test_mat = np.array([])
    y_test_vec = np.array([])

    for id in Z_train.keys():
        Z_train_mat = (
            np.vstack((Z_train_mat, Z_train[id])) if Z_train_mat.size else Z_train[id]
        )
        y_train_vec = np.concatenate((y_train_vec, y_train[id]))

    for id in Z_test.keys():
        Z_test_mat = (
            np.vstack((Z_test_mat, Z_test[id])) if Z_test_mat.size else Z_test[id]
        )
        y_test_vec = np.concatenate((y_test_vec, y_test[id]))

    # Calculate using inner product
    sim = Z_test_mat @ Z_train_mat.T

    rank = np.argsort(-sim, axis=1)  # Enable descending ordering
    label_mat_train = np.vstack([y_train_vec] * Z_test_mat.shape[0])
    label_rank = np.take_along_axis(label_mat_train, rank, axis=1)

    compare = label_rank == y_test_vec[:, np.newaxis]
    recall = np.zeros(len(k))

    for i in range(len(k)):
        recall[i] = np.any(compare[:, : k[i]], axis=1).sum() / compare.shape[0]
    return recall


def recall_k_flex(Z_train, Z_test, y_train, y_test, k=[1, 5, 10, 20, 50]):
    Z_train_mat = np.array([])
    y_train_vec = np.array([])
    Z_test_mat = np.array([])
    y_test_vec = np.array([])

    for id in Z_train.keys():
        Z_train_mat = (
            np.vstack((Z_train_mat, Z_train[id])) if Z_train_mat.size else Z_train[id]
        )
        y_train_vec = np.concatenate((y_train_vec, y_train[id]))

    for id in Z_test.keys():
        Z_test_mat = (
            np.vstack((Z_test_mat, Z_test[id])) if Z_test_mat.size else Z_test[id]
        )
        y_test_vec = np.concatenate((y_test_vec, y_test[id]))

    # Calculate using inner product
    sim = Z_test_mat @ Z_train_mat.T

    rank = np.argsort(-sim, axis=1)  # Enable descending ordering
    label_mat_train = np.vstack([y_train_vec] * Z_test_mat.shape[0])
    label_rank = np.take_along_axis(label_mat_train, rank, axis=1)

    compare = (
        (label_rank == y_test_vec[:, np.newaxis])
        + (label_rank == (y_test_vec[:, np.newaxis] + 1))
        + (label_rank == (y_test_vec[:, np.newaxis] - 1))
    )
    recall = np.zeros(len(k))

    for i in range(len(k)):
        recall[i] = np.any(compare[:, : k[i]], axis=1).sum() / compare.shape[0]
    return recall


def recall_k_flex_reconstruct(
    Z_train_mat, Z_test_mat, y_train_vec, y_test_vec, k=[1, 5, 10, 20, 50]
):
    # Calculate using inner product
    sim = Z_test_mat @ Z_train_mat.T

    rank = np.argsort(-sim, axis=1)  # Enable descending ordering
    label_mat_train = np.vstack([y_train_vec] * Z_test_mat.shape[0])
    label_rank = np.take_along_axis(label_mat_train, rank, axis=1)

    compare = (
        (label_rank == y_test_vec[:, np.newaxis])
        + (label_rank == (y_test_vec[:, np.newaxis] + 1))
        + (label_rank == (y_test_vec[:, np.newaxis] - 1))
    )
    recall = np.zeros(len(k))

    for i in range(len(k)):
        recall[i] = np.any(compare[:, : k[i]], axis=1).sum() / compare.shape[0]
    return recall


def confusion_matrix(Z_train, y_train, Z_test, y_test):
    Z_train_mat, y_train_vec, ID_train = dict_to_array(Z_train, y_train)
    Z_test_mat, y_test_vec, ID_test = dict_to_array(Z_test, y_test)

    # Calculate using inner product
    sim = Z_test_mat @ Z_train_mat.T

    rank = np.argsort(-sim, axis=1)  # Enable descending ordering
    label_mat_train = np.vstack([y_train_vec] * Z_test_mat.shape[0])
    label_rank = np.take_along_axis(label_mat_train, rank, axis=1)

    top_1_label = label_rank[:, 0]

    conf_mat = np.zeros([10, 10])
    for i in range(10):
        select_top = top_1_label[y_test_vec == i]
        conf_mat[i, :] = (
            np.bincount(select_top.astype(int), minlength=10) / select_top.size
        )

    return conf_mat


def confusion_matrix_eud(Z_train, y_train, Z_test, y_test):
    Z_train_mat, y_train_vec, ID_train = dict_to_array(Z_train, y_train)
    Z_test_mat, y_test_vec, ID_test = dict_to_array(Z_test, y_test)

    # Calculate using inner product
    sim = pairwise_distances(Z_test_mat, Z_train_mat, metric="euclidean")

    rank = np.argsort(sim, axis=1)  # Enable descending ordering
    label_mat_train = np.vstack([y_train_vec] * Z_test_mat.shape[0])
    label_rank = np.take_along_axis(label_mat_train, rank, axis=1)

    top_1_label = label_rank[:, 0]

    conf_mat = np.zeros([10, 10])
    for i in range(10):
        select_top = top_1_label[y_test_vec == i]
        conf_mat[i, :] = (
            np.bincount(select_top.astype(int), minlength=10) / select_top.size
        )

    return conf_mat


def recall_k_by_category(Z_train, Z_test, y_train, y_test, k=[1, 5, 10, 20, 50]):
    Z_train_mat = np.array([])
    y_train_vec = np.array([])
    Z_test_mat = np.array([])
    y_test_vec = np.array([])

    for id in Z_train.keys():
        Z_train_mat = (
            np.vstack((Z_train_mat, Z_train[id])) if Z_train_mat.size else Z_train[id]
        )
        y_train_vec = np.concatenate((y_train_vec, y_train[id]))

    for id in Z_test.keys():
        Z_test_mat = (
            np.vstack((Z_test_mat, Z_test[id])) if Z_test_mat.size else Z_test[id]
        )
        y_test_vec = np.concatenate((y_test_vec, y_test[id]))

    # Calculate using inner product
    sim = Z_test_mat @ Z_train_mat.T

    rank = np.argsort(-sim, axis=1)  # Enable descending ordering
    label_mat_train = np.vstack([y_train_vec] * Z_test_mat.shape[0])
    label_rank = np.take_along_axis(label_mat_train, rank, axis=1)

    compare = label_rank == y_test_vec[:, np.newaxis]
    recall = pd.DataFrame(index=k, columns=np.unique(y_test_vec))

    for i in k:
        for j in np.unique(y_test_vec):
            recall.loc[i, j] = (
                np.any(compare[y_test_vec == j, :i], axis=1).sum()
                / (y_test_vec == j).sum()
            )

    return recall


def linear_classifier_f1(X_train: dict, y_train: dict, X_test: dict, y_test: dict):
    train_keys = list(X_train.keys())

    combined_X_train = np.empty((0, X_train[train_keys[0]].shape[1]))
    combine_y_train = np.array([])

    for i in train_keys:
        combined_X_train = np.concatenate([combined_X_train, X_train[i]], axis=0)
        combine_y_train = np.concatenate([combine_y_train, y_train[i]], axis=0)

    test_keys = list(X_test.keys())

    combined_X_test = np.empty((0, X_test[0].shape[1]))
    combine_y_test = np.array([])

    for i in test_keys:
        combined_X_test = np.concatenate([combined_X_test, X_test[i]], axis=0)
        combine_y_test = np.concatenate([combine_y_test, y_test[i]], axis=0)

    # Create a logistic regression classifier
    clf = LogisticRegression(solver="liblinear", multi_class="auto")

    clf.fit(combined_X_train, combine_y_train)
    y_pred = clf.predict(combined_X_test)
    f1 = f1_score(combine_y_test, y_pred, average="macro")

    return f1


def linear_classifier_f1_by_category(
    X_train: dict, y_train: dict, X_test: dict, y_test: dict
):
    train_keys = list(X_train.keys())

    combined_X_train = np.empty((0, X_train[train_keys[0]].shape[1]))
    combine_y_train = np.array([])

    for i in train_keys:
        combined_X_train = np.concatenate([combined_X_train, X_train[i]], axis=0)
        combine_y_train = np.concatenate([combine_y_train, y_train[i]], axis=0)

    test_keys = list(X_test.keys())

    combined_X_test = np.empty((0, X_test[test_keys[0]].shape[1]))
    combine_y_test = np.array([])

    for i in test_keys:
        combined_X_test = np.concatenate([combined_X_test, X_test[i]], axis=0)
        combine_y_test = np.concatenate([combine_y_test, y_test[i]], axis=0)

    # Create a logistic regression classifier
    clf = LogisticRegression(solver="liblinear", multi_class="auto")

    clf.fit(combined_X_train, combine_y_train)
    y_pred = clf.predict(combined_X_test)
    f1 = f1_score(combine_y_test, y_pred, average=None)

    return f1


def umap_scatter(Z: dict, y: dict):
    id = list(Z.keys())

    combined_Z = np.empty((0, Z[id[0]].shape[1]))
    combine_label = np.array([])
    combine_dataID = np.array([])
    train_ID = np.array([])

    for i in id:
        combined_Z = np.concatenate([combined_Z, Z[i]], axis=0)
        combine_label = np.concatenate([combine_label, y[i]], axis=0)
        combine_dataID = np.concatenate(
            [combine_dataID, np.ones_like(y[i]) * i], axis=0
        )
        train_ID = np.concatenate([train_ID, ["train"] * np.size(y[i])], axis=0)

    dataset_n = len(np.unique(combine_dataID))
    label_n = len(np.unique(combine_label))

    colors = gen_color(cmap="tab10", n=label_n)

    color_mapping = dict(zip(range(label_n), colors))

    colors = gen_color(cmap="Accent", n=dataset_n)

    new_color_mapping = dict(zip(range(dataset_n), colors))

    embed = umap.UMAP().fit_transform(combined_Z)
    plot_pd = pd.DataFrame(
        {
            "umap-1": embed[:, 0],
            "umap-2": embed[:, 1],
            "label": combine_label,
            "dataset": combine_dataID,
        }
    )

    plot_pd["Color_label"] = plot_pd["label"].map(color_mapping)
    plot_pd["Color_dataset"] = plot_pd["dataset"].map(new_color_mapping)

    # Create figure with 1 row, 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

    # First plot
    for label in plot_pd.label.unique():
        ax1.scatter(
            plot_pd[plot_pd.label == label]["umap-1"],
            plot_pd[plot_pd.label == label]["umap-2"],
            c=color_mapping[int(label)],
            s=0.5,
        )
    #
    # sns.scatterplot(data=plot_pd, x='umap-1', y='umap-2', hue='label',
    #                 s=1, ax=ax1, legend=False)
    ax1.axis("square")
    ax1.set_title("Color by labels")

    # Second plot
    for label in plot_pd.dataset.unique():
        ax2.scatter(
            plot_pd[plot_pd.dataset == label]["umap-1"],
            plot_pd[plot_pd.dataset == label]["umap-2"],
            c=new_color_mapping[int(label)],
            s=0.5,
        )

    # sns.scatterplot(data=plot_pd, x='umap-1', y='umap-2',
    #                 hue='dataset', cmap='Dark2', s=1, ax=ax2, legend=False)
    ax2.axis("square")
    ax2.set_title("Color by datasets")

    # Adjust space between plots
    plt.subplots_adjust(wspace=0.3)

    plt.show()


def pca_scatter(Z: dict, y: dict):
    id = list(Z.keys())

    combined_Z = np.empty((0, Z[id[0]].shape[1]))
    combine_label = np.array([])
    combine_dataID = np.array([])
    train_ID = np.array([])

    colors = gen_color(cmap="tab10", n=10)

    color_mapping = dict(zip(range(10), colors))

    colors = gen_color(cmap="Accent", n=10)

    new_color_mapping = dict(zip(range(10), colors))

    for i in id:
        combined_Z = np.concatenate([combined_Z, Z[i]], axis=0)
        combine_label = np.concatenate([combine_label, y[i]], axis=0)
        combine_dataID = np.concatenate(
            [combine_dataID, np.ones_like(y[i]) * i], axis=0
        )
        train_ID = np.concatenate([train_ID, ["train"] * np.size(y[i])], axis=0)
    pca = PCA(n_components=2)
    embed = pca.fit_transform(combined_Z)
    plot_pd = pd.DataFrame(
        {
            "umap-1": embed[:, 0],
            "umap-2": embed[:, 1],
            "label": combine_label,
            "dataset": combine_dataID,
        }
    )

    plot_pd["Color_label"] = plot_pd["label"].map(color_mapping)
    plot_pd["Color_dataset"] = plot_pd["dataset"].map(new_color_mapping)

    # Create figure with 1 row, 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

    # First plot
    for label in plot_pd.label.unique():
        ax1.scatter(
            plot_pd[plot_pd.label == label]["umap-1"],
            plot_pd[plot_pd.label == label]["umap-2"],
            c=color_mapping[int(label)],
            s=0.5,
        )
    #
    # sns.scatterplot(data=plot_pd, x='umap-1', y='umap-2', hue='label',
    #                 s=1, ax=ax1, legend=False)
    # ax1.axis('square')
    ax1.set_title("Color by labels")

    # Second plot
    for label in plot_pd.dataset.unique():
        ax2.scatter(
            plot_pd[plot_pd.dataset == label]["umap-1"],
            plot_pd[plot_pd.dataset == label]["umap-2"],
            c=new_color_mapping[int(label)],
            s=0.5,
        )

    # sns.scatterplot(data=plot_pd, x='umap-1', y='umap-2',
    #                 hue='dataset', cmap='Dark2', s=1, ax=ax2, legend=False)
    # ax2.axis('square')
    ax2.set_title("Color by datasets")

    # Adjust space between plots
    plt.subplots_adjust(wspace=0.3)

    plt.show()


def umap_scatter_highlight(Z: dict, y: dict, select_label):
    id = list(Z.keys())

    combined_Z = np.empty((0, Z[id[0]].shape[1]))
    combine_label = np.array([])
    combine_dataID = np.array([])
    train_ID = np.array([])

    colors = gen_color(cmap="tab10", n=10)

    color_mapping = dict(zip(range(10), colors))

    colors = gen_color(cmap="Accent", n=8)

    new_color_mapping = dict(zip(range(8), colors))

    for i in id:
        combined_Z = np.concatenate([combined_Z, Z[i]], axis=0)
        combine_label = np.concatenate([combine_label, y[i]], axis=0)
        combine_dataID = np.concatenate(
            [combine_dataID, np.ones_like(y[i]) * i], axis=0
        )
        train_ID = np.concatenate([train_ID, ["train"] * np.size(y[i])], axis=0)

    embed = umap.UMAP().fit_transform(combined_Z)
    plot_pd = pd.DataFrame(
        {
            "umap-1": embed[:, 0],
            "umap-2": embed[:, 1],
            "label": combine_label,
            "dataset": combine_dataID,
        }
    )

    plot_pd["Color_label"] = plot_pd["label"].map(color_mapping)
    plot_pd["Color_dataset"] = plot_pd["dataset"].map(new_color_mapping)

    # Create figure with 1 row, 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

    # First plot
    for label in plot_pd.label.unique():
        ax1.scatter(
            plot_pd[plot_pd.label == label]["umap-1"],
            plot_pd[plot_pd.label == label]["umap-2"],
            c=color_mapping[int(label)],
            s=0.5,
        )
    ax1.scatter(
        plot_pd[plot_pd.label == select_label]["umap-1"],
        plot_pd[plot_pd.label == select_label]["umap-2"],
        c="k",
        s=2,
        marker="^",
    )
    # sns.scatterplot(data=plot_pd, x='umap-1', y='umap-2', hue='label',
    #                 s=1, ax=ax1, legend=False)
    ax1.axis("square")
    ax1.set_title("Color by labels")

    # Second plot
    for label in plot_pd.dataset.unique():
        ax2.scatter(
            plot_pd[plot_pd.dataset == label]["umap-1"],
            plot_pd[plot_pd.dataset == label]["umap-2"],
            c=new_color_mapping[int(label)],
            s=0.5,
        )
    # ax2.scatter(plot_pd[plot_pd.dataset == select_label]['umap-1'],
    #             plot_pd[plot_pd.dataset == select_label]['umap-2'],
    #             c=new_color_mapping[int(select_label)], s=2, marker='^', edgecolors='w', linewidths=0.1)
    # sns.scatterplot(data=plot_pd, x='umap-1', y='umap-2',
    #                 hue='dataset', cmap='Dark2', s=1, ax=ax2, legend=False)
    ax2.axis("square")
    ax2.set_title("Color by datasets")

    # Adjust space between plots
    plt.subplots_adjust(wspace=0.3)

    plt.show()


def umap_scatter_save(Z: dict, y: dict, path: str):
    id = list(Z.keys())

    combined_Z = np.empty((0, Z[id[0]].shape[1]))
    combine_label = np.array([])
    combine_dataID = np.array([])
    train_ID = np.array([])

    colors = gen_color(cmap="tab10", n=10)

    color_mapping = dict(zip(range(10), colors))

    colors = gen_color(cmap="Accent", n=8)

    new_color_mapping = dict(zip(range(8), colors))

    for i in id:
        combined_Z = np.concatenate([combined_Z, Z[i]], axis=0)
        combine_label = np.concatenate([combine_label, y[i]], axis=0)
        combine_dataID = np.concatenate(
            [combine_dataID, np.ones_like(y[i]) * i], axis=0
        )
        train_ID = np.concatenate([train_ID, ["train"] * np.size(y[i])], axis=0)

    embed = umap.UMAP().fit_transform(combined_Z)
    plot_pd = pd.DataFrame(
        {
            "umap-1": embed[:, 0],
            "umap-2": embed[:, 1],
            "label": combine_label,
            "dataset": combine_dataID,
        }
    )

    plot_pd["Color_label"] = plot_pd["label"].map(color_mapping)
    plot_pd["Color_dataset"] = plot_pd["dataset"].map(new_color_mapping)

    # Create figure with 1 row, 2 columns
    plt.figure(figsize=(5, 5))

    kf = KFold(n_splits=20)
    for train_index, test_index in kf.split(plot_pd):
        plot_pd_sub = plot_pd.iloc[test_index]
        # First plot
        for label in plot_pd_sub.label.unique():
            plt.scatter(
                plot_pd_sub[plot_pd_sub.label == label]["umap-1"],
                plot_pd_sub[plot_pd_sub.label == label]["umap-2"],
                c=color_mapping[int(label)],
                s=0.2,
            )
    plt.axis("square")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path + "_labels.png", dpi=300)
    # ax1.set_title('Color by labels')

    # Second plot
    plt.figure(figsize=(5, 5))
    kf = KFold(n_splits=20)
    for train_index, test_index in kf.split(plot_pd):
        plot_pd_sub = plot_pd.iloc[test_index]
        # First plot
        for label in plot_pd_sub.dataset.unique():
            plt.scatter(
                plot_pd_sub[plot_pd_sub.dataset == label]["umap-1"],
                plot_pd_sub[plot_pd_sub.dataset == label]["umap-2"],
                c=new_color_mapping[int(label)],
                s=0.2,
            )

    plt.axis("square")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path + "_datasets.png", dpi=300)


def umap_scatter_save_highlight(Z: dict, y: dict, path: str, label_highlight):
    id = list(Z.keys())

    combined_Z = np.empty((0, Z[id[0]].shape[1]))
    combine_label = np.array([])
    combine_dataID = np.array([])
    train_ID = np.array([])

    colors = gen_color(cmap="tab10", n=10)

    color_mapping = dict(zip(range(10), colors))

    colors = gen_color(cmap="Accent", n=8)

    new_color_mapping = dict(zip(range(8), colors))

    for i in id:
        combined_Z = np.concatenate([combined_Z, Z[i]], axis=0)
        combine_label = np.concatenate([combine_label, y[i]], axis=0)
        combine_dataID = np.concatenate(
            [combine_dataID, np.ones_like(y[i]) * i], axis=0
        )
        train_ID = np.concatenate([train_ID, ["train"] * np.size(y[i])], axis=0)

    embed = umap.UMAP().fit_transform(combined_Z)
    plot_pd = pd.DataFrame(
        {
            "umap-1": embed[:, 0],
            "umap-2": embed[:, 1],
            "label": combine_label,
            "dataset": combine_dataID,
        }
    )

    plot_pd["Color_label"] = plot_pd["label"].map(color_mapping)
    plot_pd["Color_dataset"] = plot_pd["dataset"].map(new_color_mapping)

    # Create figure with 1 row, 2 columns
    plt.figure(figsize=(5, 5))

    kf = KFold(n_splits=20)
    for train_index, test_index in kf.split(plot_pd):
        plot_pd_sub = plot_pd.iloc[test_index]
        # First plot
        for label in plot_pd_sub.label.unique():
            plt.scatter(
                plot_pd_sub[plot_pd_sub.label == label]["umap-1"],
                plot_pd_sub[plot_pd_sub.label == label]["umap-2"],
                c=color_mapping[int(label)],
                s=0.2,
            )
    plt.scatter(
        plot_pd[plot_pd.label == label_highlight]["umap-1"],
        plot_pd[plot_pd.label == label_highlight]["umap-2"],
        c="k",
        s=2,
        marker="^",
    )
    plt.axis("square")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path + "_labels.png", dpi=300)
    # ax1.set_title('Color by labels')

    # Second plot
    plt.figure(figsize=(5, 5))
    for label in plot_pd.dataset.unique():
        plt.scatter(
            plot_pd[plot_pd.dataset == label]["umap-1"],
            plot_pd[plot_pd.dataset == label]["umap-2"],
            c=new_color_mapping[int(label)],
            s=0.5,
        )

    plt.axis("square")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path + "_datasets.png", dpi=300)


def label_freq(Z, y, n=100):
    nbrs = NearestNeighbors(n_neighbors=n + 1, algorithm="ball_tree").fit(Z)
    distances, indices = nbrs.kneighbors(Z)
    nn_labels = y[indices[:, 1:]]

    label_freq = np.zeros((len(y), len(np.unique(y))))

    for i in range(len(np.unique(y))):
        label_freq[:, i] = np.sum(nn_labels == i, axis=1)

    label_freq = label_freq / n

    # entropy=np.sum(-label_freq*np.log(label_freq),axis=1)

    return label_freq


def dict_to_array(Z: dict, y: dict):
    id = list(Z.keys())

    combined_Z = np.empty((0, Z[id[0]].shape[1]))
    combine_label = np.array([])
    combine_dataID = np.array([])

    for i in id:
        combined_Z = np.concatenate([combined_Z, Z[i]], axis=0)
        combine_label = np.concatenate([combine_label, y[i]], axis=0)
        combine_dataID = np.concatenate(
            [combine_dataID, np.ones_like(y[i]) * i], axis=0
        )

    return combined_Z, combine_label, combine_dataID


def construct_null_label_distribution(y, id):
    label_observe = np.zeros((len(np.unique(y)), len(np.unique(id))))  # label x dataset

    for i in range(len(np.unique(y))):
        id_select = id[y == i]
        unique, counts = np.unique(id_select, return_counts=True)

        label_observe[i, unique.astype(int)] = counts

    label_observe = label_observe / label_observe.sum(axis=1, keepdims=True)

    y_id_map = label_observe[y.astype(int), :]
    return y_id_map


# def construct_null_label_distribution_hypoxia(y, id):
#     label_observe = np.zeros((len(np.unique(y)), len(np.unique(id))))  # label x dataset
#     for i in range(len(np.unique(y))):
#         id_select = id[y == i]
#         label_observe[i, np.unique(id_select).astype(int)] = 1
#
#     label_observe = label_observe / label_observe.sum(axis=1, keepdims=True)
#
#     y_id_map = label_observe[y.astype(int), :]
#     return y_id_map


def umap_scatter_train_test(Z: dict, y: dict, Z_test: dict, y_test: dict):
    id = list(Z.keys())

    combined_Z = np.empty((0, Z[id[0]].shape[1]))
    combine_label = np.array([])
    combine_dataID = np.array([])
    train_ID = np.array([])

    for i in id:
        combined_Z = np.concatenate([combined_Z, Z[i]], axis=0)
        combine_label = np.concatenate([combine_label, y[i]], axis=0)
        combine_dataID = np.concatenate(
            [combine_dataID, np.ones_like(y[i]) * i], axis=0
        )
        train_ID = np.concatenate([train_ID, ["train"] * np.size(y[i])], axis=0)

    id = list(Z_test.keys())

    for i in id:
        combined_Z = np.concatenate([combined_Z, Z_test[i]], axis=0)
        combine_label = np.concatenate([combine_label, y_test[i]], axis=0)
        combine_dataID = np.concatenate(
            [combine_dataID, np.ones_like(y_test[i]) * i], axis=0
        )
        train_ID = np.concatenate([train_ID, ["test"] * np.size(y_test[i])], axis=0)

    embed = umap.UMAP().fit_transform(combined_Z)
    plot_pd = pd.DataFrame(
        {
            "umap-1": embed[:, 0],
            "umap-2": embed[:, 1],
            "label": combine_label.astype(str),
            "dataset": combine_dataID.astype(str),
            "train": train_ID,
        }
    )

    # Create figure with 1 row, 2 columns
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9, 3))

    # First plot
    sns.scatterplot(
        data=plot_pd,
        x="umap-1",
        y="umap-2",
        hue="label",
        cmap="tab10",
        s=1,
        ax=ax1,
        legend=False,
    )
    ax1.axis("square")
    ax1.set_title("Color by labels")

    # Second plot
    sns.scatterplot(
        data=plot_pd,
        x="umap-1",
        y="umap-2",
        hue="dataset",
        cmap="Dark2",
        s=1,
        ax=ax2,
        legend=False,
    )
    ax2.axis("square")
    ax2.set_title("Color by datasets")

    # Third plot
    sns.scatterplot(
        data=plot_pd,
        x="umap-1",
        y="umap-2",
        hue="train",
        cmap="Dark2",
        s=1,
        ax=ax3,
        legend=False,
    )
    ax3.axis("square")
    ax3.set_title("Color by train/test")

    # Adjust space between plots
    plt.subplots_adjust(wspace=0.3)

    plt.show()


def pca_scatter_train_test(Z: dict, y: dict, Z_test: dict, y_test: dict):
    id = list(Z.keys())

    combined_Z = np.empty((0, Z[id[0]].shape[1]))
    combine_label = np.array([])
    combine_dataID = np.array([])
    train_ID = np.array([])

    for i in id:
        combined_Z = np.concatenate([combined_Z, Z[i]], axis=0)
        combine_label = np.concatenate([combine_label, y[i]], axis=0)
        combine_dataID = np.concatenate(
            [combine_dataID, np.ones_like(y[i]) * i], axis=0
        )
        train_ID = np.concatenate([train_ID, ["train"] * np.size(y[i])], axis=0)

    id = list(Z_test.keys())

    for i in id:
        combined_Z = np.concatenate([combined_Z, Z_test[i]], axis=0)
        combine_label = np.concatenate([combine_label, y_test[i]], axis=0)
        combine_dataID = np.concatenate(
            [combine_dataID, np.ones_like(y_test[i]) * i], axis=0
        )
        train_ID = np.concatenate([train_ID, ["test"] * np.size(y_test[i])], axis=0)
    pc = PCA(n_components=2)
    embed = pc.fit_transform(combined_Z)
    plot_pd = pd.DataFrame(
        {
            "PC-1": embed[:, 0],
            "PC-2": embed[:, 1],
            "label": combine_label.astype(str),
            "dataset": combine_dataID.astype(str),
            "train": train_ID,
        }
    )

    colors = gen_color(cmap="tab10", n=10)

    color_mapping = dict(zip(range(10), colors))

    # Create figure with 1 row, 2 columns
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9, 3))

    # First plot
    sns.scatterplot(
        data=plot_pd,
        x="PC-1",
        y="PC-2",
        hue="label",
        hue_order=np.sort(plot_pd.label.unique()),
        cmap="tab10",
        s=1,
        ax=ax1,
        legend=False,
    )
    ax1.axis("square")
    ax1.set_title("Color by labels")

    # Second plot
    sns.scatterplot(
        data=plot_pd,
        x="PC-1",
        y="PC-2",
        hue="dataset",
        cmap="Dark2",
        s=1,
        ax=ax2,
        legend=False,
    )
    ax2.axis("square")
    ax2.set_title("Color by datasets")

    # Third plot
    sns.scatterplot(
        data=plot_pd,
        x="PC-1",
        y="PC-2",
        hue="train",
        cmap="Dark2",
        s=1,
        ax=ax3,
        legend=False,
    )
    ax3.axis("square")
    ax3.set_title("Color by train/test")

    # Adjust space between plots
    plt.subplots_adjust(wspace=0.3)

    plt.show()


def pca_scatter_hypoxia(Z: dict, y: dict):
    label_mapping = dict(
        zip(
            [
                "Normoxia",
                "Hypoxia_6h",
                "Hypoxia_24h",
                "Hypoxia_48h",
                "Hypoxia_72h",
                "Hypoxia_6d",
                "Hypoxia_10d",
            ],
            range(7),
        )
    )

    id = list(Z.keys())

    combined_Z = np.empty((0, Z[id[0]].shape[1]))
    combine_label = np.array([])
    combine_dataID = np.array([])
    train_ID = np.array([])

    color_1 = gen_color(cmap="winter_r", n=6)
    color_2 = gen_color(cmap="Greys", n=3)
    colors = [color_2[1]] + color_1[:3] + ["#00a6fb", "#0466c8", "#03045e"]

    color_mapping = dict(zip(range(10), colors))

    colors = gen_color(cmap="Accent", n=6)

    new_color_mapping = dict(zip(range(10), colors))

    for i in id:
        combined_Z = np.concatenate([combined_Z, Z[i]], axis=0)
        combine_label = np.concatenate([combine_label, y[i]], axis=0)
        combine_dataID = np.concatenate(
            [combine_dataID, np.ones_like(y[i]) * i], axis=0
        )
        train_ID = np.concatenate([train_ID, ["train"] * np.size(y[i])], axis=0)
    pca = PCA(n_components=2)
    embed = pca.fit_transform(combined_Z)
    plot_pd = pd.DataFrame(
        {
            "umap-1": embed[:, 0],
            "umap-2": embed[:, 1],
            "label": combine_label,
            "dataset": combine_dataID,
        }
    )

    plot_pd["Color_label"] = plot_pd["label"].map(color_mapping)
    plot_pd["Color_dataset"] = plot_pd["dataset"].map(new_color_mapping)

    # Create figure with 1 row, 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

    # First plot
    for label in plot_pd.label.unique():
        ax1.scatter(
            plot_pd[plot_pd.label == label]["umap-1"],
            plot_pd[plot_pd.label == label]["umap-2"],
            c=color_mapping[int(label)],
            s=0.5,
        )

    ax1.axis("square")
    ax1.set_title("Color by labels")

    # Second plot
    for label in plot_pd.dataset.unique():
        ax2.scatter(
            plot_pd[plot_pd.dataset == label]["umap-1"],
            plot_pd[plot_pd.dataset == label]["umap-2"],
            c=new_color_mapping[int(label)],
            s=0.5,
        )

    ax2.axis("square")
    ax2.set_title("Color by datasets")

    # Adjust space between plots
    plt.subplots_adjust(wspace=0.3)

    plt.show()


def pca_scatter_hypoxia_save(Z: dict, y: dict, path: str):
    label_mapping = dict(
        zip(
            [
                "Normoxia",
                "Hypoxia_6h",
                "Hypoxia_24h",
                "Hypoxia_48h",
                "Hypoxia_72h",
                "Hypoxia_6d",
                "Hypoxia_10d",
            ],
            range(7),
        )
    )

    id = list(Z.keys())

    combined_Z = np.empty((0, Z[id[0]].shape[1]))
    combine_label = np.array([])
    combine_dataID = np.array([])
    train_ID = np.array([])

    color_1 = gen_color(cmap="winter_r", n=6)
    color_2 = gen_color(cmap="Greys", n=3)
    colors = [color_2[1]] + color_1[:3] + ["#00a6fb", "#0466c8", "#03045e"]

    color_mapping = dict(zip(range(7), colors))

    colors = gen_color(cmap="Accent", n=6)

    new_color_mapping = dict(zip(range(10), colors))

    for i in id:
        combined_Z = np.concatenate([combined_Z, Z[i]], axis=0)
        combine_label = np.concatenate([combine_label, y[i]], axis=0)
        combine_dataID = np.concatenate(
            [combine_dataID, np.ones_like(y[i]) * i], axis=0
        )
        train_ID = np.concatenate([train_ID, ["train"] * np.size(y[i])], axis=0)
    pca = PCA(n_components=2)
    embed = pca.fit_transform(combined_Z)
    plot_pd = pd.DataFrame(
        {
            "umap-1": embed[:, 0],
            "umap-2": embed[:, 1],
            "label": combine_label,
            "dataset": combine_dataID,
        }
    )

    plot_pd["Color_label"] = plot_pd["label"].map(color_mapping)
    plot_pd["Color_dataset"] = plot_pd["dataset"].map(new_color_mapping)

    # Create figure with 1 row, 2 columns
    plt.figure(figsize=(4, 4))

    # First plot
    for label in plot_pd.label.unique():
        plt.scatter(
            plot_pd[plot_pd.label == label]["umap-1"],
            plot_pd[plot_pd.label == label]["umap-2"],
            c=color_mapping[int(label)],
            s=1,
        )

    # plt.axis('square')
    plt.xlabel("PC-1")
    plt.ylabel("PC-2")
    # plt.tight_layout()
    plt.savefig(path + "_labels.png", dpi=300)

    # Second plot
    plt.figure(figsize=(4, 4))
    for label in plot_pd.dataset.unique():
        plt.scatter(
            plot_pd[plot_pd.dataset == label]["umap-1"],
            plot_pd[plot_pd.dataset == label]["umap-2"],
            c=new_color_mapping[int(label)],
            s=1,
        )

    # plt.axis('square')
    plt.xlabel("PC-1")
    plt.ylabel("PC-2")
    # plt.tight_layout()
    plt.savefig(path + "_datasets.png", dpi=300)


def pca_scatter_hypoxia_impute(Z: dict, y: dict, Z_test, y_test, path=None):
    label_mapping = dict(
        zip(
            [
                "Normoxia",
                "Hypoxia_6h",
                "Hypoxia_24h",
                "Hypoxia_48h",
                "Hypoxia_72h",
                "Hypoxia_6d",
                "Hypoxia_10d",
            ],
            range(7),
        )
    )

    id = list(Z.keys())

    combined_Z = np.empty((0, Z[id[0]].shape[1]))
    combine_label = np.array([])
    combine_dataID = np.array([])
    train_ID = np.array([])

    for i in id:
        combined_Z = np.concatenate([combined_Z, Z[i]], axis=0)
        combine_label = np.concatenate([combine_label, y[i]], axis=0)
        combine_dataID = np.concatenate(
            [combine_dataID, np.ones_like(y[i]) * i], axis=0
        )
        train_ID = np.concatenate([train_ID, ["train"] * np.size(y[i])], axis=0)

    id = list(Z_test.keys())

    for i in id:
        combined_Z = np.concatenate([combined_Z, Z_test[i]], axis=0)
        combine_label = np.concatenate([combine_label, y_test[i]], axis=0)
        combine_dataID = np.concatenate(
            [combine_dataID, np.ones_like(y_test[i]) * i], axis=0
        )
        train_ID = np.concatenate([train_ID, ["test"] * np.size(y_test[i])], axis=0)
    pc = PCA(n_components=2)
    embed = pc.fit_transform(combined_Z)
    plot_pd = pd.DataFrame(
        {
            "PC-1": embed[:, 0],
            "PC-2": embed[:, 1],
            "label": combine_label,
            "dataset": combine_dataID,
            "train": train_ID,
        }
    )

    color_1 = gen_color(cmap="winter_r", n=6)
    color_2 = gen_color(cmap="Greys", n=3)
    colors = [color_2[1]] + color_1[:3] + ["#00a6fb", "#0466c8", "#03045e"]

    color_mapping = dict(zip(range(7), colors))

    colors = gen_color(cmap="Accent", n=6)

    new_color_mapping = dict(zip(range(10), colors))

    plot_pd["Color_label"] = plot_pd["label"].map(color_mapping)
    plot_pd["Color_dataset"] = plot_pd["dataset"].map(new_color_mapping)

    # Create figure with 1 row, 2 columns
    plt.figure(figsize=(4, 4))

    # First plot
    for label in plot_pd.label.unique():
        plt.scatter(
            plot_pd[(plot_pd.label == label) & (plot_pd.train == "train")]["PC-1"],
            plot_pd[(plot_pd.label == label) & (plot_pd.train == "train")]["PC-2"],
            c=color_mapping[int(label)],
            s=5,
        )

    plot_pd_test = plot_pd[(plot_pd.train == "test")]

    plt.scatter(plot_pd_test["PC-1"], plot_pd_test["PC-2"], c="salmon", s=5)
    plt.xlabel("PC-1")
    plt.ylabel("PC-2")
    plt.grid(False)
    if path is None:
        plt.show()
    else:
        print("save")
        plt.savefig(path + "_labels.png", dpi=300)

    # Second plot
    plt.figure(figsize=(4, 4))
    for label in plot_pd.dataset.unique():
        plt.scatter(
            plot_pd[plot_pd.dataset == label]["PC-1"],
            plot_pd[plot_pd.dataset == label]["PC-2"],
            c=new_color_mapping[int(label)],
            s=5,
        )

    plt.xlabel("PC-1")
    plt.ylabel("PC-2")
    plt.grid(False)
    if path is None:
        plt.show()
    else:
        plt.savefig(path + "_datasets.png", dpi=300)


# %%
