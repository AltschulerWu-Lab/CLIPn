'''
Implementation of comparing approaches
1. CCA: multi_CCA()
Sequential integration of two datasets into a common latent space using Canonical Correlation Analysis (CCA).

2. StabMap: multi_stabmap()
Construction of linear regression models to map the features of the second dataset to the latent space of the first dataset.

3. MLP: MLP()
Diagonal concatenation of features from multiple datasets followed by a multi-layer perceptron (MLP) to learn the latent space.
Refer to Supplementary Figure 1 for the architecture.

MIT License (c) 2024, Feng Bao @ UCSF
'''

from sklearn.cross_decomposition import CCA
from sklearn.linear_model import LinearRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


class Encoder(nn.Module):

    def __init__(self, x_dim, z_dim, class_num=10):
        """
            Initialize the Encoder module.

            Args:
            x_dim: Dimension of the input feature.
            z_dim: Dimension of the latent space.
            class_num: Number of classes for the output layer.
        """
        # x_dim: dimension of the input feature
        # z_dim: dimension of the latent
        super(Encoder, self).__init__()
        self.z_dim = z_dim
        self.x_dim = x_dim

        # Vanilla MLP
        self.net = nn.Sequential(
            nn.Linear(self.x_dim, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, self.z_dim),
        )
        self.last = nn.Linear(self.z_dim, class_num)

    def forward(self, x):
        latent = self.net(x)
        last_layer = self.last(latent)

        return latent, last_layer  # Return a factorized Normal distribution


class MLP:

    def __init__(self, input_dim, output_dim, class_num=10, gpu=None):
        """
            Initialize the MLP model.

            Args:
            input_dim: Dimension of the input feature.
            output_dim: Dimension of the latent space.
            class_num: Number of classes for the output layer.
            gpu: GPU device to use. If None, use the first available GPU if there is one, otherwise use CPU.
        """

        # Set the device based on the availability of a GPU
        if gpu is not None:
            self.device = torch.device(gpu)
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # print(f"Using device {self.device}")

        # Define the layers
        self.net = Encoder(input_dim, output_dim, class_num).to(self.device)
        self.x_dim = input_dim

    def fit(self, X, y, lr=1e-6, num_epochs=300, batch_size=512):
        # convert dict to numpy array with zero padding
        print("Running MLP ...")

        max_dim = self.x_dim
        total_samples = 0
        for i in X.keys():
            total_samples += X[i].shape[0]

        features = np.zeros((total_samples, max_dim))
        labels = np.zeros(total_samples)

        start = 0

        for i in X.keys():
            features[start: start + X[i].shape[0], : X[i].shape[1]] = X[i]
            labels[start: start + X[i].shape[0]] = y[i]

            start += X[i].shape[0]

        # Create dataset and data loader
        dataset = TensorDataset(
            torch.tensor(features, dtype=torch.float32).to(self.device),
            torch.tensor(labels, dtype=torch.long).to(self.device),
        )
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.net.parameters(), lr=lr)

        # Training loop
        for epoch in range(num_epochs):
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                embed, last = self.net(X_batch)
                output = F.softmax(last, dim=1)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()
            # if (epoch + 1) % 100 == 0:
            #     print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")

    def embedding(self, X: dict):
        # Convert new data to tensor

        embeddings = dict()

        for i in X.keys():
            X_expand = np.zeros((X[i].shape[0], self.x_dim))
            X_expand[:, : X[i].shape[1]] = X[i]

            data_on_device = torch.Tensor(X_expand).to(self.device)
            embedding, _ = self.net(data_on_device)
            if self.device != "cpu":
                embeddings[i] = embedding.cpu().detach().numpy()
            else:
                embeddings[i] = embedding.detach().numpy()

        return embeddings

    def predict(self, X: dict):
        """
            Generate predictions for the input data using the trained MLP model.

            Args:
            X: Dictionary of input data.

            Returns:
            labels: Dictionary of predicted labels corresponding to the input data.
        """
        # Convert new data to tensor
        labels = dict()

        for i in X.keys():
            X_expand = np.zeros((X[i].shape[0], self.x_dim))
            X_expand[:, : X[i].shape[1]] = X[i]

            data_on_device = torch.Tensor(X_expand).to(self.device)
            embedding, _ = self.net(data_on_device)
            label = F.softmax(embedding, dim=1).argmax(dim=1)
            if self.device != "cpu":
                labels[i] = label.cpu().detach().numpy()
            else:
                labels[i] = label.detach().numpy()

        return labels


class multi_CCA():
    def __init__(self, X: dict, y: dict, latent_dim=10):
        """
            Initialize the multi_CCA model.

            Args:
            X: Dictionary of input data.
            y: Dictionary of labels corresponding to the input data.
            latent_dim: Dimension of the latent space.
        """

        print("Running CCA ...")

        self.n_datasets = len(X.keys())
        self.latent_dim = latent_dim

        self.ccas = dict()
        for i in range(1, self.n_datasets):
            self.ccas[i] = CCA(n_components=self.latent_dim)

        self.feature_dim = dict()
        for i in range(self.n_datasets):
            self.feature_dim[i] = X[i].shape[1]

    def fit_transform(self, X: dict, y: dict):
        X_pivot = X[0]
        y_pivot = y[0]

        data_id = [0] * y_pivot.size

        for i in range(1, self.n_datasets):
            X_aux = X[i]
            y_aux = y[i]
            mask = np.isin(y_aux, y_pivot)
            y_aux_filter = y_aux[mask]

            mat_a = np.empty([0, X_pivot.shape[1]])
            mat_b = np.empty([0, X_aux.shape[1]])
            for lb in np.unique(y_aux_filter):
                mat_b_temp = X_aux[y_aux == lb]
                mat_b = np.vstack([mat_b, mat_b_temp])

                pivot_data_matched = X_pivot[y_pivot == lb]
                rand_select = np.random.choice(
                    range(pivot_data_matched.shape[0]), size=mat_b_temp.shape[0])

                mat_a_temp = pivot_data_matched[rand_select, :]
                mat_a = np.vstack([mat_a, mat_a_temp])

            self.ccas[i].fit(mat_a, mat_b)

            X_pivot_cca, X_aux_cca = self.ccas[i].transform(X_pivot, X_aux)

            X_pivot = np.vstack([X_pivot_cca, X_aux_cca])
            y_pivot = np.concatenate([y_pivot, y_aux])

            data_id += [i] * y_aux.size

        data_id = np.array(data_id)
        Z = dict()
        for i in range(self.n_datasets):
            Z[i] = X_pivot[data_id == i]
        return Z

    def predict(self, X):

        Z = dict()
        dataset_id = X.keys()
        for ii in dataset_id:
            if ii == 0:
                Z_sub = X[ii]
                for i in range(1, self.n_datasets):
                    Z_sub = self.ccas[i].transform(
                        Z_sub, np.ones([1, self.feature_dim[i]]))[0]
            else:
                Z_sub = X[ii]
                for i in range(ii, self.n_datasets):
                    if i == ii:
                        if i == 1:
                            Z_sub = self.ccas[i].transform(
                                np.ones([1, self.feature_dim[i - 1]]), Z_sub)[1]
                        else:
                            Z_sub = self.ccas[i].transform(
                                np.ones([1, self.latent_dim]), Z_sub)[1]
                    else:
                        Z_sub = self.ccas[i].transform(
                            Z_sub, np.ones([1, self.feature_dim[i]]))[0]
            Z[ii] = Z_sub
        return Z


class multi_stabmap():
    def __init__(self, X: dict, y: dict, latent_dim=10):
        """
            Initialize the multi_stabmap model.

            Args:
            X: Dictionary of input data.
            y: Dictionary of labels corresponding to the input data.
            latent_dim: Dimension of the latent space.
        """
        print("Running Stabmap ...")

        self.n_datasets = len(X.keys())
        self.latent_dim = min(
            [X[0].shape[1], latent_dim, np.unique(y[0]).size - 1])
        self.lda = LinearDiscriminantAnalysis(n_components=self.latent_dim)
        self.lrs = dict()
        for i in range(1, self.n_datasets):
            self.lrs[i] = LinearRegression()

    def fit_transform(self, X: dict, y: dict):

        X_pivot = X[0]
        y_pivot = y[0]
        # LDA analysis

        self.lda.fit(X_pivot, y_pivot)

        X_pivot = self.lda.transform(X_pivot)

        data_id = [0] * y_pivot.size

        for i in range(1, self.n_datasets):
            X_aux = X[i]
            y_aux = y[i]
            mask = np.isin(y_aux, y_pivot)
            y_aux_filter = y_aux[mask]

            mat_a = np.empty([0, X_pivot.shape[1]])
            mat_b = np.empty([0, X_aux.shape[1]])
            for lb in np.unique(y_aux_filter):
                mat_b_temp = X_aux[y_aux == lb]
                mat_b = np.vstack([mat_b, mat_b_temp])

                pivot_data_matched = X_pivot[y_pivot == lb]
                rand_select = np.random.choice(
                    range(pivot_data_matched.shape[0]), size=mat_b_temp.shape[0])

                mat_a_temp = pivot_data_matched[rand_select, :]
                mat_a = np.vstack([mat_a, mat_a_temp])

            self.lrs[i].fit(mat_b, mat_a)

            X_aux_transform = self.lrs[i].predict(X_aux)

            X_pivot = np.vstack([X_pivot, X_aux_transform])
            y_pivot = np.concatenate([y_pivot, y_aux])

            data_id += [i] * y_aux.size

        data_id = np.array(data_id)
        Z = dict()
        for i in range(self.n_datasets):
            Z[i] = X_pivot[data_id == i]
        return Z

    def predict(self, X: dict):
        Z = dict()
        dataset_id = X.keys()
        for ii in dataset_id:
            if ii == 0:
                Z_sub = self.lda.transform(X[ii])
            else:
                Z_sub = self.lrs[ii].predict(X[ii])
            Z[ii] = Z_sub
        return Z
