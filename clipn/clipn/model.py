'''
CLIPn is an approach developed for the simultaneous integration of multiple phenotypic screen datasets, building upon
contrastive learning.

Feng Bao @ UCSF, 2024
'''

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler


class CLIPn:
    def __init__(self, X: dict, y: dict, latent_dim=10, gpu=None):
        """
        Initialize the CLIPn model. The input requirement for data is to construct networks with respect to the shape
            of each dataset.

        X: A dictionary of datasets. Keys of the dictionary are numerical labels starting from 0.
        y: A dictionary of labels corresponding to the datasets.
        latent_dim: The dimension of the latent space.
        gpu: The GPU device to use. If None, it will use the first available GPU if there is one ("cuda:0"),
            otherwise it will use CPU.
        """

        print("Running CLIPn ...")

        self.n_datasets = len(X)
        self.latent_dim = latent_dim

        if gpu is None:
            use_cuda = torch.cuda.is_available()
            self.device = torch.device("cuda:0" if use_cuda else "cpu")
        else:
            self.device = torch.device(gpu)

        # Collect dimension information for all datasets.
        self.feature_dim = []
        for i in range(self.n_datasets):
            self.feature_dim.append(X[i].shape[1])
            self.model = CellCLIP(self.feature_dim, self.latent_dim, self.device)

        self.model.to(self.device)

    def fit(self, X, y, epochs=300, lr=1e-6, batch_size=256):
        """
        Fit the model to the data.

        X: A dictionary of datasets (the same as initialized X).
        y: A dictionary of labels corresponding to the datasets.
        epochs: The number of epochs to train for. Each epoch trains optimize contrastive loss for all datasets
            simultaneously.
        lr: The learning rate for the optimizer. It was determined by grid search on simulated datasets.
        batch_size: The batch size for the DataLoader.
        """

        # Construct data loader.
        dataloaders = dict()
        sample_size = []

        for i in range(self.n_datasets):
            feature, label = X[i], y[i]
            sample_size.append(len(y[i]))

            # Construct weighted sampler.
            label_counts = torch.bincount(torch.tensor(label))
            class_weights = 1.0 / label_counts.float()
            weights = [class_weights[lab] for lab in label]
            sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

            feature = torch.Tensor(feature)
            label = torch.Tensor(label)

            dataset = TensorDataset(feature.to(self.device), label.to(self.device))
            train_dl = DataLoader(dataset, batch_size, sampler=sampler)

            dataloaders[i] = train_dl

        batch_epoch = round((max(sample_size) + 0.0) / batch_size) + 1

        losses = self.model.fit(dataloaders, epochs=epochs, lr=lr, batch_epoch=batch_epoch * 2)

        return losses

    def predict(self, X: dict):
        """
        Inference the embeddings for the given data using trained model.
        X: A dictionary of datasets with their raw features.
        """

        data_on_device = dict()
        for i in X.keys():
            data_on_device[i] = torch.Tensor(X[i]).to(self.device)

        embed = self.model.inference(data_on_device)

        embed_off_device = dict()
        for i in X.keys():
            embed_off_device[i] = embed[i].detach().cpu().numpy()

        return embed_off_device


class CellCLIP(nn.Module):
    def __init__(self, feature_dim, z_dim, device):
        """
        Initialize the CellCLIP model.

        feature_dim: The dimension vector of the features from all datasets.
        z_dim: The dimension of the latent space.
        device: The device to use for computations.
        """

        super(CellCLIP, self).__init__()

        self.feature_dim = feature_dim
        self.z_dim = z_dim
        self.device = device
        self.dataset_num = len(feature_dim)

        # Build encoders based on number of datasets
        self.encoders = dict()

        for i in range(self.dataset_num):
            self.encoders[i] = Encoder(self.feature_dim[i], z_dim=self.z_dim).to(self.device)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).to(self.device)

    def embedding(self, datasets: dict):
        """
        Compute the embeddings for the given dataset dictionary.

        datasets: A dictionary of datasets.
        """

        embeddings = dict()
        for i in datasets.keys():
            z = self.encoders[i](datasets[i])
            # l2 normalized features
            embeddings[i] = z / z.norm(dim=1, keepdim=True)

        return embeddings

    def similarity(self, embedding_1, embedding_2):
        """
        Compute the similarity between two embeddings.

        embedding_1: The embeddings from first dataset.
        embedding_2: The embedding from second dataset.
        """

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * embedding_1 @ embedding_2.t()

        return logits

    def label_matching(self, label_1, label_2):
        """
        Construct label matching matrix between two datasets.

        label_1: The label vector from first dataset.
        label_2: The label vector from second dataset.
        """
        label_consist = (label_1.unsqueeze(1) == label_2.unsqueeze(0)).float()

        return label_consist

    def soft_label_CE_loss(self, logits, soft_labels):

        """
        Compute the cross-entropy loss between the logits and the soft labels.

        logits: The logits output by the model.
        soft_labels: The soft labels.
        """

        p = F.log_softmax(logits, dim=1)
        loss = -p * soft_labels

        return torch.mean(loss)

    def fit(self, train_dls, epochs=200, lr=1e-6, weight_decay=0, batch_epoch=20):
        """
        Fit the model to the data.

        train_dls: A dictionary of DataLoaders for the training data.
        epochs: The number of epochs to train for.
        lr: The learning rate for the optimizer.
        weight_decay: The weight decay for the optimizer.
        batch_epoch: The number of batches per epoch.
        """

        torch.cuda.empty_cache()
        losses = []

        # Define optimizers
        opt_params = []
        for i in range(self.dataset_num):
            opt_params += list(self.encoders[i].parameters())

        opt = torch.optim.Adam(opt_params, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999), eps=1e-08,
                               amsgrad=True)

        # Epoch loop over datasets
        for epoch in tqdm(range(epochs)):
            x = dict()
            y = dict()

            for i in range(self.dataset_num):
                x[i], y[i] = next(iter(train_dls[i]))
            # Epoch loop over all samples
            for epoch_inner in range(batch_epoch):
                loss = self.train_loss(x, y, opt)

                losses.append(loss)

        return losses

    def train_loss(self, x: dict, y: dict, opt):
        """
        Compute the training loss.

        x: A dictionary of datasets.
        y: A dictionary of labels corresponding to the datasets.
        opt: The optimizer.
        """

        embeddings = self.embedding(datasets=x)

        total_loss = 0.0
        for select_index in range(self.dataset_num):
            x_A = embeddings[select_index]
            y_A = y[select_index]

            x_B = torch.cat([embeddings[i] for i in range(self.dataset_num) if i != select_index], dim=0)
            y_B = torch.cat([y[i] for i in range(self.dataset_num) if i != select_index], dim=0)

            logits = self.similarity(x_A, x_B)
            ground_truth = self.label_matching(y_A, y_B)

            loss = self.soft_label_CE_loss(logits, ground_truth)
            loss_t = self.soft_label_CE_loss(logits.T, ground_truth.T)

            total_loss += (loss + loss_t) / 2

        opt.zero_grad()
        total_loss.backward()
        opt.step()

        return total_loss.item()

    def inference(self, datasets: dict):
        """
        Compute the embeddings for the given datasets.

        datasets: A dictionary of datasets.
        """

        embed = dict()
        for i in datasets.keys():
            z = self.encoders[i](datasets[i])
            embed[i] = z / z.norm(dim=1, keepdim=True)

        return embed


class Encoder(nn.Module):
    def __init__(self, x_dim, z_dim):
        super(Encoder, self).__init__()
        self.z_dim = z_dim
        self.x_dim = x_dim

        # Vanilla MLP
        self.net = nn.Sequential(

            nn.Linear(self.x_dim, 1024),

            nn.Linear(1024, 512),

            nn.Linear(512, 128),

            nn.Linear(128, self.z_dim),
        )

    def forward(self, x):
        latent = self.net(x)
        return latent
