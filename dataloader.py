import os
import random
from math import ceil
import pickle
import json
from typing import List
import tqdm
import numpy as np
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms


def data_loader(dataset_name, dataroot, batch_size, val_ratio, total_clients, world_size, rank, group, heterogeneity=0, extra_bs=None, num_workers=1, small=False, full_batch=False):
    """
    Args:
        dataset_name (str): the name of the dataset to use, currently only
            supports 'MNIST', 'FashionMNIST', 'CIFAR10' and 'CIFAR100'.
        dataroot (str): the location to save the dataset.
        batch_size (int): batch size used in training.
        val_ratio (float): the percentage of trainng data used as validation, if there
            is no separate validation set.
        total_clients (int): total number of clients participating in training.
        world_size (int): how many processes will be used in training.
        rank (int): the rank of this process.
        heterogeneity (float): dissimilarity between data distribution across clients.
            Between 0 and 1.
        extra_bs (int): Batch size for extra data loader.
        small (bool): Whether to use miniature dataset.
        full_batch (bool): have a dataset with a singular batch

    Outputs:
        iterators over training, validation, and test data.
    """
    if ((val_ratio < 0) or (val_ratio > 1.0)):
        raise ValueError("[!] val_ratio should be in the range [0, 1].")
    if heterogeneity < 0:
        raise ValueError("Data heterogeneity must be positive.")
    if total_clients == 1 and heterogeneity > 0:
        raise ValueError("Cannot create a heterogeneous dataset when total_clients == 1.")

    # Dataset-specific processing.
    if dataset_name == 'CIFAR10':
        dataset = torchvision.datasets.CIFAR10
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                         (0.2470, 0.2435, 0.2616))
        transform_train = transforms.Compose([transforms.RandomHorizontalFlip(),
                                              transforms.RandomCrop(32, padding=4),
                                              transforms.ToTensor(),
                                              normalize])
        transform_test = transforms.Compose([transforms.ToTensor(),
                                             normalize])
        train_kwargs = {"train": True, "download": True, "transform": transform_train}
        val_kwargs = {"train": True, "download": True, "transform": transform_test}
        test_kwargs = {"train": False, "download": True, "transform": transform_test}
        num_labels = 10
        separate_val = False
        predefined_clients = False

    elif dataset_name == 'CIFAR100':
        dataset = torchvision.datasets.CIFAR100
        normalize = transforms.Normalize((0.5071, 0.4866, 0.4409),
                                         (0.2673, 0.2564, 0.2762))
        transform_train = transforms.Compose([transforms.RandomHorizontalFlip(),
                                              transforms.RandomCrop(32, padding=4),
                                              transforms.ToTensor(),
                                              normalize])
        transform_test = transforms.Compose([transforms.ToTensor(),
                                             normalize])
        train_kwargs = {"train": True, "download": True, "transform": transform_train}
        val_kwargs = {"train": True, "download": True, "transform": transform_test}
        test_kwargs = {"train": False, "download": True, "transform": transform_test}
        num_labels = 100
        separate_val = False
        predefined_clients = False

    elif dataset_name == 'MNIST':
        dataset = torchvision.datasets.MNIST
        normalize = transforms.Normalize((0.1307,), (0.3081,))
        transform_train = transforms.Compose([transforms.ToTensor(),
                                              normalize])
        transform_test = transform_train
        train_kwargs = {"train": True, "download": True, "transform": transform_train}
        val_kwargs = {"train": True, "download": True, "transform": transform_test}
        test_kwargs = {"train": False, "download": True, "transform": transform_test}
        num_labels = 10
        separate_val = False
        predefined_clients = False

    elif dataset_name == 'FashionMNIST':
        dataset = torchvision.datasets.FashionMNIST
        normalize = transforms.Normalize((0.2860,), (0.3530,))
        transform_train = transforms.Compose([transforms.ToTensor(),
                                              normalize])
        transform_test = transform_train
        train_kwargs = {"train": True, "download": True, "transform": transform_train}
        val_kwargs = {"train": True, "download": True, "transform": transform_test}
        test_kwargs = {"train": False, "download": True, "transform": transform_test}
        num_labels = 10
        separate_val = False
        predefined_clients = False

    

    else:
        raise NotImplementedError

    #
    train_set = dataset(root=dataroot, **train_kwargs)
    val_set = dataset(root=dataroot, **val_kwargs)
    test_set = dataset(root=dataroot, **test_kwargs)



    # Handle the case of predefined clients (for datasets designed for federated
    # learning) versus non-predefined clients (for centralized datasets which we split
    # into clients for federated learning).
    if predefined_clients:
        get_idxs = lambda dset: list(range(dset.num_clients))
    else:
        get_idxs = lambda dset: list(range(len(dset)))

    # Split training set into training and validation.
    if separate_val:
        train_idxs = get_idxs(train_set)
        val_idxs = get_idxs(val_set)
        random.shuffle(val_idxs)
    else:
        total_train_idxs = get_idxs(train_set)
        random.shuffle(total_train_idxs)
        val_split = round(val_ratio * len(total_train_idxs))
        train_idxs = total_train_idxs[val_split:]
        val_idxs = total_train_idxs[:val_split]
    test_idxs = get_idxs(test_set)
    random.shuffle(test_idxs)

    # Partition the training data into multiple clients if needed. Data partitioning to
    # create heterogeneity is performed according to the specifications in
    # https://arxiv.org/abs/1910.06378.
    if total_clients > 1:
        random.seed(1234)

    # Split data into iid pool and non-iid pool. If we don't have predefined clients,
    # ensure that label distribution of iid pool and non-iid pool matches that of
    # overall dataset.
    if predefined_clients:

        # Client partitioning only works for binary classification, since we sort
        # clients by proportion of positive samples.
        if train_set.n_classes != 2:
            raise NotImplementedError

        # Divide clients into iid pool and non-iid pool.
        random.shuffle(train_idxs)
        iid_split = round((1.0 - heterogeneity) * len(train_idxs))
        iid_pool = train_idxs[:iid_split]
        non_iid_pool = train_idxs[iid_split:]

        # Sort non-iid pool by proportion of positive samples.
        positive_prop = np.zeros(len(non_iid_pool))
        for i, client in enumerate(non_iid_pool):
            client_labels = [train_set.labels[idx] for idx in train_set.user_items[client]]
            positive_prop[i] = np.mean(client_labels)
        pool_order = np.argsort(positive_prop)
        non_iid_pool = [non_iid_pool[idx] for idx in pool_order]

    else:

        # Collect indices of instances with each label.
        train_label_idxs = get_label_indices(dataset_name, train_set, num_labels)
        for l in range(num_labels):
            if not separate_val:
                train_label_idxs[l] = [i for i in train_label_idxs[l] if i in train_idxs]
            random.shuffle(train_label_idxs[l])

        # Divide samples from each label into iid pool and non-iid pool. Note that samples
        # in iid pool are shuffled while samples in non-iid pool are sorted by label.
        iid_pool = []
        non_iid_pool = []
        for i in range(num_labels):
            iid_split = round((1.0 - heterogeneity) * len(train_label_idxs[i]))
            iid_pool += train_label_idxs[i][:iid_split]
            non_iid_pool += train_label_idxs[i][iid_split:]
        random.shuffle(iid_pool)

    # Allocate iid and non-iid samples to each client.
    client_train_idxs = [[] for _ in range(total_clients)]
    num_iid = len(iid_pool) // total_clients
    num_non_iid = len(non_iid_pool) // total_clients
    partition_size = num_iid + num_non_iid
    for j in range(total_clients):
        client_train_idxs[j] += iid_pool[num_iid * j: num_iid * (j+1)]
        client_train_idxs[j] += non_iid_pool[num_non_iid * j: num_non_iid * (j+1)]
        random.shuffle(client_train_idxs[j])

    # Get indices of local validation and test dataset. Note that the validation and
    # test set are not split into `total_clients` partitions, just `world_size`
    # partitions, since evaluation does not need to happen separate for each client.
    # TODO: Do we really need to enforce that each partition of the test set has the
    # same size? We are just throwing away some test examples here and it may be
    # unnecessary.
    val_partition = len(val_idxs) // world_size
    test_partition = len(test_idxs) // world_size
    local_val_idxs = val_idxs[rank * val_partition: (rank+1) * val_partition]
    local_test_idxs = test_idxs[rank * test_partition: (rank+1) * test_partition]

    # Use miniature dataset, if necessary.
    if small:
        for r in range(total_clients):
            client_train_idxs[r] = client_train_idxs[r][:round(len(client_train_idxs[r]) / 100)]
        local_val_idxs = local_val_idxs[:round(len(local_val_idxs) / 100)]
        local_test_idxs = local_test_idxs[:round(len(local_test_idxs) / 100)]

    # If partitioning clients, convert list of clients into list of dataset elements.
    if predefined_clients:

        for r in range(total_clients):
            current_idxs = list(client_train_idxs[r])
            client_train_idxs[r] = []
            for i in current_idxs:
                client_train_idxs[r] += train_set.user_items[i]

        current_idxs = list(local_val_idxs)
        local_val_idxs = []
        for i in current_idxs:
            local_val_idxs += val_set.user_items[i]

        current_idxs = list(local_test_idxs)
        local_test_idxs = []
        for i in current_idxs:
            local_test_idxs += test_set.user_items[i]

    # Construct loaders for train, val, test, and extra sets. To support client
    # subsampling, we use a MultiClientLoader for the training loader, which allows each
    # worker process to switch between client datasets at the beginning of each round.
    loader_kwargs = {"batch_size": batch_size}
    

    # Dealing with situations when full-batch GD is used. In these cases, only one batch
    # per each client, so each portion of client_train_idxs is clipped off from idx batch_size
    # onward. 
    if full_batch:
        client_train_idxs = [client_set[:batch_size] for client_set in client_train_idxs]

    train_loaders = [DataLoader(
                train_set,
                batch_size=batch_size,
                sampler=SubsetRandomSampler(client_train_idxs[rank]),
            ) for rank in range(total_clients)]
    val_loader = DataLoader(
        val_set, sampler=SubsetRandomSampler(local_val_idxs), **loader_kwargs
    )
    test_loader = DataLoader(
        test_set, sampler=SubsetRandomSampler(local_test_idxs), **loader_kwargs
    )

    
    
    return train_loaders, val_loader, test_loader



def get_label_indices(dataset_name, dset, num_labels):
    """
    Returns a dictionary mapping each label to a list of the indices of elements in
    `dset` with the corresponding label.
    """

    if dataset_name in ["CIFAR10", "CIFAR100", "MNIST", "FashionMNIST"]:
        label_indices = [[] for _ in range(num_labels)]
        for idx, label in enumerate(dset.targets):
            label_indices[label].append(idx)
    elif dataset_name == "SNLI":
        label_indices = [
            (dset.targets == i).nonzero()[0].tolist()
            for i in range(dset.n_classes)
        ]
    else:
        raise NotImplementedError

    return label_indices


def get_num_classes(dataset_name):
    if dataset_name == "CIFAR10":
        classes = 10
    else:
        raise NotImplementedError
    return classes