import torch
from torch import nn
import argparser
import copy
import dataloader

from train import train_models
args = argparser.arg_parser()

if args.model == "CNN":
    if args.dataset == "CIFAR10":
        if args.layers == 2:
            net = nn.Sequential(
                    nn.Conv2d(3, 32, bias=True, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(32, 32, bias=True, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Flatten(),
                    nn.Linear(2048, 10, bias=True)
                )
        else:
            from models import CIFAR_CNN
            net = CIFAR_CNN(num_layers=args.layers, num_classes=10)
    elif args.dataset == "MNIST" or args.dataset == "FashionMNIST":
        from models import SimpleCNN

        in_size = (28, 28, 1)
        out_size = 10
        net = SimpleCNN(in_size=in_size, out_size=out_size)

elif args.model == "MLP":
    start_size = 32*32*3 if args.dataset == "CIFAR10" else 28*28*1
    layers = []
    layers.append(nn.Flatten())
    hidden_size = args.width
    for i in range(args.layers - 1):
        layers.append(nn.Linear(start_size if i == 0 else hidden_size, hidden_size))
        layers.append(nn.Tanh())
    layers.append(nn.Linear(hidden_size, 10))
    net = nn.Sequential(*layers)

else:
    raise ValueError(f"Model {args.model} not recognized.")

num_copies = int(args.world_size)
model_copies = [copy.deepcopy(net) for _ in range(num_copies)]

datasets = dataloader.data_loader(
    dataset_name=args.dataset,
    dataroot="data",
    batch_size=args.batchsize,
    val_ratio=0.1,
    total_clients=args.world_size,
    world_size=args.world_size,
    rank=0,
    group=None,
    heterogeneity=args.heterogeneity,
    full_batch=args.full_batch
)

train_loaders, val_loader, test_loader = datasets
print(f"Initialized {args.model} model with {args.layers} layers for {args.dataset} dataset.")

train_models(
    args,
    model_copies,
    train_loaders,
    val_loader,
    test_loader
)













