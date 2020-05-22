from torchvision.datasets import CIFAR10
from torch.utils.data.sampler import SubsetRandomSampler
import torch
import numpy as np


def get_train_val_loaders(data_dir,
                         train_transforms,
                         val_transforms,
                         batch_size,
                         random_seed=42,
                         val_size=0.9,
                         shuffle=True,
                         num_workers=1,
                         pin_memory=True):
    """
    Function for loading and returning train and val loader of
    CIFAR-10 dataset.
    Params
    ------
    - data_dir [string] : path directory to the dataset.
    - transforms [transforms.Compose]: image transforms
    - batch_size [int]: amount of samples per iteration during training.
    - random_seed [int]: fix seed for reproducibility.
    - val_size [float]: percentage split of the training set used for
      the validation set. Should be in the range [0, 1].
    - shuffle [bool]: whether to shuffle the train/validation indices.
    - num_workers [int]: number of subprocesses to use when loading the dataset.
    - pin_memory [bool]: whether to copy tensors into CUDA pinned memory.
      Set it to True if using GPU.
    Returns
    -------
    - train_loader [torch.utils.data.DataLoader]: training set iterator.
    - valid_loader [torch.utils.data.DataLoader]: validation set iterator.
    """

    train_dataset = CIFAR10(
        root=data_dir, train=True,
        download=True, transform=train_transforms,
    )

    val_dataset = CIFAR10(
        root=data_dir, train=True,
        download=True, transform=val_transforms,
    )

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(val_size * num_train)

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, val_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, sampler=val_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return (train_loader, val_loader)


def get_test_loader(data_dir,
                    transforms,
                    batch_size,
                    shuffle=True,
                    num_workers=4,
                    pin_memory=False):
    """
    Function for loading and returning test loader of CIFAR-10 dataset.
    Params
    ------
    - data_dir [string] : path directory to the dataset.
    - transforms [transforms.Compose]: image transforms
    - batch_size [int]: amount of samples per iteration during training.
    - shuffle [bool]: whether to shuffle the train/validation indices.
    - num_workers [int]: number of subprocesses to use when loading the dataset.
    - pin_memory [bool]: whether to copy tensors into CUDA pinned memory.
      Set it to True if using GPU.
    Returns
    -------
    - test_loader [torch.utils.data.DataLoader]: testing set iterator.
    """

    dataset = CIFAR10(
        root=data_dir, train=False,
        download=True, transform=transforms,
    )

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return data_loader
