import torch
import os
import numpy as np


def accuracy(predicitons, labels):
    """
    Calculate accuracy with tensors. Return accuracy in percentage.
    Params
    ------
    - predicitons [Tensor] : contains score for each class
    - labels [Tensor] : contains label of ground truth class
    Returns
    -------
    - acc [float] : percentage accuracy
    """
    # Get label of predicted class
    _, pred_labels = torch.max(predicitons.data, 1)
    total = labels.size(0)
    correct = (pred_labels == labels).sum().item()

    acc = 100 * correct / total
    return acc


def custom_decrease(epoch):
    """
    Function describes the LR multiplier depending on epoch. Step function
    as in the paper.
    Params
    ------
    - epoch [int] : epoch number.
    Returns
    -------
    - multiplier [int] : multiplier for learning rate.
    """
    if epoch < 80:
        return 1
    if epoch < 120:
        return 0.1
    if epoch >= 120:
        return 0.01


def save_predictions(predicitons, metric_values, name):
    """
    Function save predictions as predicted labels in csv format.
    Params
    ------
    - predicitons [Tensor] : Tensor with predictions.
    - name [str] : network name. Output file -> {name}_pred.csv
    """
    answers = list(map(lambda x: x.argmax().item(), predicitons))

    dir_name = "predictions"
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    with open(os.path.join(dir_name, name + "_pred.csv"), "w") as f:
        f.write(", ".join(map(str, answers)))
    with open(os.path.join(dir_name, name + "_metric.txt"), "w") as f:
        f.write(str(np.mean(metric_values)))
