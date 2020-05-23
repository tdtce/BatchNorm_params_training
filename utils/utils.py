import torch
import os
from models.builder import build_model


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


def save_predictions(predicitons, name):
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


def count_params(model):
    """
    Function count params of model.
    Params
    ------
    - model [nn.Module] : torch network.
    Returns
    -------
    - params_count [tuple] : total params, batch norm params,
      trainable params, linear params, skip params
    """
    total_params = 0
    batch_norm_params = 0
    trainable_params = 0
    linear_params = 0
    skip_params = 0

    for name, params in model.named_parameters():
        num_params = params.numel()

        total_params += num_params
        if "bn" in name or "skip.1" in name:
            batch_norm_params += num_params
        if params.requires_grad:
            trainable_params += num_params
        if "fc" in name:
            linear_params += num_params
        if "skip" in name:
            skip_params += num_params

    return batch_norm_params, skip_params, linear_params, total_params, trainable_params


def print_table(models):
    """
    Function print table of params amount in models.
    Params
    ------
    - models [list] : list with model names.
    """
    col_names = ["BatchNorm", "Shortcut", "Output", "Total",  "Trainable"]

    data = [count_params(build_model(m, "cpu")) for m in models]
    row_format ="{:>18}" * (len(col_names) + 1)
    print(row_format.format("", *col_names))
    for model, row in zip(models, data):
        print(row_format.format(model, *row))
