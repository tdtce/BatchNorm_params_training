import torch
from torch.nn.init import kaiming_normal_
from torch import nn


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

def custom_decrease(it):
    if it < 80:
        return 1
    if it < 120:
        return 0.1
    if it >= 120:
        return 0.01

def is_valid_cifar_name(name):
    params = name.split('_')
    is_len_correct = 5 > len(params) > 2
    is_params_positive = all([int(x) > 0 for x in params[2:]])
    is_params_digit = all([x.isdigit() for x in params[2:]])
    is_div_6 = (int(params[2]) - 2) % 6 == 0

    is_valid = is_len_correct and \
               is_params_positive and \
               is_params_digit and \
               is_div_6
    return is_valid

def get_plan(name):
    params = name.split('_')
    W = 16 if len(params) == 3 else int(params[3])
    D = int(params[2])
    D = (D - 2) // 6
    plan = [(W, D), (2*W, D), (4*W, D)]
    return plan

def weight_init(m):
    if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight)
    if isinstance(m, torch.nn.BatchNorm2d):
        m.weight.data = torch.rand(m.weight.data.shape)
        m.bias.data = torch.zeros_like(m.bias.data)

def save_predictions (predicitons, name):
    answers = list(map(lambda x: x.argmax().item(), test_predictions))

    dir_name = "predictions"
    if not os.path.exists(dir_name):
        os.mkdir(dirName)
    with open(os.path.join(dir_name, name + "pred.csv")) as f:
        f.write(", ".join(map(str, answers)))


                # The naming scheme for a ResNet is 'cifar_resnet_N[_W]'.
                # The ResNet is structured as an initial convolutional layer followed by three "segments"
                # and a linear output layer. Each segment consists of D blocks. Each block is two
                # convolutional layers surrounded by a residual connection. Each layer in the first segment
                # has W filters, each layer in the second segment has 32W filters, and each layer in the
                # third segment has 64W filters.
                # The name of a ResNet is 'cifar_resnet_N[_W]', where W is as described above.
                # N is the total number of layers in the network: 2 + 6D.
                # The default value of W is 16 if it isn't provided.
                # For example, ResNet-20 has 20 layers. Exclusing the first convolutional layer and the final
                # linear layer, there are 18 convolutional layers in the blocks. That means there are nine
                # blocks, meaning there are three blocks per segment. Hence, D = 3.
                # The name of the network would be 'cifar_resnet_20' or 'cifar_resnet_20_16'.
