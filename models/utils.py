import torch


def is_valid_cifar_name(name):
    """
    Function check CIFAR ResNet name because name has information about
    network structure.
    Params
    ------
    - name [string] : CIFAR ResNet name (template: cifar_resnet_N[_W]).
    Returns
    -------
    - is_valid [bool] : is name valid or not.
    """
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
    """
    Function scan CIFAR ResNet name because name has information about
    network structure. Name template - cifar_resnet_N[_W]. Each network building
    by that plan:
    Conv -> [Segment 1] -> [Segment 2] -> [Segment 3] -> Linear
    Each segment consists of D blocks. Each block is 2 convolutional layers
    surrounded by a residual connection. Each layer in the first segment
    has W filters, each layer in the second segment has 32W filters, and
    each layer in the third segment has 64W filters.
    N - total amount of layers. N = 1 conv + 1 Linear + 3 * D * 2. By default
    W = 16.
    Example. cifar_resnet_20
    => N = 20, W = 16
    N = 1 + 2*size(Segment 1) + 2*size(Segment 2) + 2*size(Segment 3) + 1
    N - 2 = 6 * size(Segment)
    N - 2 = 6 * D
    D = (N - 2) // 6
    D = 3
    Params
    ------
    - name [string] : CIFAR ResNet name (example: cifar_resnet_14).
    Returns
    -------
    - plan [list] : network structure.
    """
    params = name.split('_')
    W = 16 if len(params) == 3 else int(params[3]) * 16
    D = int(params[2])
    D = (D - 2) // 6
    plan = [(W, D), (2 * W, D), (4 * W, D)]
    return plan


def weight_init(m):
    """
    Initializer for ResNet. Convolutional and Linear layers get He
    initialization. BatchNorm2d params has only 1 dimension and that type of
    initialization can't be applied. For BatchNorm using rand init for weight
    and zero init for bias
    Params
    ------
    - m [Layer] : torch network layer.
    """
    if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight)
    if isinstance(m, torch.nn.BatchNorm2d):
        m.weight.data = torch.rand(m.weight.data.shape)
        m.bias.data = torch.zeros_like(m.bias.data)
