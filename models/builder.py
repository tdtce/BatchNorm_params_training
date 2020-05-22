import torchvision.models as models
import torch.nn as nn
from utils.utils import is_valid_cifar_name, get_plan, weight_init
from models.resnet_cifar import ResNetCifar


def build_model(name, device, output=10):
    """
    Function build model by name.
    Params
    ------
    - name [string] : name of architecture.
    - device [string] : device for calculating cpu or cuda.
    - output [int] : amount of network output.
    Returns
    -------
    - model [nn.Module] : torch network model.
    """
    if name == "ResNet18":
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, output, bias=True)
        model.to(device)
        return model

    if name.startswith('cifar_resnet_'):
        if not is_valid_cifar_name(name):
            raise ValueError(f'Invalid model name: {name}')

        plan = get_plan(name)
        model = ResNetCifar(plan, weight_init, output)
        model.to(device)
        return model

    if name.startswith('cifar_freeze_'):
        if not is_valid_cifar_name(name):
            raise ValueError(f'Invalid model name: {name}')

        plan = get_plan(name)
        model = ResNetCifar(plan, weight_init, output)
        # Freeze all params without BN bias
        for k, v in model.named_parameters():
            # Don't feeze BN bias layer
            if "bn" in k and "bias" in k:
                continue
            # Don't freeze BN bias on skip connections
            if "skip" in k and "bias" in k:
                continue
            # Otherwise disable grad
            v.requires_grad = False

        model.to(device)
        return model
    raise ValueError("Invalid model!")
