import torchvision.models as models
import torch.nn as nn


def build_model(name, device):
    if name == "ResNet18":
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 10, bias=True)
        model.to(device)
        return model

    if name == "se_resnet50":
        model = se_resnet50(num_classes=1000, pretrained='imagenet')
        model.last_linear = nn.Linear(model.last_linear.in_features, 2 * NUM_LANDMARKS, bias=True)
        model.to(device)
        return model
