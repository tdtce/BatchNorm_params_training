import torchvision


def get_train_transforms():
    return torchvision.transforms.Compose([
        torchvision.transforms.RandomVerticalFlip(p=0.5),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        torchvision.transforms.RandomCrop(32, 4),
        torchvision.transforms.ToTensor(),
    ])


def get_val_transforms():
    return torchvision.transforms.Compose([
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        torchvision.transforms.ToTensor(),
    ])


def get_test_transforms():
    return torchvision.transforms.Compose([
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        torchvision.transforms.ToTensor(),
    ])
