import torch.nn as nn
import sys
import numpy as np
import torch

from argparse import ArgumentParser
from torch.utils.tensorboard import SummaryWriter
from dataloader import get_train_val_loader, get_test_loader
from transforms import get_train_transforms, get_val_transforms
from transforms import get_test_transforms
from utils import accuracy, custom_decrease
from model import build_model
from train import train_one_epoch, validate
from torch import optim
from utils import

def runner(args):
    """
    Main fuction that get config from command line and creates
    everything you need for training and testing - model, optimizer,
    loss function and metric. Also runner performs a training or testing
    process and logging.
    Params
    ------
    - args : contains all variables from command line
    """
    print("Preparation in progress ...")
    device = torch.device("cuda: 0") if args.gpu else torch.device("cpu")
    model = build_model("ResNet18", device)

    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, custom_decrease)
    loss_fn = nn.CrossEntropyLoss()
    metric = accuracy
    # Make config for easy use with functions
    config = {"model" : model,
              "optimizer" : optimizer,
              "lr_scheduler" : lr_scheduler,
              "loss_fn" : loss_fn,
              "metric" : metric,
              "device" : device}

    if (args.train):
        print("Training mode")
        print("Reading transforms")
        train_transforms = get_train_transforms()
        val_transforms = get_val_transforms()

        print("Reading data")
        train_dataloader, val_dataloader = get_train_val_loader(args.data_dir,
                                                                train_transforms,
                                                                val_transforms,
                                                                args.batch_size)
        # Add dataloaders in config
        config.update({"train_dataloader" : train_dataloader})
        config.update({"val_dataloader" : val_dataloader})

        print("Start training")
        # Writer save train loss and val loss on every last_epoch
        # Also writer save train loss on every batch
        writer = SummaryWriter()
        best_val_loss = np.inf
        for epoch in range(args.epoch):
            print(f"Starting {epoch}/{args.epoch} epoch")
            train_losses = train_one_epoch(config)
            train_loss = np.mean(train_losses)

            val_losses, val_metric_values = validate(config)
            val_loss = np.mean(val_losses)
            val_metric_value = np.mean(val_metric_values)
            # Logging
            writer.add_scalar('Loss/train', train_loss)
            writer.add_scalar('Loss/val', val_loss)
            writer.add_scalar('Metric/val', val_metric_value)
            print(f"Epoch {epoch}/{args.epoch}:" + \
                  f" train loss = {train_loss} " + \
                  f"val loss = {val_loss}")
            # Save best model
            if best_val_loss > val_loss:
                torch.save(model.state_dict(), f"{args.name}_best.pth")
                best_val_loss = val_loss
        print (f"Training is over! Best validation loss = {best_val_loss}")
    else:
        print("Testing mode")
        test_transforms = get_test_transform()
        test_dataloader = get_test_loader(args.data_dir,
                                          test_transforms,
                                          args.batch_size)
        # Add dataloader in config
        config.update({"test_dataloader" : test_dataloader})

        # Load weights
        # if weight path don't specified used best model from training
        if (args.weight_path == ""):
            state_dict = torch.load(f"{args.name}_best.pth")
            model.load_state_dict(state_dict)
        else:
            state_dict = torch.load(args.weight_path)
            model.load_state_dict(state_dict)

        test_predictions, metric_values = predict(config)
        print(f"Metric value on test = {np.mean(metric_values)}")


def parse_arguments():
    """
    Parse argument from command line

    Parser arguments
    ----------------
    - name: name of experiment for saving purposes.
    - train: enable train mode.
    - test: enable test mode.
    - epoch: amount of epoch for training.
    - batch_size: amount of samples per iteration during training.
    - learning_rate: initial learning rate.
    - weight_path: weight for loading.
    - gpu: select device for eval (if gpu not specified then used CPU).
    - data_dir: folder for downloading dataset
    """
    parser = ArgumentParser(__doc__)
    parser.add_argument("--name", "-n", help="Experiment name", default="baseline")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--epoch", default=80, type=int)
    parser.add_argument("--batch-size", "-b", default=512, type=int)
    parser.add_argument("--learning-rate", "-lr", default=1e-3, type=float)
    parser.add_argument("--weight-path", default="")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--data-dir", default="~")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    sys.exit(runner(args))
