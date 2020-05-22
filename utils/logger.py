from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch


class Logger:
    """
    Logger save loss and metric values on every epoch. For correct
    working need to call train_step and val_step on every epoch.
    All training info in folder runs.
    """
    def __init__(self, epoch_max):
        self._writer = SummaryWriter()

        self._best_val_loss = np.inf
        self._best_val_metric = np.inf

        self._train_loss = 0
        self._train_metric = 0
        self._val_loss = 0
        self._val_metric = 0

        self._epoch = 0
        self._epoch_max = epoch_max

    def train_step(self, losses, metrics):
        self._train_loss = np.mean(losses)
        self._train_metric = np.mean(metrics)
        self._writer.add_scalar('Loss/train',
                                self._train_loss,
                                self._epoch)
        self._writer.add_scalar('Metric/train',
                                self._train_metric,
                                self._epoch)

    def val_step(self, losses, metrics):
        self._val_loss = np.mean(losses)
        self._val_metric = np.mean(metrics)
        self._writer.add_scalar('Loss/val', self._val_loss, self._epoch)
        self._writer.add_scalar('Metric/val', self._val_metric, self._epoch)
        self._epoch += 1

    def save_best(self, model, name):
        if self._best_val_loss > self._val_loss:
            torch.save(model.state_dict(), f"{name}_best.pth")
            self._best_val_loss = self._val_loss
            self._best_val_metric = self._val_metric

    def print_summary(self):
        print(f"Training is over! \n" +
              f"Best validation loss = {self._best_val_loss:.2f} \n" +
              f"Accuracy for best loss = {self._best_val_metric:.2f}")

    def print_current(self,):
        # Draw val and train on same plot
        self._writer.add_scalars('Loss/trainval',
                                 {'val': self._val_loss,
                                  'train': self._train_loss},
                                 self._epoch)
        self._writer.add_scalars('Metric/trainval',
                                 {'val': self._val_metric,
                                  'train': self._train_metric},
                                 self._epoch)
        print(f"Epoch {self._epoch}/{self._epoch_max}:\n" +
              f"TRAIN loss = {self._train_loss:.2f}   |   " +
              f"accuracy = {self._train_metric:.2f}%\n" +
              f"VALID loss = {self._val_loss:.2f}   |   " +
              f"accuracy = {self._val_metric:.2f}%")
