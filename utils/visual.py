from matplotlib import pyplot as plt
import numpy as np
from models.builder import build_model

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


def plot_train_val(name,
                   metric,
                   loss,
                   title="Train val",
                   label1="Train",
                   label2="Validation"):
    """
    Function plot graph for accuracy or loss.
    Params
    ------
    - name [str] : name of architecture for title.
    - metric [list] : train and val accuracy values.
    - loss [list] : train and val loss values.
    """
    with plt.style.context('bmh'):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
        ax1.set_title(f"{title} accuracy for {name}")
        ax1.set_ylabel("Accuracy")
        ax1.set_xlabel("Epoch")
        ax2.set_title(f"{title} loss for {name}")
        ax2.set_ylabel("Loss")
        ax2.set_xlabel("Epoch")

        for values, ax in zip([metric, loss], [ax1, ax2]):
            train = values.values[:, 1]
            val = values.values[:, 0]
            assert(len(train) == len(val))

            epoch = len(train)

            xs = range(epoch)

            ax.set_xlim(0, epoch)
            y_lim_min = min([*train, *val]) - 0.1 * max([*train, *val])
            y_lim_max = max([*train, *val]) + 0.1 * max([*train, *val])
            ax.set_ylim(y_lim_min, y_lim_max)

            ax.grid(True)
            # Plot train val data
            ax.plot(xs, train, label=label1)
            ax.plot(xs, val, label=label2)
            # Plot vertical lines
            line, = ax.plot([80] * 160, np.linspace(-20, 120, 160), 'm--', label='Reduce learning rate')
            line.set_linewidth(1)
            line, = ax.plot([120] * 160, np.linspace(-20, 120, 160), 'm--')
            line.set_linewidth(1)

            ax.legend()

        plt.show()


def plot_accuaracy(names, vals):
    """
    Function plot graph for accuracy on test dividing into classes
    Params
    ------
    - names [str] : name of architecture for title.
    - vals [list] : accuracy on test.
    """
    wide = {"name": [], "val": [], "label": []}
    deep = {"name": [], "val": [], "label": []}
    wide_name = []
    wide_val = []
    wide_label = []
    deep_name = []
    deep_val = []
    deep_label = []
    for name, val in zip(names, vals):
        name_split = name.split("_")
        if len(name_split) == 3:
            deep["name"].append(name_split[2])
            deep["val"].append(val)
            deep["label"].append(name_split[1])
        else:
            wide["name"].append("_".join(name_split[2:]))
            wide["val"].append(val)
            wide["label"].append(name_split[1])

    with plt.style.context('bmh'):
        assert(len(names) == len(vals))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle("Accuracy on varios architectures")
        ax1.set_title("Change deep")
        ax1.set_ylabel("Accuracy")
        ax1.set_xlabel("Network architecture")
        ax2.set_title("Change wide")
        ax2.set_ylabel("Accuracy")
        ax2.set_xlabel("Network architecture")

        # Plot train val data
        for arr, ax in zip([deep, wide],[ax1, ax2]):
            for label in set(arr["label"]):
                ns = []
                vs = []
                for v, n, l in zip(arr["val"], arr["name"], arr["label"]):
                    if l == label:
                        ns.append(n)
                        vs.append(v)
                ax.plot(ns, vs, label=label)
            ax.grid(True)
            ax.set_ylim(0, 100)
            major_ticks = np.arange(0, 100, 10)
            ax.set_yticks(major_ticks)

        ax.legend()

        plt.show()
