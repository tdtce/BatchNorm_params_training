from tqdm import tqdm
import torch


def train_one_epoch(config):
    """
    Function read config and train the model for one epoch.
    Params
    ------
    - config [dict] : save all training component - model, loader, device, loss
      function, optimizer and scheduler.
    Returns
    -------
    - losses [list]: list of batch losses during one epoch.
    """
    # Read config
    model = config["model"]
    train_loader = config["train_dataloader"]
    device = config["device"]
    loss_fn = config["loss_fn"]
    optimizer = config["optimizer"]
    lr_scheduler = config["lr_scheduler"]

    model.train()
    losses = []

    # Make iterator
    t = tqdm(iter(train_loader), leave=False, total=len(train_loader))
    for step, (x, y) in enumerate(t):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        prediciton = model(x)

        loss = loss_fn(prediciton, y)
        losses.append(loss.item())
        loss.backward()

        optimizer.step()
        lr_scheduler.step()

    return losses


def validate(config):
    """
    Function read config and validate model over validation data loader.
    Params
    ------
    - config [dict] : save all training component - model, loader, device, loss
      function, optimizer and scheduler.
    Returns
    -------
    - losses [list]: list of batch losses.
    """
    # Read config
    model = config["model"]
    val_loader = config["val_dataloader"]
    device = config["device"]
    loss_fn = config["loss_fn"]
    metric = config["metric"]
    model.eval()

    losses = []
    metric_values = []

    # Make iterator
    t = tqdm(iter(val_loader), leave=False, total=len(val_loader))
    for step, (x, y) in enumerate(t):
        x = x.to(device)
        y = y.to(device)

        with torch.no_grad():
            prediciton = model(x)

        loss = loss_fn(prediciton, y)
        losses.append(loss.item())

        metric_on_batch = metric(prediciton, y)
        metric_values.append(metric_on_batch)
    return losses, metric_values
