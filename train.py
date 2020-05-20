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
    for step, (x, y) in enumerate(train_loader):
        prediciton = model(x)

        loss = loss_fn(prediciton, y)
        losses.append(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Scheduler ReduceLROnPlateau should step after validation
        # if using another scheduler uncomment next line
        # lr_scheduler.step()

    return losses


def validate(model, loader, loss_fn, val_steps, writer, device):
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
    for step, (x, y) in enumerate(val_loader):

        with torch.no_grad():
            prediciton = model(x)

        loss = loss_fn(prediciton, y)
        losses.append(loss)

        metric_on_batch = metric(predicitons, y)
        metric_values.append(metric_on_batch)
    return losses, metric_values
