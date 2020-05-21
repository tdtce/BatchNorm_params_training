from tqdm import tqdm
import torch


def predict(config):
    """
    Function read config and test the model on test data.
    Params
    ------
    - config [dict] : save all training component - model, loader, device, loss
      function, optimizer and scheduler.
    Returns
    -------
    - metric_values [list]: list of metric values during one epoch.
    """
    # Read config
    model = config["model"]
    test_loader = config["test_dataloader"]
    device = config["device"]
    metric = config["metric"]

    model.eval()
    metric_values = []
    predictions = []

    # Make iterator
    t = tqdm(iter(test_loader), leave=False, total=len(test_loader))
    for step, (x, y) in enumerate(t):
        x = x.to(device)
        y = y.to(device)

        with torch.no_grad():
            pred = model(x)
        predictions.extend(pred)
        metric_on_batch = metric(pred, y)
        metric_values.append(metric_on_batch)
    return predictions, metric_values
