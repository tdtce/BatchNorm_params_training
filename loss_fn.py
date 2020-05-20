def accuracy(predicitons, labels):
    # Get id of predicted class
    _, predicted = torch.max(predicitons.data, 1)
    total = labels.size(0)
    correct = (predicted == labels).sum().item()
    # Accuracy in %
    return 100 * correct / total
