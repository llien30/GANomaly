import torch


def L2_Loss(input, target, size_average=True):
    """
    Args:
        input (FloatTensor): Input tensor
        target (FloatTensor): Output tensor
    Returns:
        [FloatTensor]: L2 distance between input and target
    """
    if size_average:
        return torch.mean(torch.pow((input - target), 2))
    else:
        return torch.pow((input - target), 2)
