import torch
import torch.nn.functional as F


def get_loss_function(name):
    """
    Returns a loss function based on name.
    Available options: 'mse', 'l1', 'smooth_l1', 'huber', 'fairness_weighted'
    """
    name = name.lower()

    if name == 'mse':
        return lambda pred, target: F.mse_loss(pred, target)

    elif name == 'l1':
        return lambda pred, target: F.l1_loss(pred, target)

    elif name == 'smooth_l1':
        return lambda pred, target: F.smooth_l1_loss(pred, target)

    elif name == 'huber':
        return lambda pred, target: F.huber_loss(pred, target, delta=1.0)

    elif name == 'fairness_weighted':
        def fairness_weighted_loss(pred, target):
            weight = torch.log1p(target + 1.0)  # or some domain-informed weight
            return F.mse_loss(pred * weight, target * weight)
        return fairness_weighted_loss

    else:
        raise ValueError(f"Unknown loss function: {name}")
