import torch

def huber_loss(y_true, y_pred, delta=1.0) -> float:
    """
    Compute the Huber Loss between true and predicted values.

    Args:
        y_true: Ground truth values (torch.Tensor, list, or scalar)
        y_pred: Predicted values (torch.Tensor, list, or scalar)
        delta (float): Transition threshold between MSE and MAE behavior

    Returns:
        float: Average Huber loss
    """
    # Your code here
    y_pred = torch.as_tensor(y_pred, dtype = torch.float)
    y_true = torch.as_tensor(y_true, dtype = torch.float)
    
    err = torch.abs(y_pred - y_true)
    loss = torch.where(
        err <= delta,
        0.5 * err ** 2,
        delta * (err - 0.5 * delta)
    )
    return loss.mean().item()