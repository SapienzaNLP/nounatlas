import torch

def top_k_accuracy(preds: torch.Tensor, labels: torch.Tensor, k: int = 4):
    """Computes the top-k accuracy.

    Args:
    preds (torch.Tensor): The model predictions, a tensor of shape (N, C).
    labels (toch.Tensor): The ground truth labels, a tensor of shape (N,).
    k (int): The number of highest probabilities to consider.

    Returns (torch.Tensor): The top-k accuracy.
    """

    batch_size = labels.shape[0]
    top_k_probs, top_k_idx = torch.topk(preds, k, dim=1)
    correct = top_k_idx.eq(labels.view(-1, 1).expand_as(top_k_idx)).sum()
    return correct / batch_size

