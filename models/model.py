import torch.nn as nn
from .matching_loss import MatchingLoss

class What2Wear(nn.Module):
  def __init__(self, threshold, temperature, dropout_rate):
    super().__init__()
    self.loss_fn = MatchingLoss(threshold = threshold, temperature = temperature)

  def forward(self, original_vectors, modified_vectors):
    return self.loss_fn(original_vectors, modified_vectors)