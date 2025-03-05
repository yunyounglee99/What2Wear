import torch
import random
import torch.nn as nn
import torch.nn.functional as F

class MatchingLoss(nn.Module):
  def __init__(self, threshold, temperature, dropout_rate):
    super(MatchingLoss, self).__init__()
    self.threshold = threshold
    self.temperature = nn.Parameter(torch.tensor(temperature))
    self.W = nn.Parameter(torch.randn(512, 512))
    self.dropout = nn.Dropout(dropout_rate)
    self.norm_layer = nn.LayerNorm(512)

  def forward(self, original_vectors, modified_vectors):
    original_embedded = F.normalize(torch.matmul(original_vectors, self.W), p=2, dim=-1)
    modified_embedded = F.normalize(torch.matmul(modified_vectors, self.W), p=2, dim=-1)

    original_similarties = []
    modified_similarties = []

    for i in range(3):
      for j in range(i+1, 3):
        original_sim = F.cosine_similarity(
          original_embedded[:, i, :],
          original_embedded[:, j, :],
          dim=-1
        ) / self.temperature
        modified_sim = F.cosine_similarity(
          modified_embedded[:, i, :],
          modified_embedded[:, j, :],
          dim=-1
        ) / self.temperature

        original_similarties.append(original_sim)
        modified_similarties.append(modified_sim)

    original_similarties = torch.stack(original_similarties, dim=1).mean(dim=1)
    modified_similarties = torch.stack(modified_similarties, dim=1).mean(dim=1)

    target_loss = torch.relu(1-original_similarties.mean())
    matching_loss = torch.relu(self.threshold - modified_similarties.mean())
    total_loss = target_loss + matching_loss

    return total_loss + 0.01*torch.norm(self.W, p=2)