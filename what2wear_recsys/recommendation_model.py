import torch
import torch.nn.functional as F
import numpy as np

class W2WRecommendation:
  def __init__(self, model):
    self.model = model

  def find_best_match(self,
                      top_vector=None,
                      bottom_vector=None,
                      shoes_vector=None,
                      bottoms=None, 
                      shoes=None, 
                      tops=None, 
                      bottom_paths=None,
                      shoes_paths=None,
                      top_paths=None):

    if top_vector is None and bottom_vector is None and shoes_vector is None:
      raise ValueError("Invalid input : Please provide appropriate vectors")
    else:
      return self._find_best_items(tops, bottoms, shoes, top_paths, bottom_paths, shoes_paths, top_vector, bottom_vector, shoes_vector)
    
  def _find_best_items(self,
                      tops,
                      bottoms,
                      shoes,
                      top_paths,
                      bottom_paths,
                      shoes_paths,
                      top_vector=None,
                      bottom_vector=None,
                      shoes_vector=None,):
      
      if top_vector is not None:
        candidate_tops = [(top_vector, -1)]
      else:
        candidiate_tops = list(zip(tops, range(len(tops))))

      if bottom_vector is not None:
        candidate_bottoms = [(bottom_vector, -1)]
      else:
        candidate_bottoms = list(zip(bottoms, range(len(bottoms))))

      if shoes_vector is not None:
        candidate_shoes = [(shoes_vector, -1)]

      best_loss = float('inf')
      best_top_idx = None
      best_bottom_idx = None
      best_shoes_idx = None

      for t_vec, t_idx in candidate_tops:
        for b_vec, b_idx in candidate_bottoms:
          for s_vec, s_idx in candidate_shoes:
            outfit_vector = torch.stack([t_vec, b_vec, s_vec], dim = 0)

            loss = self.model(outfit_vector.unsqueeze(0), outfit_vector.unsqueeze(0))

            if loss < best_loss:
              best_loss = loss
              best_top_idx = t_idx
              best_bottom_idx = b_idx
              best_shoes_idx = s_idx
      
      best_top_path = top_paths[best_top_idx] if best_top_idx != -1 else None
      best_bottom_path = bottom_paths[best_bottom_idx] if best_bottom_idx is not None else None
      best_shoes_path = shoes_paths[best_shoes_idx] if best_shoes_idx is not None else None

      return {
        "loss" : best_loss,
        "top_idx" : best_top_idx,
        "shoes_idx" : best_shoes_idx,
        "top_path" : best_top_path,
        "bottom_path" : best_bottom_path,
        "shoes_path" : best_shoes_path
      }