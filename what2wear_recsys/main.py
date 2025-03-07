import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from ..models.model import What2Wear
from ..what2wear_recsys.recommendation_model import W2WRecommendation

def item_recommendation(save_path):
  model = What2Wear()
  model.load_state_dict(torch.load(save_path))
  model.eval()

  rec_sys = W2WRecommendation(model)

  output = rec_sys.find_best_match()
