import json
import random
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from datasets import Dataset, features

class VectorDataset(Dataset):
  def __init__(self, json_path, change_count, augment = False):
    with open(json_path, 'r') as f:
      data = json.load(f)

    self.codi_vectors = []
    self.change_count = change_count
    self.augment = augment

    for v in data.values():
      if isinstance(v, list) and len(v) == 3:
        vector = torch.tensor(v, dtype = torch.float32).view(3, 512)
        self.codi_vectors.append(vector)

        if self.augment:
          augmented_ver = self.augment_vector(vector)
          self.codi_vectors.extend(augmented_ver)

  def __len__(self):
    return len(self.codi_vectors)
  
  def __getitem__(self, idx):
    original_vector = self.codi_vectors[idx]
    modified_vector = self.modify_vector(original_vector)
    return original_vector, modified_vector
  
  def augment_vector(self, vector):
    augmented_versions = []

    augmented_versions.append(self.add_noise(vector.clone()))
    augmented_versions.append(self.scale_vector(vector.clone()))

    return augmented_versions
  
  def modify_vector(self, original_vector, change_count):
    change_count = self.change_count
    modified_vector = original_vector.clone()
    available_indices = list(range(3))
    for _ in range(change_count):
      if not available_indices:
        break
      item_index = random.choice(available_indices)
      available_indices.remove(item_index)
      random_idx = random.randrange(len(self.codi_vectors))
      random_codi = self.codi_vectors[random_idx]
      modified_vector[item_index] = random_codi[item_index]
    return modified_vector
  
  def add_noise(self, vector, noise_level = 0.01):
    noise = torch.randn_like(vector) * noise_level
    return vector + noise
  
  def scale_vector(self, vector, scale_range=(0.9, 1,1)):
    scale = random.uniform(*scale_range)
    return vector * scale