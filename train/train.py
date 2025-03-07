import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from datasets import Dataset, features
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from ..dataset.vector_dataset import VectorDataset
from ..models.model import What2Wear
import yaml

def train(config, save_path):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  dataset = VectorDataset(json_path=config['data_path'], change_count=config['change_count'])
  
  train_size = int(0.8 * len(dataset))
  test_size = len(dataset) - train_size
  train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
  train_loader = DataLoader(train_dataset, batch_size = 8, shuffle = True, collate_fn = collate_fn)
  test_loader = DataLoader(test_dataset, batch_size = 8, shuffle = False, collate_fn = collate_fn)

  model = What2Wear(threshold = config['threshold'], temperature = config['temperature'], dropout_rate = config['dropout_rate'])
  model.to(device)

  optimizer = Adam(model.parameters(), lr = config['learning_rate'], weight_decay = config['weight_decay'])
  scheduler = StepLR(optimizer, step_size = config['step_size'], gamma=config['gamma'])

  train_losses = []
  val_losses = []

  for epoch in range(config['num_epochs']):
    model.train()
    total_loss = 0
    batch_count = 0

    for i in range(0, len(train_dataset), config['batch_size']):
      batch_indices = list(range(i+config['batch_size'], len(train_dataset)))
      original_vectors_list = []
      modified_vectors_list = []

      for idx in batch_indices:
        original, modified = dataset[idx]
        original_vectors_list.append(original)
        modified_vectors_list.append(modified)
      
      original_vectors = torch.stack(original_vectors_list).to(device)
      modified_vectors = torch.stack(modified_vectors_list).to(device)

      optimizer.zero_grad()
      loss = model(original_vectors, modified_vectors)
      loss.backward()
      optimizer.step()

      total_loss += loss.item()
      batch_count += 1

    avg_loss = total_loss / batch_count
    train_losses.append(avg_loss)
    print(f"Epoch [{epoch+1}/{config['num_epochs']}], Training Loss : {avg_loss:4f}")

    model.eval()
    val_loss = 0
    with torch.no_grad():
      for i in range(0, len(test_dataset), config['batch_size']):
        batch_indices = list(range(i+config['batch_size'], len(test_dataset)))
        original_vectors_list = []
        modified_vectors_list = []

        for idx in batch_indices:
          original, modified = dataset[idx]
          original_vectors_list.append(original)
          modified_vectors_list.append(modified)
        
        original_vectors = torch.stack(original_vectors_list).to(device)
        modified_vectors = torch.stack(modified_vectors_list).to(device)

        loss = model(original_vectors, modified_vectors)
        val_loss += loss.item()

    val_loss /= (len(test_dataset) // config['batch_size'])
    val_losses.append(val_loss)
    print(f"Epoch [{epoch+1}/{config['num_epochs']}], Validation Loss: {val_loss:.4f}")

    scheduler.step()

def collate_fn(batch):
  original_vectors, modified_vectors = zip(*batch)
  return torch.stack(original_vectors), torch.stack(modified_vectors)

# if __name__ == "__main__":