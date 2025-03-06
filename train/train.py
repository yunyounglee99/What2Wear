import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from datasets import Dataset, features
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

# def train(json_path, save_path, num_epoch, batch_size):