import os
import random
import shutil
import sys
import uuid
from typing import Any

import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from ..utils.setting import seed_everything


# データセット
class MyDataset(Dataset):
    def __init__(self, data, targets) -> None:
        self.data = data
        self.targets = targets

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Any:
        return self.data[index], self.target[index]


# ネットワーク
class MyNet(nn.Module):
    def __init__(
        self, in_features, n_layers, hidden_dim, out_features, dropout_rate=0.5
    ):
        super(MyNet, self).__init__()

        layers = []
        layers.append(nn.Linear(in_features, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(p=dropout_rate))
        for _ in range(n_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_rate))

        self.middle_layers = nn.Sequential(*layers)
        self.final_layer = nn.Linear(hidden_dim, out_features)

    def forward(self, x):
        h = self.middle_layers(x)
        out = self.final_layer(h)
        return out
