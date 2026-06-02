import numpy as np
import torch
from torch.utils.data import Dataset

class HeatSourceDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        T = data['T'].astype(np.float32)
        Q = data['Q'].astype(np.float32)

        # Normalize T to [0, 1] — critical given narrow 7K range
        self.T_min = T.min()
        self.T_max = T.max()
        T_norm = (T - self.T_min) / (self.T_max - self.T_min + 1e-8)

        # Normalize Q to [0, 1]
        self.Q_max = Q.max()
        Q_norm = Q / (self.Q_max + 1e-8)

        # Shape: (N, 1, 64, 64) — the "1" is the channel dimension for CNN
        self.T = torch.tensor(T_norm).unsqueeze(1)
        self.Q = torch.tensor(Q_norm).unsqueeze(1)

    def __len__(self):
        return len(self.T)

    def __getitem__(self, idx):
        return self.T[idx], self.Q[idx]