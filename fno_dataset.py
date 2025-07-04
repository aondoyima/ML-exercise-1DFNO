# fno_dataset.py

import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
import re

class FNODataset1DWithParams(Dataset):
    def __init__(self, sim_dirs, fields=['psi1', 'psi2', 'h'], param_names=None,
                 rollout_steps=1, mean=None, std=None):
        
        self.data = []
        self.fields = fields
        self.param_names = param_names or ['alpha']
        self.rollout_steps = rollout_steps
        self.mean = torch.tensor(mean) if mean is not None else None  # torch tensor of shape (C,)
        self.std = torch.tensor(std) if std is not None else None 

        for sim_dir in sim_dirs:
            frame_files = sorted(
                [f for f in os.listdir(sim_dir) if f.startswith("frame") and f.endswith(".p")],
                key=lambda f: int(f.replace("frame", "").replace(".p", ""))
            )

            if len(frame_files) < rollout_steps + 1:
                continue

            # Extract parameters
            param_vec = self.extract_params_from_path(sim_dir)
            if param_vec is None:
                continue

            frames = [pickle.load(open(os.path.join(sim_dir, f), 'rb')) for f in frame_files]

            for i in range(len(frames) - rollout_steps):
                input_seq = []
                target_seq = []

                for j in range(rollout_steps + 1):
                    arr = np.stack([frames[i + j][field] for field in self.fields], axis=0)  # (C, X)
                    if self.mean is not None and self.std is not None:
                        arr = (arr - self.mean[:, None].numpy()) / self.std[:, None].numpy()
                    if j == 0:
                        input_seq = arr  # single input (C, X)
                    else:
                        target_seq.append(arr)

                target_seq = np.stack(target_seq, axis=0)  # (rollout_steps, C, X)

                self.data.append((
                    torch.tensor(input_seq, dtype=torch.float32),
                    torch.tensor(param_vec, dtype=torch.float32),
                    torch.tensor(target_seq, dtype=torch.float32)
                ))
        
    def extract_params_from_path(self, path):
        param_file = os.path.join(path, 'params.p')
        if not os.path.exists(param_file):
            return None

        try:
            params_dict = pickle.load(open(param_file, 'rb'))
            return np.array([params_dict[p] for p in self.param_names], dtype=np.float32)
        except (KeyError, FileNotFoundError, pickle.UnpicklingError):
            return None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Returns: u0: (C, X), theta: (P,), target_seq: (N, C, X)
        return self.data[idx]
