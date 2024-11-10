
# data_loader.py
import torch
import torch.utils.data as Data
from torch.utils.data import Dataset, Subset
import joblib
import os
from dataclasses import dataclass
import numpy as np
from transformers import WhisperTokenizer
import csv
import whisper 
from sklearn.model_selection import train_test_split
import torchaudio.transforms as at
import torchaudio
from typing import Any, Dict, List, Union
# ... (Rest of your code from data_loader_script.py)

def train_dataloader(train_dataset, collate_fn):
    return Data.DataLoader(
        train_dataset,
        batch_size=2,
        drop_last=True,
        shuffle=True,
        num_workers=2,  # Now this should work correctly
        collate_fn=collate_fn
    )

def eval_dataloader(eval_dataset, collate_fn):
    return Data.DataLoader(
        eval_dataset,
        batch_size=2,
        num_workers=2,  # Now this should work correctly
        collate_fn=collate_fn
    )
