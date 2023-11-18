import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from typing import Tuple, Any
import warnings

base2code_dna = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}

class RNAdata(Dataset):

    @property
    def train_labels(self):
        warnings.warn("train_labels has been renamed label")
        return self.label

    @property
    def test_labels(self):
        warnings.warn("test_labels has been renamed label")
        return self.label

    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.data

    @property
    def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.data

    def __init__(self, datafile, seqlen):
        """
        Inputs:
            mode: train, valid, test
        """
        self.datafile = datafile
        self.seqlen = seqlen

        self.data = pd.read_csv(datafile, index_col=0).to_numpy()

        self._len = len(self.data[0][0])

    def __getitem__(self, index: int) -> Tuple[Any, Any]:

        x = self.data[index][0][self._len//2 - self.seqlen//2: self._len//2 + self.seqlen//2 + 1]
        y = self.data[index][1]

        x = np.array([base2code_dna[i] for i in x])
        x = torch.tensor(x)
        y = torch.tensor(y)

        return x, y

    def __len__(self):

        return len(self.data)