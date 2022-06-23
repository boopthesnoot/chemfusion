import numpy as np
import pandas as pd
from pathlib import Path
from pysmilesutils.augment import MolAugmenter
from rdkit import Chem
from torch.utils.data import Dataset


class _AbsDataset(Dataset):
    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, item):
        raise NotImplementedError()


class RegPropDataset(_AbsDataset):
    def __init__(self, data_path, task, transform=None, aug_prob=0.0):
        super(_AbsDataset, self).__init__()
        path = Path(data_path)
        df = pd.read_csv(path)
        if task == "regression":
            df = df.dropna()
        data_in = df.iloc[:, 0].tolist()
        data_out = df.iloc[:, 1].tolist()

        if len(data_in) != len(data_out):
            raise ValueError(f"There must be an equal number of reactants and products")
        is_correct_mol = [
            True if Chem.MolFromSmiles(smi) is not None else False for smi in data_in
        ]
        is_not_too_long = [True if len(smi) < 120 else False for smi in data_in]
        self.smiles = np.array(data_in)[
            (np.asarray(is_correct_mol) & np.array(is_not_too_long))
        ]
        self.properties = np.array(data_out, dtype=float)[
            (np.asarray(is_correct_mol) & np.array(is_not_too_long))
        ]
        self.transform = transform
        self.aug_prob = aug_prob
        self.aug = MolAugmenter()


    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, item):
        # print("item", item)
        # print("self smiles", self.smiles, len(self.smiles))

        one_smiles = self.smiles[item]
        one_prop = self.properties[item]
        output = (one_smiles, one_prop)
        output = self.transform(*output) if self.transform is not None else output
        return output
