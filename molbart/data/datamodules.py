import os
import pytorch_lightning as pl
import torch
from functools import partial
from loguru import logger
from torch.utils.data import DataLoader
from typing import List, Optional

from graph.datasets import GraphStringDataset, StringDataset
from molbart.data.datasets import RegPropDataset
from molbart.data.util import TokenSampler


class _AbsDataModule(pl.LightningDataModule):
    def __init__(
        self,
        cfg,
        sample_type,
        task,
        datapath,
        tokeniser,
        batch_size,
        max_seq_len,
        sample_size=None,
        train_token_batch_size=None,
        num_buckets=None,
        pin_memory=False,
    ):
        # print(sample_type)
        super().__init__()

        self.datapath = datapath
        self.tokeniser = tokeniser

        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.train_token_batch_size = train_token_batch_size
        self.num_buckets = num_buckets
        self.pin_memory = pin_memory
        self.graph_dim = {"nodes": 1}

        self._num_workers = 8  # len(os.sched_getaffinity(0))

        ds_params = {"task": task, "sample_size": sample_size}
        if cfg.model.str and not cfg.model.graph:
            self.train_dataset = StringDataset(cfg, sample_type="train", **ds_params)
            self.val_dataset = StringDataset(cfg, sample_type="valid", **ds_params)
            self.test_dataset = StringDataset(cfg, sample_type="test", **ds_params)

        else:
            self.train_dataset = GraphStringDataset(
                cfg, sample_type="train", **ds_params
            )
            self.val_dataset = GraphStringDataset(cfg, sample_type="valid", **ds_params)
            self.test_dataset = GraphStringDataset(cfg, sample_type="test", **ds_params)
            self.graph_dim = self.train_dataset.graph_dims
            for dataset, sample_type in zip(
                [self.train_dataset, self.val_dataset, self.test_dataset],
                ["train", "val", "test"],
            ):
                ds_size = len(dataset.smiles)
                logger.info(
                    f"{sample_type} size: {ds_size} | "
                    f"Graph feats: {dataset.graph_dims}"
                )
                # dims = {"graph_dim": graph_dim, "str_dim": str_dim, "seq_size": seq_size}

    # Use train_token_batch_size with TokenSampler for training and batch_size for validation and testing
    def train_dataloader(self):
        if self.train_token_batch_size is None:
            loader = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                num_workers=self._num_workers,
                collate_fn=self._collate,
                shuffle=True,
                pin_memory=self.pin_memory,
            )
            return loader

        sampler = TokenSampler(
            self.num_buckets,
            self.train_dataset.seq_lengths,
            self.train_token_batch_size,
            shuffle=True,
        )
        loader = DataLoader(
            self.train_dataset,
            batch_sampler=sampler,
            num_workers=self._num_workers,
            collate_fn=self._collate,
            pin_memory=self.pin_memory,
        )
        return loader

    def val_dataloader(self):
        loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self._num_workers,
            collate_fn=partial(self._collate, train=False),
            pin_memory=self.pin_memory,
        )
        return loader

    def test_dataloader(self):
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self._num_workers,
            collate_fn=partial(self._collate, train=False),
            pin_memory=self.pin_memory,
        )
        return loader

    def _collate(self, batch, train=True):
        raise NotImplementedError()

    def _check_seq_len(self, tokens, mask):
        """Warn user and shorten sequence if the tokens are too long, otherwise return original

        Args:
            tokens (List[List[str]]): List of token sequences
            mask (List[List[int]]): List of mask sequences

        Returns:
            tokens (List[List[str]]): List of token sequences (shortened, if necessary)
            mask (List[List[int]]): List of mask sequences (shortened, if necessary)
        """

        seq_len = max([len(ts) for ts in tokens])
        if seq_len > self.max_seq_len:
            print(
                f"WARNING -- Sequence length {seq_len} is larger than maximum sequence size"
            )

            tokens_short = [ts[: self.max_seq_len] for ts in tokens]
            mask_short = [ms[: self.max_seq_len] for ms in mask]

            return tokens_short, mask_short

        return tokens, mask

    def _build_att_mask(self, enc_length, dec_length):
        seq_len = enc_length + dec_length
        enc_mask = torch.zeros((seq_len, enc_length))
        upper_dec_mask = torch.ones((enc_length, dec_length))
        lower_dec_mask = torch.ones((dec_length, dec_length)).triu_(1)
        dec_mask = torch.cat((upper_dec_mask, lower_dec_mask), dim=0)
        mask = torch.cat((enc_mask, dec_mask), dim=1)
        mask = mask.masked_fill(mask == 1, float("-inf"))
        return mask

    def _build_target_mask(self, enc_length, dec_length, batch_size):
        # Take one and add one because we shift the target left one token
        # So the first token of the target output will be at the same position as the separator token of the input,
        # And the separator token is not present in the output
        enc_mask = [1] * (enc_length - 1)
        dec_mask = [0] * (dec_length + 1)
        mask = [enc_mask + dec_mask] * batch_size
        mask = torch.tensor(mask, dtype=torch.bool).T
        return mask
