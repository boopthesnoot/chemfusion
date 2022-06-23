import dgl
import math
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from pysmilesutils.augment import SMILESAugmenter
from rdkit import Chem
from torch.optim.lr_scheduler import OneCycleLR
from typing import List, Optional

from molbart.data.datamodules import _AbsDataModule
from molbart.models.pre_train import _AbsTransformerModel
from molbart.models.util import PreNormEncoderLayer, PreNormDecoderLayer
from molbart.tokeniser import MolEncTokeniser


# ----------------------------------------------------------------------------------------------------------
# ---------------------------------------------- Data Modules ----------------------------------------------
# ----------------------------------------------------------------------------------------------------------


class RegPropDataModule(_AbsDataModule):
    """
    code adapted from Irwin Ross
    """

    def __init__(
        self,
        cfg,
        sample_type,
        task,
        datapath: Path,
        tokeniser: MolEncTokeniser,
        batch_size: int,
        max_seq_len: int,
        sample_size,
        train_token_batch_size: Optional[int] = None,
        num_buckets: Optional[int] = None,
        forward_pred: Optional[bool] = True,
        augment: Optional[bool] = True,
    ):
        super().__init__(
            cfg,
            sample_type,
            task,
            datapath,
            tokeniser,
            batch_size,
            max_seq_len,
            sample_size,
            train_token_batch_size,
            num_buckets,
        )

        if augment:
            print("Augmenting the SMILES strings.")
            self.aug = SMILESAugmenter()  # SMILESRandomizer() : List[str]->List[str]
        else:
            print("No data augmentation.")
            self.aug = None

        self.forward_pred = forward_pred

    def _collate(self, batch, train=True):
        # print("collating", batch)
        token_output = self._prepare_tokens(batch)
        enc_tokens = token_output["encoder_tokens"]
        enc_pad_mask = token_output["encoder_pad_mask"]
        dec_tokens = token_output["decoder_tokens"]
        dec_pad_mask = token_output["decoder_pad_mask"]
        target_smiles = token_output["target_smiles"]

        enc_token_ids = self.tokeniser.convert_tokens_to_ids(enc_tokens)
        dec_token_ids = self.tokeniser.convert_tokens_to_ids(dec_tokens)

        enc_token_ids = torch.tensor(enc_token_ids).transpose(0, 1)
        enc_pad_mask = torch.tensor(enc_pad_mask, dtype=torch.bool).transpose(0, 1)
        dec_token_ids = torch.tensor(dec_token_ids).transpose(0, 1)
        dec_pad_mask = torch.tensor(dec_pad_mask, dtype=torch.bool).transpose(0, 1)

        # SMILES_tokens = token_output["SMILES_tokens"]
        # SMILES_mask = token_output["SMILES_mask"]
        graphs = token_output["graphs"]
        targets = token_output["props"]
        # raw_smiles = token_output["raw_smiles"]

        # SMILES_token_ids = self.tokeniser.convert_tokens_to_ids(SMILES_tokens)
        # SMILES_token_ids = torch.tensor(SMILES_token_ids).transpose(0, 1)
        # SMILES_pad_mask = torch.tensor(SMILES_mask, dtype=torch.bool).transpose(0, 1)

        collate_output = {
            # stay consistent with Ross code for the dictionary keys
            "graphs": dgl.batch(graphs),
            # "raw_smiles": raw_smiles,
            # "encoder_input": SMILES_token_ids,
            # "encoder_pad_mask": SMILES_pad_mask,
            "encoder_input": enc_token_ids,
            "encoder_pad_mask": enc_pad_mask,
            "decoder_input": dec_token_ids[:-1, :],
            "decoder_pad_mask": dec_pad_mask[:-1, :],
            "target_reco": dec_token_ids.clone()[1:, :],
            "target_mask": dec_pad_mask.clone()[1:, :],
            "target_smiles": target_smiles,
            "target": targets,
        }
        return collate_output

    def _prepare_tokens(self, batch):
        self.task = "mask"

        graph, inpSMILES, props = tuple(zip(*batch))
        batch = inpSMILES
        aug = self.aug is not None
        if aug:
            encoder_mols = self.aug(batch)
        else:
            encoder_mols = batch[:]

        if self.task == "mask" or self.task is None:
            decoder_mols = encoder_mols[:]
        elif self.task == "mask_aug" or self.task == "aug":
            decoder_mols = self.aug(encoder_mols)
        else:
            raise ValueError(f"Unknown task: {self.task}")

        canonical = self.aug is None
        enc_smiles = []
        dec_smiles = []

        # There is a very rare possibility that RDKit will not be able to generate the SMILES for the augmented mol
        # In this case we just use the canonical mol to generate the SMILES
        for idx, (enc_mol, dec_mol) in enumerate(zip(encoder_mols, decoder_mols)):
            # print(enc_mol, dec_mol)
            enc_mol = Chem.MolFromSmiles(enc_mol)
            dec_mol = Chem.MolFromSmiles(dec_mol)
            try:
                enc_smi = Chem.MolToSmiles(enc_mol, canonical=canonical)
            except RuntimeError:
                enc_smi = Chem.MolToSmiles(batch[idx], canonical=True)
                print(f"Could not generate smiles after augmenting: {enc_smi}")

            try:
                dec_smi = Chem.MolToSmiles(dec_mol, canonical=canonical)
            except RuntimeError:
                dec_smi = Chem.MolToSmiles(batch[idx], canonical=True)
                print(f"Could not generate smiles after augmenting: {dec_smi}")

            enc_smiles.append(enc_smi)
            dec_smiles.append(dec_smi)
        if self.task == "aug" or self.task is None:
            enc_token_output = self.tokeniser.tokenise(enc_smiles, pad=True)
            enc_tokens = enc_token_output["original_tokens"]
            enc_mask = enc_token_output["original_pad_masks"]
        elif self.task == "mask" or self.task == "mask_aug":
            enc_token_output = self.tokeniser.tokenise(enc_smiles, mask=True, pad=True)
            enc_tokens = enc_token_output["masked_tokens"]
            enc_mask = enc_token_output["masked_pad_masks"]
        else:
            raise ValueError(f"Unknown task: {self.task}")

        dec_token_output = self.tokeniser.tokenise(dec_smiles, pad=True)
        dec_tokens = dec_token_output["original_tokens"]
        dec_mask = dec_token_output["original_pad_masks"]

        enc_tokens, enc_mask = self._check_seq_len(enc_tokens, enc_mask)
        dec_tokens, dec_mask = self._check_seq_len(dec_tokens, dec_mask)

        # Ensure that the canonical form is used for evaluation
        dec_mols = [Chem.MolFromSmiles(smi) for smi in dec_smiles]
        canon_targets = [Chem.MolToSmiles(mol) for mol in dec_mols]

        token_output = {
            "graphs": graph,
            "SMILES_tokens": enc_tokens,
            "SMILES_mask": enc_mask,
            "props": torch.unsqueeze(torch.tensor(props), dim=1),
            "encoder_tokens": enc_tokens,
            "encoder_pad_mask": enc_mask,
            "decoder_tokens": dec_tokens,
            "decoder_pad_mask": dec_mask,
            "target_smiles": canon_targets,
        }
        return token_output


# ----------------------------------------------------------------------------------------------------------
# -------------------------------------------  Models ------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------


class RegrTransformerModel(pl.LightningModule):
    """
    code adapted from Irwin Ross

    Encoder for Regression to train from scratch
    """

    def __init__(
        self,
        vocab_size,
        d_model,
        num_layers,
        num_heads,
        d_feedforward,
        h_feedforward,
        lr,
        weight_decay,
        activation,
        num_steps,
        dropout,
        dropout_p,
        max_seq_len,
        batch_size,
        epochs,
        augment=None,
    ):
        super(RegrTransformerModel, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_feedforward = d_feedforward
        self.h_feedforward = h_feedforward
        self.lr = lr
        self.weight_decay = weight_decay
        self.activation = activation
        self.num_steps = num_steps
        self.dropout = dropout
        self.dropout_p = dropout_p
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.epochs = epochs
        self.augment = augment

        self.save_hyperparameters()

        self.emb = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pos_emb", self._positional_embs())

        enc_norm = nn.LayerNorm(d_model)
        enc_layer = PreNormEncoderLayer(
            d_model, num_heads, d_feedforward, dropout, activation
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers, norm=enc_norm)

        dec_norm = nn.LayerNorm(d_model)
        dec_layer = PreNormDecoderLayer(
            d_model, num_heads, d_feedforward, dropout, activation
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers, norm=dec_norm)

        self.token_fc = nn.Linear(d_model, vocab_size)
        self.loss_fn = nn.CrossEntropyLoss(reduction="none", ignore_index=pad_token_idx)
        self.log_softmax = nn.LogSoftmax(dim=2)

        # Regression part
        self.drpmem = nn.Dropout(dropout_p)
        self.ln = nn.LayerNorm(d_model)
        self.hidden_fc = nn.Linear(d_model, h_feedforward)

        self.drp = nn.Dropout(dropout_p)
        self.ln2 = nn.LayerNorm(h_feedforward)

        self.predict_fc = nn.Linear(h_feedforward, 1)
        self.loss_fn = nn.MSELoss()

        self._init_params()

    def __deepcopy__(self):
        return self

    def forward(self, x):
        """Apply SMILES strings to model

        The dictionary returned will be passed to other functions, so its contents are fairly flexible,
        except that it must contain the key "token_output" which is the output of the model
        (possibly after any fully connected layers) for each token.

        Arg:
            x (dict {
                "masked_tokens": tensor of token_ids of shape (seq_len, batch_size),
                "pad_masks": bool tensor of padded elems of shape (seq_len, batch_size),
                "sentence_masks" (optional): long tensor (0 or 1) of shape (seq_len, batch_size)
            }):

        Returns:
            Output from model (dict containing key "token_output")
        """

        encoder_input = x["encoder_input"]
        encoder_pad_mask = x["encoder_pad_mask"].transpose(0, 1)

        encoder_embs = self._construct_input(encoder_input)

        memory = self.encoder(encoder_embs, src_key_padding_mask=encoder_pad_mask)

        memory = memory[
            1, :, :
        ]  # in the 2rd element is the gene_symbol i.e. 1st in python
        # eg '^','<OPRD1>','O','=','C','1','N',...

        #         memory = torch.mean(memory,dim=0)  #memory.reshape(memory.shape[1], -1) #to take the average from all last outputs of encoder

        x = self.drpmem(memory)
        x = self.ln(x)
        x = self.hidden_fc(x)

        x = F.relu(x)
        x = self.drp(x)
        x = self.ln2(x)
        model_output = self.predict_fc(x)

        return model_output

    def training_step(self, batch, batch_idx):
        self.train()
        model_output = self.forward(batch)
        loss = self.loss_fn(batch["target"], model_output)
        self.log("train_loss", loss, on_step=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        model_output = self.forward(batch)
        loss = self.loss_fn(batch["target"], model_output)
        self.log("val_loss", loss, on_epoch=True, on_step=False, prog_bar=True)

    def test_step(self, batch, batch_idx):
        model_output = self.forward(batch)
        loss = self.loss_fn(batch["target"], model_output)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            betas=(0.9, 0.998),
            eps=1e-08,
            weight_decay=self.weight_decay,
        )

        scheduler = OneCycleLR(optim, max_lr=self.lr, total_steps=self.num_steps)
        lr_dict = {"scheduler": scheduler, "interval": "step"}

        return [optim], [lr_dict]

    def _construct_input(self, token_ids, sentence_masks=None):
        seq_len, _ = tuple(token_ids.size())
        token_embs = self.emb(token_ids)

        # Scaling the embeddings like this is done in other transformer libraries
        token_embs = token_embs * math.sqrt(self.d_model)

        positional_embs = self.pos_emb[:seq_len, :].unsqueeze(0).transpose(0, 1)
        embs = token_embs + positional_embs
        embs = self.dropout(embs)
        return embs

    def _positional_embs(self):
        """Produces a tensor of positional embeddings for the model

        Returns a tensor of shape (self.max_seq_len, self.d_model) filled with positional embeddings,
        which are created from sine and cosine waves of varying wavelength
        """

        encs = torch.tensor([dim / self.d_model for dim in range(0, self.d_model, 2)])
        encs = 10000 ** encs
        encs = [
            (torch.sin(pos / encs), torch.cos(pos / encs))
            for pos in range(self.max_seq_len)
        ]
        encs = [torch.stack(enc, dim=1).flatten()[: self.d_model] for enc in encs]
        encs = torch.stack(encs)
        return encs

    def _generate_square_subsequent_mask(self, sz):
        """
        Method from Pytorch transformer.
        Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).

        Args:
            sz (int): Size of mask to generate

        Returns:
            torch.Tensor: Square autoregressive mask for decode
        """

        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def _init_params(self):
        """
        Apply Xavier uniform initialisation of learnable weights
        or Kaiming He uniform initialisation
        """

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


##################################################################################################################


class FineTuneTransformerModel(pl.LightningModule):
    """
    code adapted from Irwin Ross
    Encoder for Regression to fine-tune
    """

    def __init__(
        self,
        vocab_size,
        d_premodel,
        premodel,
        h_feedforward,
        lr,
        weight_decay,
        activation,
        num_steps,
        dropout_p,
        max_seq_len,
        batch_size,
        epochs,
        augment=None,
    ):
        super(FineTuneTransformerModel, self).__init__()

        self.premodel = premodel
        self.d_premodel = d_premodel
        self.h_feedforward = h_feedforward
        self.lr = lr
        self.weight_decay = weight_decay
        self.activation = activation
        self.num_steps = num_steps
        self.dropout_p = dropout_p
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.epochs = epochs
        self.augment = augment

        # Regression part
        self.drpmem = nn.Dropout(dropout_p)
        self.ln = nn.LayerNorm(self.d_premodel)
        self.hidden_fc = nn.Linear(self.d_premodel, h_feedforward)
        self.drp = nn.Dropout(dropout_p)
        self.predict_fc = nn.Linear(h_feedforward, 1)
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        """Apply SMILES strings to model

        The dictionary returned will be passed to other functions, so its contents are fairly flexible,
        except that it must contain the key "token_output" which is the output of the model
        (possibly after any fully connected layers) for each token.

        Arg:
            x (dict {
                "masked_tokens": tensor of token_ids of shape (seq_len, batch_size),
                "pad_masks": bool tensor of padded elems of shape (seq_len, batch_size),
                "sentence_masks" (optional): long tensor (0 or 1) of shape (seq_len, batch_size)
            }):

        Returns:
            Output from model (dict containing key "token_output")
        """

        memory = self.premodel(x)
        memory = memory[
            1, :, :
        ]  # in the 2rd element is the gene_symbol i.e. 1nd in python
        # eg '^','<OPRD1>','O','=','C','1','N',...
        #         memory = torch.mean(memory,dim=0) # take the average of the memory

        ## ADD A DROPOUT and try
        x = self.drpmem(memory)
        x = self.ln(x)
        x = self.hidden_fc(x)
        x = F.relu(x)
        x = self.drp(x)
        model_output = self.predict_fc(x)

        return model_output

    def training_step(self, batch, batch_idx):
        self.train()
        model_output = self.forward(batch)
        loss = self.loss_fn(
            batch["target"], model_output
        )  # loss_fct(logits.view(-1), labels.view(-1))
        self.log("train_loss", loss, on_step=True, logger=True)  # , prog_bar=True

        return loss

    def validation_step(self, batch, batch_idx):
        model_output = self.forward(batch)
        loss = self.loss_fn(batch["target"], model_output)
        self.log("val_loss", loss, on_epoch=True, on_step=False, prog_bar=True)

    def test_step(self, batch, batch_idx):
        model_output = self.forward(batch)
        loss = self.loss_fn(batch["target"], model_output)
        return loss

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        self.log("test_epoch_end_val", avg_loss)

    def configure_optimizers(self):
        optim = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            betas=(0.9, 0.998),
            eps=1e-08,
            weight_decay=self.weight_decay,
        )
        scheduler = OneCycleLR(optim, max_lr=self.lr, total_steps=self.num_steps)
        lr_dict = {"scheduler": scheduler, "interval": "step"}
        return [optim], [lr_dict]

    def _construct_input(self, token_ids, sentence_masks=None):
        seq_len, _ = tuple(token_ids.size())
        token_embs = self.emb(token_ids)

        # Scaling the embeddings like this is done in other transformer libraries
        token_embs = token_embs * math.sqrt(self.d_premodel)

        positional_embs = self.pos_emb[:seq_len, :].unsqueeze(0).transpose(0, 1)
        embs = token_embs + positional_embs
        embs = self.dropout(embs)
        return embs

    def _positional_embs(self):
        """Produces a tensor of positional embeddings for the model

        Returns a tensor of shape (self.max_seq_len, self.d_premodel) filled with positional embeddings,
        which are created from sine and cosine waves of varying wavelength
        """

        encs = torch.tensor(
            [dim / self.d_premodel for dim in range(0, self.d_premodel, 2)]
        )
        encs = 10000 ** encs
        encs = [
            (torch.sin(pos / encs), torch.cos(pos / encs))
            for pos in range(self.max_seq_len)
        ]
        encs = [torch.stack(enc, dim=1).flatten()[: self.d_premodel] for enc in encs]
        encs = torch.stack(encs)
        return encs

    def _generate_square_subsequent_mask(self, sz):
        """
        Method from Pytorch transformer.
        Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).

        Args:
            sz (int): Size of mask to generate

        Returns:
            torch.Tensor: Square autoregressive mask for decode
        """

        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def _init_params(self):
        """
        Apply Xavier uniform initialisation of learnable weights
        or Kaiming He uniform initialisation
        """

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


####################################### EncoderOfBARTModel ####################################################


class EncoderOfBARTModel(_AbsTransformerModel):
    """
    code adapted from Irwin Ross
    This is the same BARTModel class from Ross but
    the Decoder part is erased
    This is needed just to load the pretrained model
    """

    def __init__(
        self,
        decode_sampler,
        pad_token_idx,
        vocab_size,
        d_model,
        num_layers,
        num_heads,
        d_feedforward,
        lr,
        weight_decay,
        activation,
        num_steps,
        max_seq_len,
        schedule="cycle",
        warm_up_steps=None,
        dropout=0.1,
        **kwargs,
    ):
        super().__init__(
            pad_token_idx,
            vocab_size,
            d_model,
            num_layers,
            num_heads,
            d_feedforward,
            lr,
            weight_decay,
            activation,
            num_steps,
            max_seq_len,
            dropout,
            # schedule=schedule,
            warm_up_steps=warm_up_steps,
            **kwargs,
        )

        self.sampler = decode_sampler
        self.val_sampling_alg = "greedy"
        self.test_sampling_alg = "beam"
        self.num_beams = 10

        self.schedule = schedule
        self.warm_up_steps = warm_up_steps

        if self.schedule == "transformer":
            assert (
                warm_up_steps is not None
            ), "A value for warm_up_steps is required for transformer LR schedule"

        enc_norm = nn.LayerNorm(d_model)
        dec_norm = nn.LayerNorm(d_model)

        enc_layer = PreNormEncoderLayer(
            d_model, num_heads, d_feedforward, dropout, activation
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers, norm=enc_norm)

        dec_layer = PreNormDecoderLayer(
            d_model, num_heads, d_feedforward, dropout, activation
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers, norm=dec_norm)

        self.token_fc = nn.Linear(d_model, vocab_size)
        self.loss_fn = nn.CrossEntropyLoss(reduction="none", ignore_index=pad_token_idx)
        self.log_softmax = nn.LogSoftmax(dim=2)

        self._init_params()

    def forward(self, x):
        """Apply SMILES strings to model

        The dictionary returned will be passed to other functions, so its contents are fairly flexible,
        except that it must contain the key "token_output" which is the output of the model
        (possibly after any fully connected layers) for each token.

        Arg:
            x (dict {
                "encoder_input": tensor of token_ids of shape (src_len, batch_size),
                "encoder_pad_mask": bool tensor of padded elems of shape (src_len, batch_size),
                "decoder_input": tensor of decoder token_ids of shape (tgt_len, batch_size)
                "decoder_pad_mask": bool tensor of decoder padding mask of shape (tgt_len, batch_size)
            }):

        Returns:
            Output from model (dict containing key "token_output" and "model_output")
        """

        encoder_input = x["encoder_input"]
        encoder_pad_mask = x["encoder_pad_mask"].transpose(0, 1)

        encoder_embs = self._construct_input(encoder_input)

        memory = self.encoder(encoder_embs, src_key_padding_mask=encoder_pad_mask)
        return memory

    def encode(self, batch):
        """Construct the memory embedding for an encoder input

        Args:
            batch (dict {
                "encoder_input": tensor of token_ids of shape (src_len, batch_size),
                "encoder_pad_mask": bool tensor of padded elems of shape (src_len, batch_size),
            })

        Returns:
            encoder memory (Tensor of shape (seq_len, batch_size, d_model))
        """

        encoder_input = batch["encoder_input"]
        encoder_pad_mask = batch["encoder_pad_mask"].transpose(0, 1)
        encoder_embs = self._construct_input(encoder_input)
        model_output = self.encoder(encoder_embs, src_key_padding_mask=encoder_pad_mask)
        return model_output

    def decode(self, batch):
        """Construct an output from a given decoder input

        Args:
            batch (dict {
                "decoder_input": tensor of decoder token_ids of shape (tgt_len, batch_size)
                "decoder_pad_mask": bool tensor of decoder padding mask of shape (tgt_len, batch_size)
                "memory_input": tensor from encoded input of shape (src_len, batch_size, d_model)
                "memory_pad_mask": bool tensor of memory padding mask of shape (src_len, batch_size)
            })
        """

        decoder_input = batch["decoder_input"]
        decoder_pad_mask = batch["decoder_pad_mask"].transpose(0, 1)
        memory_input = batch["memory_input"]
        memory_pad_mask = batch["memory_pad_mask"].transpose(0, 1)

        decoder_embs = self._construct_input(decoder_input)

        seq_len, _, _ = tuple(decoder_embs.size())
        tgt_mask = self._generate_square_subsequent_mask(
            seq_len, device=decoder_embs.device
        )

        model_output = self.decoder(
            decoder_embs,
            memory_input,
            tgt_key_padding_mask=decoder_pad_mask,
            memory_key_padding_mask=memory_pad_mask,
            tgt_mask=tgt_mask,
        )
        token_output = self.token_fc(model_output)
        token_probs = self.log_softmax(token_output)
        return token_probs

    def configure_optimizers(self):
        params = self.parameters()
        optim = torch.optim.Adam(
            params, lr=self.lr, weight_decay=self.weight_decay, betas=(0.9, 0.999)
        )

        if self.schedule == "const":
            print("Using constant LR schedule.")
            sch = LambdaLR(optim, lr_lambda=lambda epoch: 1)

        elif self.schedule == "cycle":
            print("Using cyclical LR schedule.")
            cycle_sch = OneCycleLR(optim, self.lr, total_steps=self.num_steps)
            sch = {"scheduler": cycle_sch, "interval": "step"}

        elif self.schedule == "transformer":
            print("Using original transformer schedule.")
            trans_sch = FuncLR(optim, lr_lambda=self._transformer_lr)
            sch = {"scheduler": trans_sch, "interval": "step"}

        else:
            raise ValueError(f"Unknown schedule {self.schedule}")

        return [optim], [sch]

    def _transformer_lr(self, step):
        mult = self.d_model ** -0.5
        step = 1 if step == 0 else step  # Stop div by zero errors
        lr = min(step ** -0.5, step * (self.warm_up_steps ** -1.5))
        return self.lr * mult * lr

    def validation_step(self, batch, batch_idx):
        self.eval()

        model_output = self.forward(batch)
        target_smiles = batch["target_smiles"]

        loss = self._calc_loss(batch, model_output)
        token_acc = self._calc_token_acc(batch, model_output)
        perplexity = self._calc_perplexity(batch, model_output)
        mol_strs, log_lhs = self.sample_molecules(
            batch, sampling_alg=self.val_sampling_alg
        )
        metrics = self.sampler.calc_sampling_metrics(mol_strs, target_smiles)

        mol_acc = torch.tensor(metrics["accuracy"], device=loss.device)
        invalid = torch.tensor(metrics["invalid"], device=loss.device)

        # Log for prog bar only
        self.log("mol_acc", mol_acc, prog_bar=True, logger=False, sync_dist=True)

        val_outputs = {
            "val_loss": loss,
            "val_token_acc": token_acc,
            "perplexity": perplexity,
            "val_molecular_accuracy": mol_acc,
            "val_invalid_smiles": invalid,
        }
        return val_outputs

    def validation_epoch_end(self, outputs):
        avg_outputs = self._avg_dicts(outputs)
        self._log_dict(avg_outputs)

    def test_step(self, batch, batch_idx):
        self.eval()

        model_output = self.forward(batch)
        target_smiles = batch["target_smiles"]

        loss = self._calc_loss(batch, model_output)
        token_acc = self._calc_token_acc(batch, model_output)
        perplexity = self._calc_perplexity(batch, model_output)
        mol_strs, log_lhs = self.sample_molecules(
            batch, sampling_alg=self.test_sampling_alg
        )
        metrics = self.sampler.calc_sampling_metrics(mol_strs, target_smiles)

        test_outputs = {
            "test_loss": loss.item(),
            "test_token_acc": token_acc,
            "test_perplexity": perplexity,
            "test_invalid_smiles": metrics["invalid"],
        }

        if self.test_sampling_alg == "greedy":
            test_outputs["test_molecular_accuracy"] = metrics["accuracy"]

        elif self.test_sampling_alg == "beam":
            test_outputs["test_molecular_accuracy"] = metrics["top_1_accuracy"]
            test_outputs["test_molecular_top_1_accuracy"] = metrics["top_1_accuracy"]
            test_outputs["test_molecular_top_2_accuracy"] = metrics["top_2_accuracy"]
            test_outputs["test_molecular_top_3_accuracy"] = metrics["top_3_accuracy"]
            test_outputs["test_molecular_top_5_accuracy"] = metrics["top_5_accuracy"]
            test_outputs["test_molecular_top_10_accuracy"] = metrics["top_10_accuracy"]

        else:
            raise ValueError(
                f"Unknown test sampling algorithm, {self.test_sampling_alg}"
            )

        return test_outputs

    def test_epoch_end(self, outputs):
        avg_outputs = self._avg_dicts(outputs)
        self._log_dict(avg_outputs)

    def _calc_loss(self, batch_input, model_output):
        """Calculate the loss for the model

        Args:
            batch_input (dict): Input given to model,
            model_output (dict): Output from model

        Returns:
            loss (singleton tensor),
        """

        tokens = batch_input["target"]
        pad_mask = batch_input["target_pad_mask"]
        token_output = model_output["token_output"]

        token_mask_loss = self._calc_mask_loss(token_output, tokens, pad_mask)

        return token_mask_loss

    def _calc_mask_loss(self, token_output, target, target_mask):
        """Calculate the loss for the token prediction task

        Args:
            token_output (Tensor of shape (seq_len, batch_size, vocab_size)): token output from transformer
            target (Tensor of shape (seq_len, batch_size)): Original (unmasked) SMILES token ids from the tokeniser
            target_mask (Tensor of shape (seq_len, batch_size)): Pad mask for target tokens

        Output:
            loss (singleton Tensor): Loss computed using cross-entropy,
        """

        seq_len, batch_size = tuple(target.size())

        token_pred = token_output.reshape((seq_len * batch_size, -1)).float()
        loss = self.loss_fn(token_pred, target.reshape(-1)).reshape(
            (seq_len, batch_size)
        )

        inv_target_mask = ~(target_mask > 0)
        num_tokens = inv_target_mask.sum()
        loss = loss.sum() / num_tokens

        return loss

    def _calc_perplexity(self, batch_input, model_output):
        target_ids = batch_input["target"]
        target_mask = batch_input["target_pad_mask"]
        vocab_dist_output = model_output["token_output"]

        inv_target_mask = ~(target_mask > 0)
        log_probs = vocab_dist_output.gather(2, target_ids.unsqueeze(2)).squeeze(2)
        log_probs = log_probs * inv_target_mask
        log_probs = log_probs.sum(dim=0)

        seq_lengths = inv_target_mask.sum(dim=0)
        exp = -(1 / seq_lengths)
        perp = torch.pow(log_probs.exp(), exp)
        return perp.mean()

    def _calc_token_acc(self, batch_input, model_output):
        token_ids = batch_input["target"]
        target_mask = batch_input["target_pad_mask"]
        token_output = model_output["token_output"]

        target_mask = ~(target_mask > 0)
        _, pred_ids = torch.max(token_output.float(), dim=2)
        correct_ids = torch.eq(token_ids, pred_ids)
        correct_ids = correct_ids * target_mask

        num_correct = correct_ids.sum().float()
        total = target_mask.sum().float()

        accuracy = num_correct / total
        return accuracy

    def sample_molecules(self, batch_input, sampling_alg="greedy"):
        """Sample molecules from the model

        Args:
            batch_input (dict): Input given to model
            sampling_alg (str): Algorithm to use to sample SMILES strings from model

        Returns:
            ([[str]], [[float]]): Tuple of molecule SMILES strings and log lhs (outer dimension is batch)
        """

        enc_input = batch_input["encoder_input"]
        enc_mask = batch_input["encoder_pad_mask"]

        # Freezing the weights reduces the amount of memory leakage in the transformer
        self.freeze()

        encode_input = {"encoder_input": enc_input, "encoder_pad_mask": enc_mask}
        memory = self.encode(encode_input)
        mem_mask = enc_mask.clone()

        _, batch_size, _ = tuple(memory.size())

        decode_fn = partial(self._decode_fn, memory=memory, mem_pad_mask=mem_mask)

        if sampling_alg == "greedy":
            mol_strs, log_lhs = self.sampler.greedy_decode(
                decode_fn, batch_size, memory.device
            )

        elif sampling_alg == "beam":
            mol_strs, log_lhs = self.sampler.beam_decode(
                decode_fn, batch_size, memory.device, k=self.num_beams
            )

        else:
            raise ValueError(f"Unknown sampling algorithm {sampling_alg}")

        # Must remember to unfreeze!
        self.unfreeze()

        return mol_strs, log_lhs

    def _decode_fn(self, token_ids, pad_mask, memory, mem_pad_mask):
        decode_input = {
            "decoder_input": token_ids,
            "decoder_pad_mask": pad_mask,
            "memory_input": memory,
            "memory_pad_mask": mem_pad_mask,
        }
        model_output = self.decode(decode_input)
        return model_output


class BARTModel(_AbsTransformerModel):
    def __init__(
        self,
        decode_sampler,
        pad_token_idx,
        vocab_size,
        d_model,
        num_layers,
        num_heads,
        d_feedforward,
        lr,
        weight_decay,
        activation,
        num_steps,
        max_seq_len,
        schedule="cycle",
        warm_up_steps=None,
        dropout=0.1,
        **kwargs,
    ):
        super().__init__(
            pad_token_idx,
            vocab_size,
            d_model,
            num_layers,
            num_heads,
            d_feedforward,
            lr,
            weight_decay,
            activation,
            num_steps,
            max_seq_len,
            schedule,
            warm_up_steps,
            dropout,
            **kwargs,
        )

        self.sampler = decode_sampler
        self.val_sampling_alg = "greedy"
        self.test_sampling_alg = "beam"
        self.num_beams = 10

        enc_norm = nn.LayerNorm(d_model)
        enc_layer = PreNormEncoderLayer(
            d_model, num_heads, d_feedforward, dropout, activation
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers, norm=enc_norm)

        dec_norm = nn.LayerNorm(d_model)
        dec_layer = PreNormDecoderLayer(
            d_model, num_heads, d_feedforward, dropout, activation
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers, norm=dec_norm)

        self.token_fc = nn.Linear(d_model, vocab_size)
        self.loss_fn = nn.CrossEntropyLoss(reduction="none", ignore_index=pad_token_idx)
        self.log_softmax = nn.LogSoftmax(dim=2)

        self._init_params()

    def forward(self, x):
        """Apply SMILES strings to model
        The dictionary returned will be passed to other functions, so its contents are fairly flexible,
        except that it must contain the key "token_output" which is the output of the model
        (possibly after any fully connected layers) for each token.
        Arg:
            x (dict {
                "encoder_input": tensor of token_ids of shape (src_len, batch_size),
                "encoder_pad_mask": bool tensor of padded elems of shape (src_len, batch_size),
                "decoder_input": tensor of decoder token_ids of shape (tgt_len, batch_size)
                "decoder_pad_mask": bool tensor of decoder padding mask of shape (tgt_len, batch_size)
            }):
        Returns:
            Output from model (dict containing key "token_output" and "model_output")
        """
        if self.mode == "reconstruction":

            encoder_input = x["encoder_input"]
            decoder_input = x["decoder_input"]
            encoder_pad_mask = x["encoder_pad_mask"].transpose(0, 1)
            decoder_pad_mask = x["decoder_pad_mask"].transpose(0, 1)

            encoder_embs = self._construct_input(encoder_input)
            decoder_embs = self._construct_input(decoder_input)

            seq_len, _, _ = tuple(decoder_embs.size())
            tgt_mask = self._generate_square_subsequent_mask(
                seq_len, device=encoder_embs.device
            )

            memory = self.encoder(encoder_embs, src_key_padding_mask=encoder_pad_mask)
            model_output = self.decoder(
                decoder_embs,
                memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=decoder_pad_mask,
                memory_key_padding_mask=encoder_pad_mask.clone(),
            )

            token_output = self.token_fc(model_output)

            output = {"model_output": model_output, "token_output": token_output}
        else:
            encoder_input = x["encoder_input"]
            encoder_pad_mask = x["encoder_pad_mask"].transpose(0, 1)

            encoder_embs = self._construct_input(encoder_input)

            memory = self.encoder(encoder_embs, src_key_padding_mask=encoder_pad_mask)
            return memory

        return output

    def encode(self, batch):
        """Construct the memory embedding for an encoder input
        Args:
            batch (dict {
                "encoder_input": tensor of token_ids of shape (src_len, batch_size),
                "encoder_pad_mask": bool tensor of padded elems of shape (src_len, batch_size),
            })
        Returns:
            encoder memory (Tensor of shape (seq_len, batch_size, d_model))
        """

        encoder_input = batch["encoder_input"]
        encoder_pad_mask = batch["encoder_pad_mask"].transpose(0, 1)
        encoder_embs = self._construct_input(encoder_input)
        model_output = self.encoder(encoder_embs, src_key_padding_mask=encoder_pad_mask)
        return model_output

    def decode(self, batch):
        """Construct an output from a given decoder input
        Args:
            batch (dict {
                "decoder_input": tensor of decoder token_ids of shape (tgt_len, batch_size)
                "decoder_pad_mask": bool tensor of decoder padding mask of shape (tgt_len, batch_size)
                "memory_input": tensor from encoded input of shape (src_len, batch_size, d_model)
                "memory_pad_mask": bool tensor of memory padding mask of shape (src_len, batch_size)
            })
        """

        decoder_input = batch["decoder_input"]
        decoder_pad_mask = batch["decoder_pad_mask"].transpose(0, 1)
        memory_input = batch["memory_input"]
        memory_pad_mask = batch["memory_pad_mask"].transpose(0, 1)

        decoder_embs = self._construct_input(decoder_input)

        seq_len, _, _ = tuple(decoder_embs.size())
        tgt_mask = self._generate_square_subsequent_mask(
            seq_len, device=decoder_embs.device
        )

        model_output = self.decoder(
            decoder_embs,
            memory_input,
            tgt_key_padding_mask=decoder_pad_mask,
            memory_key_padding_mask=memory_pad_mask,
            tgt_mask=tgt_mask,
        )
        token_output = self.token_fc(model_output)
        token_probs = self.log_softmax(token_output)
        return token_probs

    def _calc_loss(self, batch_input, model_output):
        """Calculate the loss for the model
        Args:
            batch_input (dict): Input given to model,
            model_output (dict): Output from model
        Returns:
            loss (singleton tensor),
        """

        tokens = batch_input["target"]
        pad_mask = batch_input["target_mask"]
        token_output = model_output["token_output"]

        token_mask_loss = self._calc_mask_loss(token_output, tokens, pad_mask)

        return token_mask_loss

    def _calc_mask_loss(self, token_output, target, target_mask):
        """Calculate the loss for the token prediction task
        Args:
            token_output (Tensor of shape (seq_len, batch_size, vocab_size)): token output from transformer
            target (Tensor of shape (seq_len, batch_size)): Original (unmasked) SMILES token ids from the tokeniser
            target_mask (Tensor of shape (seq_len, batch_size)): Pad mask for target tokens
        Output:
            loss (singleton Tensor): Loss computed using cross-entropy,
        """

        seq_len, batch_size = tuple(target.size())

        token_pred = token_output.reshape((seq_len * batch_size, -1)).float()
        loss = self.loss_fn(token_pred, target.reshape(-1)).reshape(
            (seq_len, batch_size)
        )

        inv_target_mask = ~(target_mask > 0)
        num_tokens = inv_target_mask.sum()
        loss = loss.sum() / num_tokens

        return loss

    def sample_molecules(self, batch_input, sampling_alg="greedy"):
        """Sample molecules from the model
        Args:
            batch_input (dict): Input given to model
            sampling_alg (str): Algorithm to use to sample SMILES strings from model
        Returns:
            ([[str]], [[float]]): Tuple of molecule SMILES strings and log lhs (outer dimension is batch)
        """

        enc_input = batch_input["encoder_input"]
        enc_mask = batch_input["encoder_pad_mask"]

        # Freezing the weights reduces the amount of memory leakage in the transformer
        self.freeze()

        encode_input = {"encoder_input": enc_input, "encoder_pad_mask": enc_mask}
        memory = self.encode(encode_input)
        mem_mask = enc_mask.clone()

        _, batch_size, _ = tuple(memory.size())

        decode_fn = partial(self._decode_fn, memory=memory, mem_pad_mask=mem_mask)

        if sampling_alg == "greedy":
            mol_strs, log_lhs = self.sampler.greedy_decode(
                decode_fn, batch_size, memory.device
            )

        elif sampling_alg == "beam":
            mol_strs, log_lhs = self.sampler.beam_decode(
                decode_fn, batch_size, memory.device, k=self.num_beams
            )

        else:
            raise ValueError(f"Unknown sampling algorithm {sampling_alg}")

        # Must remember to unfreeze!
        self.unfreeze()

        return mol_strs, log_lhs

    def _decode_fn(self, token_ids, pad_mask, memory, mem_pad_mask):
        decode_input = {
            "decoder_input": token_ids,
            "decoder_pad_mask": pad_mask,
            "memory_input": memory,
            "memory_pad_mask": mem_pad_mask,
        }
        model_output = self.decode(decode_input)
        return model_output
