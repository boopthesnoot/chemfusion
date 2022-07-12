import copy
import dgl
import json
import math
import re
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from dgl.nn.pytorch import GraphConv
from torch.optim.lr_scheduler import OneCycleLR

sys.path.append("..")
from graph.finetune_regression_modules import (
    FineTuneTransformerModel,
)

from molbart.tokeniser import MolEncTokeniser

from molbart.models.util import PreNormEncoderLayer, PreNormDecoderLayer


class GraphTransformerModel(FineTuneTransformerModel):
    """
    code adapted from Irwin Ross
    Encoder for Regression to fine-tune
    """

    def __init__(
        self,
        cfg,
        mode,
        graph_dim,
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
        super(GraphTransformerModel, self).__init__(
            d_premodel=d_premodel,
            vocab_size=vocab_size,
            premodel=premodel,
            epochs=epochs,
            batch_size=batch_size,
            h_feedforward=h_feedforward,
            lr=lr,
            weight_decay=weight_decay,
            activation=activation,
            num_steps=num_steps,
            max_seq_len=max_seq_len,
            dropout_p=dropout_p,
            augment=augment,
        )
        self.vocab_size = vocab_size
        self.cfg = cfg
        self.register_buffer("pos_emb", self._positional_embs())
        self.graph_dim = graph_dim
        self.mode = mode
        params = cfg.layers
        self.params = params
        self.premodel = None
        self.emb = nn.Embedding(vocab_size, cfg.layers.d_model, padding_idx=0)

        # Reconstruction part:
        if cfg.run.reco or cfg.run.regr_with_reco:
            self.token_fc = nn.Linear(cfg.layers.d_model, vocab_size)
            self.reco_loss_fn = nn.CrossEntropyLoss(reduction="none", ignore_index=0)
            self.log_softmax = nn.LogSoftmax()
            dec_norm = nn.LayerNorm(cfg.layers.d_model)
            dec_layer = PreNormDecoderLayer(
                cfg.layers.d_model,
                cfg.layers.num_heads,
                cfg.layers.d_feedforward,
                cfg.layers.default_dropout,
                cfg.layers.activation,
            )
            self.decoder = nn.TransformerDecoder(
                dec_layer, cfg.layers.num_layers, norm=dec_norm
            )

        # Fusion
        if self.cfg.model.str and self.cfg.model.graph:
            self.fusion_mu = nn.Sequential(
                nn.Linear(2 * h_feedforward, 2 * h_feedforward),
                nn.Linear(2 * h_feedforward, h_feedforward),
            )

        # Graph
        if self.cfg.model.graph:
            self.gconv1 = GraphConv(self.graph_dim, params.gconv1)
            self.gconv2 = GraphConv(params.gconv1, params.gconv2)

            self.fc13 = nn.Linear(params.gconv2, h_feedforward)
            self.fc14 = nn.Linear(params.gconv2, h_feedforward)

        # Transformer
        if self.cfg.model.str:
            self.premodel = premodel
            self.hidden_fc1 = nn.Linear(self.d_premodel, h_feedforward)
            self.drpmem = nn.Dropout(dropout_p)

        self.ln = nn.LayerNorm(self.d_premodel)
        self.ln2 = nn.LayerNorm(h_feedforward)

        self.drp = nn.Dropout(dropout_p)
        self.loss_fn = nn.MSELoss()

        self.hidden_fc = nn.Linear(self.d_premodel, h_feedforward)
        self.predict_fc = nn.Linear(h_feedforward, 1)
        self.regr = nn.Sequential(self.hidden_fc, self.predict_fc)

        self._init_params()

    def graph_encode(self, g):
        h = g.ndata["node_feats"]
        h = self.gconv1(g, h)
        h = F.relu(h)
        h = self.gconv2(g, h)
        g_transformed = deepcopy(g)
        g_transformed.ndata["node_feats"] = F.relu(h)
        h = dgl.mean_nodes(g_transformed, "node_feats")
        return self.fc13(h)

    def str_encode(self, x):
        memory = self.premodel(x)
        # encoded = memory[1, :, :]
        encoded = torch.mean(memory, dim=0)
        return encoded

    def fusion_layer(self, str_encoded, graph_encoded):
        fused = self.fusion_mu(torch.cat((str_encoded, graph_encoded), dim=1))
        return fused

    def forward(self, batch):
        """Apply SMILES strings to model

        The dictionary returned will be passed to other functions, so its contents are fairly flexible,
        except that it must contain the key "token_output" which is the output of the model
        (possibly after any fully connected layers) for each token.

        Arg:
            batch (dict {
                "graphs": list of DGL graphs
                "masked_tokens": tensor of token_ids of shape (seq_len, batch_size),
                "pad_masks": bool tensor of padded elems of shape (seq_len, batch_size),
                "sentence_masks" (optional): long tensor (0 or 1) of shape (seq_len, batch_size)
            }):

        Returns:
            Output from model (dict containing key "token_output")
        """
        assert batch["graphs"] is not None or batch["masked_tokens"] is not None
        assert self.mode in ["regression", "reconstruction"]
        pred_recon = None
        pred_regr = None
        if self.cfg.model.str and not self.cfg.model.graph:
            encoded = self.str_encode(batch)

        elif not self.cfg.model.str and self.cfg.model.graph:
            encoded = self.graph_encode(batch["graphs"])

        elif self.cfg.model.str and self.cfg.model.graph:
            encoded_str = self.str_encode(batch)
            encoded_graph = self.graph_encode(batch["graphs"])
            encoded = self.fusion_layer(encoded_str, encoded_graph)
        else:
            raise Exception

        if self.mode == "reconstruction":
            model_output, token_output = self.str_decode(batch, encoded)
            pred_recon = {"model_output": model_output, "token_output": token_output}

        else:
            x = self.drp(encoded)
            x = self.ln(x)
            x = self.hidden_fc(x)

            x = F.relu(x)
            x = self.drp(x)
            x = self.ln2(x)
            pred_regr = self.predict_fc(x)
        return pred_regr, pred_recon

    def str_decode(self, x, memory_input):
        encoder_input_dim = x["encoder_input"].shape[0]
        decoder_input = x["decoder_input"]
        decoder_pad_mask = x["decoder_pad_mask"].transpose(0, 1)
        decoder_embs = self._construct_input(decoder_input)
        seq_len, _, _ = tuple(decoder_embs.size())
        memory_input = memory_input.view(
            1, memory_input.size(0), memory_input.size(-1)
        ).repeat(encoder_input_dim, 1, 1)
        encoder_pad_mask = x["encoder_pad_mask"].transpose(0, 1)
        tgt_mask = self._generate_square_subsequent_mask(seq_len, device="cuda")
        model_output = self.decoder(
            decoder_embs,
            memory_input,
            tgt_key_padding_mask=decoder_pad_mask,
            memory_key_padding_mask=encoder_pad_mask.clone(),
            tgt_mask=tgt_mask,
        )
        token_output = self.token_fc(model_output)
        token_probs = self.log_softmax(token_output)
        return token_output, token_probs

    def calc_reco_loss(self, batch_input, model_output):
        """Calculate the loss for the model

        Args:
            batch_input (dict): Input given to model,
            model_output (dict): Output from model

        Returns:
            loss (singleton tensor),
        """
        tokens = batch_input["target_reco"]
        tgt_mask = batch_input["target_mask"]
        token_output = model_output["token_output"]

        token_mask_loss = self._calc_mask_loss(token_output, tokens, tgt_mask)

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
        loss = self.reco_loss_fn(token_pred, target.reshape(-1)).reshape(
            (seq_len, batch_size)
        )

        inv_target_mask = ~(target_mask > 0)
        num_tokens = inv_target_mask.sum()
        loss = loss.sum() / num_tokens

        return loss

    def training_step(self, batch, batch_idx):
        self.train()

        model_output, pred_reco = self.forward(batch)
        if self.mode == "regression":
            loss = self.loss_fn(batch["target"], model_output)
        else:
            loss = self.calc_reco_loss(batch, pred_reco)
        self.log("train_loss", loss, on_step=True, logger=True)  # , prog_bar=True

        return loss

    def validation_step(self, batch, batch_idx):
        self.eval()
        model_output, pred_reco = self.forward(batch)

        if self.mode == "regression":
            batch_preds = model_output.squeeze(dim=1)  # .tolist()
            batch_targs = batch["target"].squeeze(dim=1)  # .tolist()
            loss = self.loss_fn(batch_targs, batch_preds)
        else:
            loss = self.calc_reco_loss(batch, pred_reco)

        self.log("val_loss", loss, on_epoch=True, on_step=False, prog_bar=True)

    def test_step(self, batch, batch_idx):
        self.eval()
        model_output, pred_reco = self.forward(batch)
        if self.mode == "regression":
            loss = self.loss_fn(batch["target"], model_output)
        else:
            loss = self.loss_reco(pred_reco, x.transpose(1, 2), mu, lv)
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
        token_embs.to("cuda")
        # Scaling the embeddings like this is done in other transformer libraries
        token_embs = token_embs * math.sqrt(self.d_premodel)

        positional_embs = self.pos_emb[:seq_len, :].unsqueeze(0).transpose(0, 1)
        embs = token_embs + positional_embs
        embs = self.drp(embs)
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

    def _generate_square_subsequent_mask(self, sz, device="cuda"):
        """
        Method from Pytorch transformer.
        Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).

        Args:
            sz (int): Size of mask to generate

        Returns:
            torch.Tensor: Square autoregressive mask for decode
        """

        mask = (torch.triu(torch.ones(sz, sz)).to(device) == 1).transpose(0, 1)
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

    def _decode_fn(self, token_ids, pad_mask, memory, mem_pad_mask):
        decode_input = {
            "decoder_input": token_ids,
            "decoder_pad_mask": pad_mask,
            "memory_input": memory,
            "memory_pad_mask": mem_pad_mask,
        }
        model_output = self.decode(decode_input)
        return model_output

    def __deepcopy__(self):
        return GraphTransformerModel(
            self.cfg,
            self.task,
            self.graph_dim,
            self.vocab_size,
            self.d_premodel,
            self.premodel,
            self.h_feedforward,
            self.lr,
            self.weight_decay,
            self.activation,
            self.num_steps,
            self.dropout_p,
            self.max_seq_len,
            self.batch_size,
            self.epochs,
            augment=None,
        )
