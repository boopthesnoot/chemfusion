import argparse
import copy

import hydra
import numpy as np
import os
import pandas as pd
import sys
import torch
from loguru import logger
from pathlib import Path
from omegaconf import OmegaConf

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    StochasticWeightAveraging,
)
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

sys.path.append(os.path.abspath(".."))
import molbart.util as util
from graph.finetune_regression_modules import (
    RegPropDataModule,
    FineTuneTransformerModel,
    EncoderOfBARTModel,
)
from graph_transformer_model import GraphTransformerModel
from molbart.decoder import DecodeSampler


def load_premodel(
    cfg, model_path, args, vocab_size, total_steps, pad_token_idx, tokeniser
):
    sampler = DecodeSampler(tokeniser, cfg.layers.max_seq_len)
    print("model_path", model_path)
    premodel = EncoderOfBARTModel.load_from_checkpoint(
        model_path,
        decode_sampler=sampler,
        pad_token_idx=pad_token_idx,
        vocab_size=vocab_size,
        d_model=cfg.layers.d_premodel,
        num_layers=cfg.layers.num_layers,
        num_heads=cfg.layers.num_heads,
        d_feedforward=cfg.layers.d_feedforward,
        lr=cfg.regression.optim.lr,
        weight_decay=cfg.regression.optim.weight_decay,
        activation="relu",
        num_steps=total_steps,
        max_seq_len=cfg.layers.max_seq_len,
        schedule=args.schedule,
        warm_up_steps=args.warm_up_steps,
        dropout=cfg.regression.dropout,
    )
    premodel.decoder = torch.nn.Identity()
    premodel.token_fc = torch.nn.Identity()
    premodel.loss_fn = torch.nn.Identity()
    premodel.log_softmax = torch.nn.Identity()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    premodel.to(device)
    return premodel


def load_model(
    cfg,
    mode,
    graph_dim,
    args,
    vocab_size,
    total_steps,
    pad_token_idx,
    tokeniser,
    premodel=None,
):
    if not premodel:
        premodel = load_premodel(
            cfg,
            args.model_path,
            args,
            vocab_size,
            total_steps,
            pad_token_idx,
            tokeniser,
        )
    if mode == "regression":
        lr = cfg.regression.optim.lr
        weight_decay = cfg.regression.optim.weight_decay
    else:
        lr = cfg.reconstruction.optim.lr
        weight_decay = cfg.reconstruction.optim.weight_decay

    model = GraphTransformerModel(
        cfg=cfg,
        mode=mode,
        graph_dim=graph_dim,
        d_premodel=cfg.layers.d_premodel,
        vocab_size=vocab_size,
        premodel=premodel,
        epochs=cfg.regression.epochs,
        batch_size=cfg.regression.batch_size,
        h_feedforward=cfg.layers.h_feedforward,
        lr=lr,
        weight_decay=weight_decay,
        activation="gelu",
        num_steps=total_steps,
        max_seq_len=cfg.layers.max_seq_len,
        dropout_p=cfg.regression.dropout,
        augment=cfg.reconstruction.augment,
        tokeniser=tokeniser,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model


def build_trainer(cfg, task, monitor="val_loss", model_save_path=None):
    """
    build the Trainer using pytorch_lightning
    """
    gpus = cfg.general.gpus
    precision = 32  # 16
    assert task in ["regression", "reconstruction"]
    res_dir = (
        "model"
        + ("_str" if cfg.model.str else "")
        + ("_graph" if cfg.model.graph else "")
    )
    if task == "regression":
        min_epochs = int(cfg.regression.epochs * 0.7)
        max_epochs = cfg.regression.epochs
        # res_path = os.path.join(cfg.run.regr_dataset, res_dir, task)
        clip_grad = cfg.regression.clip_grad
        grad_batches = cfg.regression.acc_batches
    else:
        min_epochs = int(cfg.reconstruction.epochs * 0.7)
        max_epochs = cfg.reconstruction.epochs
        # res_path = os.path.join(cfg.run.reco_dataset, res_dir, task)
        clip_grad = 1
        grad_batches = 1
    # if not os.path.exists(res_path):
    #     os.makedirs(res_path, exist_ok=True)
    # logger = TensorBoardLogger("tb_logs", name=res_path)
    # logger = WandbLogger(**cfg)
    logger = None
    if not logger:
        logger = WandbLogger(project="chemfusion", save_dir=os.getcwd())
        logger.experiment.config.update(cfg, allow_val_change=True)
    lr_monitor = LearningRateMonitor(logging_interval="step")
    if not model_save_path:
        model_save_path = os.path.join(os.getcwd(), cfg.run.regr_dataset, res_dir, task)
    # if run_number:
    #     dirpath = os.path.join(dirpath, f"run_{run_number}")
    #     Path(dirpath).mkdir(parents=True, exist_ok=True)
    checkpoint_cb = ModelCheckpoint(
        dirpath=model_save_path,
        monitor=monitor,
        save_last=True,
        save_top_k=1,
        mode="min",
    )
    # swa = StochasticWeightAveraging(swa_lrs=[3e-4])
    # early_stop_callback = EarlyStopping(
    #     monitor=monitor, min_delta=0.00, patience=20, verbose=False, mode="min"
    # )
    trainer = Trainer(
        logger=logger,
        # gpus=gpus,
        min_epochs=min_epochs,
        max_epochs=max_epochs,
        precision=precision,
        accumulate_grad_batches=grad_batches,
        gradient_clip_val=clip_grad,
        # auto_lr_find=True,
        callbacks=[lr_monitor, checkpoint_cb],
        # progress_bar_refresh_rate=0,
        devices=1,
        accelerator="gpu",
        limit_val_batches=4,
        auto_select_gpus=True,
    )

    return trainer


def get_targs_preds(model, dl):
    """
    get the prediction and the targets
    Args: model, dataloader
    Returns: two lists with prediction and the targets
    """
    preds = []
    targs = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    for i, batch in enumerate(iter(dl)):
        batch = {
            k: v.to(device=device, non_blocking=True) if hasattr(v, "to") else v
            for k, v in batch.items()
        }
        batch_preds, pred_reco = model(batch)
        batch_preds = batch_preds.squeeze(dim=1).tolist()
        batch_targs = batch["target"].squeeze(dim=1).tolist()
        preds.extend(batch_preds)
        targs.extend(batch_targs)
    return preds, targs


def get_metrics(cfg, tokeniser, model, task):
    """
    compute the RMSE and R^2 for each gene symbol separately and save them to a .csv file
    Args: data_path, tokeniser, model
    Returns: two lists with prediction and the targets
    """
    results = None
    # model.to("cuda")
    dm = RegPropDataModule(
        cfg=cfg,
        sample_size=None,
        sample_type="regr",
        task=task,
        datapath=Path("."),
        tokeniser=tokeniser,
        batch_size=cfg.regression.batch_size,
        max_seq_len=cfg.layers.max_seq_len,
        augment=False,
        forward_pred=True,
        num_buckets=cfg.general.num_buckets,
    )
    dm.setup()

    assert task in ["regression", "reconstruction"]
    if task == "regression":
        pred_train, y_train = get_targs_preds(model=model, dl=dm.train_dataloader())
        torch.no_grad()
        model.eval()
        pred_test, y_test = get_targs_preds(model=model, dl=dm.test_dataloader())
        R2_train = np.power(np.corrcoef(pred_train, y_train)[0, 1], 2)
        RMSE_train = np.sqrt(np.mean((np.array(pred_train) - np.array(y_train)) ** 2))
        R2_test = np.power(np.corrcoef(pred_test, y_test)[0, 1], 2)
        RMSE_test = np.sqrt(np.mean((np.array(pred_test) - np.array(y_test)) ** 2))
        results = pd.DataFrame(
            {
                "RMSE_train": [RMSE_train],
                "R2_train": [R2_train],
                "RMSE_test": [RMSE_test],
                "R2_test": [R2_test],
            }
        )
    if task == "reconstruction":
        test_loader = dm.test_dataloader()
        model.eval()

        smiles = []
        log_lhs = []
        original_smiles = []
        device = "cpu"
        for b_idx, batch in enumerate(test_loader):
            device_batch = {
                key: val.to(device) if type(val) == torch.Tensor else val
                for key, val in batch.items()
            }
            with torch.no_grad():
                smiles_batch, log_lhs_batch = model.sample_molecules(
                    device_batch, sampling_alg="greedy"
                )

            smiles.extend(smiles_batch)
            log_lhs.extend(log_lhs_batch)
            original_smiles.extend(batch["target_smiles"])
            break
        results = pd.DataFrame(
            {
                "smiles": original_smiles,
                "reconstructed smiles": smiles,
            }
        )
    return results


def get_path(cfg, task):
    res_dir = (
        "model"
        + ("_str" if cfg.model.str else "")
        + ("_graph" if cfg.model.graph else "")
    )
    res_path = os.path.join(
        os.getcwd(),
        cfg.run.regr_dataset,
        res_dir,
        task,
    )
    if not os.path.exists(res_path):
        os.makedirs(res_path, exist_ok=True)
    return res_path


def save_results(cfg, model, metrics, task, trainer=None, finetune=False):
    assert task in ["regression", "reconstruction"]
    model_prefix = None
    metrics_name = "metrics.csv"
    res_path = get_path(cfg, task)
    print("res_path", res_path)
    if task == "regression":
        model_prefix = "regr_model"
    if task == "reconstruction":
        model_prefix = "reconstr_model"
    if finetune:
        model_prefix += "_finetuned"
        metrics_name = "finetuned_" + metrics_name

    assert model_prefix

    if trainer:
        trainer.save_checkpoint(os.path.join(res_path, f"{model_prefix}.ckpt"))
    # torch.save(model, os.path.join(res_path, f"{model_prefix}.pth"))
    if metrics is not None:
        metrics.to_csv(os.path.join(res_path, metrics_name), index=False)
    OmegaConf.save(cfg, f=open(os.path.join(res_path, "params.yaml"), "w"))
    return res_path


def get_dm(cfg, tokeniser, task):
    if task == "reconstruction":
        batch_size = cfg.reconstruction.batch_size
    else:
        batch_size = cfg.regression.batch_size
    dm = RegPropDataModule(
        cfg=cfg,
        sample_size=None,
        sample_type="regr",
        task=task,
        datapath=Path("."),
        tokeniser=tokeniser,
        batch_size=batch_size,
        max_seq_len=cfg.layers.max_seq_len,
        augment=cfg.reconstruction.augment,
        num_buckets=cfg.general.num_buckets,
    )
    return dm


@hydra.main(config_name="config")
def main(cfg):

    util.seed_everything(cfg.general.fixed_random_seed)
    global args
    premodel_path = args.model_path
    model_path = None
    tokeniser = util.load_tokeniser(
        "../../../prop_bart_vocab.txt",
        272,
    )

    print("Finished tokeniser.")
    premodel = None
    model = None
    task = None
    task_name = None
    trainer = None
    premodel_pretrained = None
    metrics = []
    if cfg.run.regr_vanilla:
        task_name = "Vanilla regression"
        task = "regression"
    if cfg.run.reco:
        task_name = "Reconstruction model training"
        task = "reconstruction"
    if cfg.run.regr_with_reco:
        task_name = "Regression following reconstruction pretraining"
        task = "regression"
    assert task in ["regression", "reconstruction"]

    model_mode = (
        "model"
        + ("_str" if cfg.model.str else "")
        + ("_graph" if cfg.model.graph else "")
    )
    logger.info(f"{cfg.run.regr_dataset} | {task_name} | Mode {model_mode}")
    vocab_size = len(tokeniser)
    pad_token_idx = tokeniser.vocab[tokeniser.pad_token]
    # premodel = load_premodel(
    #         cfg,
    #         args.model_path,
    #         args,
    #         vocab_size,
    #         total_steps,
    #         pad_token_idx,
    #         tokeniser,
    #     )

    if task_name == "Regression following reconstruction pretraining":
        dm_reco = get_dm(cfg, tokeniser, "reconstruction")
        # pad_token_idx = tokeniser.vocab[tokeniser.pad_token]
        # vocab_size = len(tokeniser)
        train_steps = util.calc_train_steps(cfg, dm_reco)
        premodel_pretrained = load_premodel(
            cfg,
            premodel_path,
            args,
            vocab_size,
            train_steps + 1,
            pad_token_idx,
            tokeniser,
        )
        model_pretrained = load_model(
            cfg,
            "reconstruction",
            dm_reco.train_dataset.graph_dims["nodes"],
            args,
            vocab_size,
            train_steps + 1,
            pad_token_idx,
            tokeniser,
            premodel=premodel_pretrained,
        )
        model_save_path = os.path.join(
            os.getcwd(), cfg.run.regr_dataset, model_mode, "reconstruction"
        )
        Path(model_save_path).mkdir(parents=True, exist_ok=True)
        trainer_reco = build_trainer(cfg, "reconstruction", "val_loss", model_save_path)
        trainer_reco.fit(model_pretrained, dm_reco)
        reconstruction_metrics = get_metrics(
            cfg, tokeniser, model_pretrained, "reconstruction"
        )
        model_dir = save_results(
            cfg,
            model_pretrained,
            reconstruction_metrics,
            "reconstruction",
            trainer_reco,
        )
        model_path = os.path.join(model_dir, "reconstr_model.ckpt")

        # fine-tuning weights while reconstructing the regression dataset
        cfg.run.reco_dataset = cfg.run.regr_dataset
        # cfg.reconstruction.epochs = 20
        dm_reco_regr = get_dm(cfg, tokeniser, "reconstruction")
        train_steps_reco_regr = util.calc_train_steps(cfg, dm_reco_regr)
        model_reco_regr = GraphTransformerModel.load_from_checkpoint(
            model_path,
            cfg=cfg,
            mode="regression",
            graph_dim=dm_reco_regr.train_dataset.graph_dims["nodes"],
            vocab_size=vocab_size,
            d_premodel=cfg.layers.d_premodel,
            premodel=premodel_pretrained,
            h_feedforward=cfg.layers.h_feedforward,
            lr=cfg.reconstruction.optim.lr,
            weight_decay=cfg.reconstruction.optim.weight_decay,
            activation="gelu",
            num_steps=train_steps_reco_regr + 1,
            dropout_p=0,
            max_seq_len=cfg.layers.max_seq_len,
            batch_size=cfg.reconstruction.batch_size,
            epochs=cfg.reconstruction.epochs,
            augment=cfg.reconstruction.augment,
            tokeniser=tokeniser,
        )
        trainer_reco_regr = build_trainer(cfg, "reconstruction", "val_loss")
        trainer_reco_regr.fit(model_reco_regr, dm_reco_regr)
        reconstruction_metrics = get_metrics(
            cfg, tokeniser, model_reco_regr, "reconstruction"
        )
        model_dir = save_results(
            cfg,
            model_reco_regr,
            reconstruction_metrics,
            "reconstruction",
            trainer_reco_regr,
            finetune=True,
        )
        # pretraining_dir = pretrain(
        #     cfg, pretraining_dir, args, tokeniser, "reconstruction", finetune=True
        # )
        # premodel_path = (
        #     args.model_path
        # )  # os.path.join(pretraining_dir, "reconstr_model.pth")
        model_path = os.path.join(model_dir, "reconstr_model_finetuned.ckpt")
        task = "regression"
    for run_number in range(1, cfg.run.n_runs + 1):
        logger.info(
            f"{cfg.run.regr_dataset} | {task_name} | Run {run_number}/{cfg.run.n_runs}"
        )
        print("Building data module...")
        dm = RegPropDataModule(
            cfg=cfg,
            sample_size=None,
            sample_type="regr",
            task=task,
            datapath=Path("."),
            tokeniser=tokeniser,
            batch_size=cfg.regression.batch_size,
            max_seq_len=cfg.layers.max_seq_len,
            augment=cfg.reconstruction.augment,
            num_buckets=cfg.general.num_buckets,
        )
        print("Finished datamodule.")
        train_steps = util.calc_train_steps(cfg, dm)
        print(f"Train steps: {train_steps}")

        print("Loading model...")
        if not premodel:
            premodel = load_premodel(
                cfg,
                premodel_path,
                args,
                vocab_size,
                train_steps + 1,
                pad_token_idx,
                tokeniser,
            )

        if cfg.run.regr_vanilla:
            model = load_model(
                cfg,
                "regression",
                dm.train_dataset.graph_dims["nodes"],
                args,
                vocab_size,
                train_steps + 1,
                pad_token_idx,
                tokeniser,
                premodel=premodel,
            )
        else:
            premodel = copy.deepcopy(premodel_pretrained)
            # model = copy.deepcopy(model_pretrained)
            model = GraphTransformerModel.load_from_checkpoint(
                model_path,
                cfg=cfg,
                mode="regression",
                graph_dim=dm.train_dataset.graph_dims["nodes"],
                vocab_size=vocab_size,
                d_premodel=cfg.layers.d_premodel,
                premodel=premodel,
                h_feedforward=cfg.layers.h_feedforward,
                lr=cfg.regression.optim.lr,
                weight_decay=cfg.regression.optim.weight_decay,
                activation="gelu",
                num_steps=train_steps + 1,
                dropout_p=cfg.regression.dropout,
                max_seq_len=cfg.layers.max_seq_len,
                batch_size=cfg.regression.batch_size,
                epochs=cfg.regression.epochs,
                augment=cfg.reconstruction.augment,
                tokeniser=tokeniser,
            )
        print("Finished model.")

        print("Building trainer...")

        model_save_path = os.path.join(
            os.getcwd(), cfg.run.regr_dataset, model_mode, task, f"run_{run_number}"
        )
        Path(model_save_path).mkdir(parents=True, exist_ok=True)
        trainer = build_trainer(cfg, "regression", model_save_path=model_save_path)
        print("Finished trainer.")

        print("Fitting data module to trainer")
        trainer.fit(model, dm)
        best_model_name = [
            i for i in os.listdir(model_save_path) if i.startswith("epoch")
        ][0]

        # max_version = max([int(i.split("_")[1]) for i in os.listdir(checkpoints_dir)])
        # print(
        #     "path to model is ",
        #     os.path.join(
        #         checkpoints_dir,
        #         best_model_name,
        #     ),
        # )
        model = model.load_from_checkpoint(
            os.path.join(
                model_save_path,
                best_model_name,
            ),
            cfg=cfg,
            mode="regression",
            graph_dim=dm.train_dataset.graph_dims["nodes"],
            vocab_size=vocab_size,
            d_premodel=cfg.layers.d_premodel,
            premodel=premodel,
            h_feedforward=cfg.layers.h_feedforward,
            lr=cfg.regression.optim.lr,
            weight_decay=cfg.regression.optim.weight_decay,
            activation="gelu",
            num_steps=train_steps + 1,
            dropout_p=cfg.regression.dropout,
            max_seq_len=cfg.layers.max_seq_len,
            batch_size=cfg.regression.batch_size,
            epochs=cfg.regression.epochs,
            augment=cfg.reconstruction.augment,
        )
        print("Finished training.")

        metric = get_metrics(cfg, tokeniser, model, task)
        print(metric)
        metrics.append(metric)
        logger.info(f"Run {run_number} is DONE!\n")

    metrics = pd.concat(metrics)
    print(f"RMSE Mean (test) {np.mean(metrics['RMSE_test']):3.3f}")
    print(f"RMSE Std (test) {np.std(metrics['RMSE_test']):3.3f}")
    print(
        f"RMSE (test) {np.mean(metrics['RMSE_test']):3.3f} +- {np.std(metrics['RMSE_test']):2.3f}"
    )
    print(
        f"R2 (test) {np.mean(metrics['R2_test']):2.3f} +- {np.std(metrics['R2_test']):2.3f}"
    )
    save_results(cfg, model, metrics=metrics, task=task, trainer=trainer)
    logger.info(f"{cfg.run.regr_dataset} | {task_name} is DONE!\n\n")


if __name__ == "__main__":
    DEFAULT_vocab_path = "prop_bart_vocab.txt"
    DEFAULT_model_path = (
        "/data/user/mikhaillebedev/chemfusion/combined/step1000000.ckpt"
    )
    DEFAULT_SCHEDULE = "transformer"
    DEFAULT_WARM_UP_STEPS = 3000
    DEFAULT_TRAIN_TOKENS = None
    DEFAULT_LIMIT_VAL_BATCHES = 1.0
    DEFAULT_drp = 0.2
    DEFAULT_Hdrp = 0.4
    parser = argparse.ArgumentParser()

    parser.add_argument("--vocab_path", type=str, default=DEFAULT_vocab_path)
    parser.add_argument("--model_path", type=str, default=DEFAULT_model_path)
    parser.add_argument("--drp", type=int, default=DEFAULT_drp)
    parser.add_argument("--Hdrp", type=int, default=DEFAULT_Hdrp)
    parser.add_argument(
        "--chem_token_start_idx", type=int, default=util.DEFAULT_CHEM_TOKEN_START
    )
    parser.add_argument("--schedule", type=str, default=DEFAULT_SCHEDULE)
    parser.add_argument("--warm_up_steps", type=int, default=DEFAULT_WARM_UP_STEPS)
    parser.add_argument("--train_tokens", type=int, default=DEFAULT_TRAIN_TOKENS)
    parser.add_argument(
        "--limit_val_batches", type=float, default=DEFAULT_LIMIT_VAL_BATCHES
    )
    args = parser.parse_args()
    main()
