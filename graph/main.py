import argparse
import hydra
import numpy as np
import os
import pandas as pd
import sys
import torch
from loguru import logger
from pathlib import Path
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

import molbart.util as util
from graph.finetune_regression_modules import (
    RegPropDataModule,
    FineTuneTransformerModel,
    EncoderOfBARTModel,
)
from graph_transformer_model import GraphTransformerModel
from molbart.decoder import DecodeSampler


def load_premodel(cfg, graph_dim, args, vocab_size, total_steps, pad_token_idx, tokeniser):
    sampler = DecodeSampler(tokeniser, args.max_seq_len)
    premodel = EncoderOfBARTModel.load_from_checkpoint(
        args.model_path,
        decode_sampler=sampler,
        pad_token_idx=pad_token_idx,
        vocab_size=vocab_size,
        num_steps=total_steps,
        lr=cfg.regression.optim.lr,
        weight_decay=cfg.regression.optim.weight_decay,
        schedule=args.schedule,
        warm_up_steps=args.warm_up_steps,
        dropout=cfg.regression.dropout,
    )
    premodel.decoder = torch.nn.Identity()
    premodel.token_fc = torch.nn.Identity()
    premodel.loss_fn = torch.nn.Identity()
    premodel.log_softmax = torch.nn.Identity()
    premodel.to("cuda")
    return premodel

def load_model(cfg, mode, graph_dim, args, vocab_size, total_steps, pad_token_idx, tokeniser):
    premodel = load_premodel(cfg, graph_dim, args, vocab_size, total_steps, pad_token_idx, tokeniser)
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
        d_premodel=args.d_premodel,
        vocab_size=vocab_size,
        premodel=premodel,
        epochs=cfg.regression.epochs,
        batch_size=cfg.regression.batch_size,
        h_feedforward=cfg.layers.h_feedforward,
        lr=lr,
        weight_decay=weight_decay,
        activation="gelu",
        num_steps=total_steps,
        max_seq_len=args.max_seq_len,
        dropout_p=cfg.regression.dropout,
        augment=args.augment,
    )
    model.to("cuda")
    return model


def build_trainer(cfg, task, monitor="val_loss"):
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
        min_epochs = cfg.regression.epochs
        max_epochs = cfg.regression.epochs
        res_path = os.path.join(cfg.run.regr_dataset, res_dir, task)
        clip_grad = cfg.regression.clip_grad
        grad_batches = cfg.regression.acc_batches
    else:
        min_epochs = cfg.reconstruction.epochs
        max_epochs = cfg.reconstruction.epochs
        res_path = os.path.join(cfg.run.reco_dataset, res_dir, task)
        clip_grad = 1
        grad_batches = 1
    if not os.path.exists(res_path):
        os.makedirs(res_path, exist_ok=True)
    logger = TensorBoardLogger("tb_logs", name=res_path)
    lr_monitor = LearningRateMonitor(logging_interval="step")
    checkpoint_cb = ModelCheckpoint(
        monitor=monitor, save_last=True, save_top_k=1, mode="min"
    )
    trainer = Trainer(
        logger=logger,
        gpus=gpus,
        min_epochs=min_epochs,
        max_epochs=max_epochs,
        precision=precision,
        accumulate_grad_batches=grad_batches,
        gradient_clip_val=clip_grad,
        callbacks=[lr_monitor, checkpoint_cb],
        # progress_bar_refresh_rate=0,
        limit_val_batches=4,
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
        batch = {k: v.to(device=device, non_blocking=True) if hasattr(v, 'to') else v for k, v in batch.items()}
        batch_preds, pred_reco, mu, lv = model(batch)
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
    model.to("cuda")
    dm1 = RegPropDataModule(
        cfg=cfg,
        sample_size=None,
        sample_type="regr",
        task=task,
        datapath=Path("."),
        tokeniser=tokeniser,
        batch_size=cfg.regression.batch_size,
        max_seq_len=300,
        augment=False,
        forward_pred=True,
        num_buckets=24,
    )
    dm1.setup()

    pred_train, y_train = get_targs_preds(model=model, dl=dm1.train_dataloader())
    R2_train = np.power(np.corrcoef(pred_train, y_train)[0, 1], 2)
    RMSE_train = np.sqrt(np.mean((np.array(pred_train) - np.array(y_train)) ** 2))
    torch.no_grad()
    model.eval()
    pred_test, y_test = get_targs_preds(model=model, dl=dm1.test_dataloader())
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
    return results


def save_results(cfg, model, metrics, task):
    assert task in ["regression", "reconstruction"]
    res_dir = (
        "model"
        + ("_str" if cfg.model.str else "")
        + ("_graph" if cfg.model.graph else "")
    )
    res_path = os.path.join(
        "/data/user/mikhaillebedev/chemfusion/",
        "results",
        cfg.run.regr_dataset,
        res_dir,
        task,
    )
    if not os.path.exists(res_path):
        os.makedirs(res_path, exist_ok=True)
    torch.save(model, os.path.join(res_path, "model.pth"))
    if metrics is not None:
        metrics.to_csv(os.path.join(res_path, "metrics.csv"), index=False)
        # json.dump(metrics, open(os.path.join(res_path, "metrics.json"), "w"))
    # OmegaConf.save(cfg.model, f=open(os.path.join(res_path, "params.yaml"), "w"))


@hydra.main(config_name="config")
def main(cfg):
    global args
    tokeniser = util.load_tokeniser(
        "//graph/prop_bart_vocab.txt",
        272,
    )

    print("Finished tokeniser.")

    model = None
    task = None
    task_name = None
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

    res_dir = (
        "model"
        + ("_str" if cfg.model.str else "")
        + ("_graph" if cfg.model.graph else "")
    )

    vocab_size = len(tokeniser)


    pad_token_idx = tokeniser.vocab[tokeniser.pad_token]

    if task_name == "Regression following reconstruction pretraining":
        print("Pretraining the model:")
        dm_reco = RegPropDataModule(
            cfg=cfg,
            sample_size=None,
            sample_type="regr",
            task="reconstruction",
            datapath=Path("."),
            tokeniser=tokeniser,
            batch_size=cfg.reconstruction.batch_size,
            max_seq_len=300,
            augment=False,
            num_buckets=24,
        )
        train_steps = util.calc_train_steps(cfg, dm_reco)
        print(f"Train steps: {train_steps}")
        model = load_model(
            cfg,
            "reconstruction",
            dm_reco.train_dataset.graph_dims["nodes"],
            args,
            vocab_size,
            train_steps + 1,
            pad_token_idx,
            tokeniser,
        )
        trainer_reco = build_trainer(cfg, "reconstruction", "val_loss")
        trainer_reco.fit(model, dm_reco)
        print("Finished pretraining the model.")
        best_model_name_reco = [
            i
            for i in os.listdir(os.path.join(trainer_reco.log_dir, "checkpoints"))
            if i.startswith("epoch")
        ][0]
        max_version_reco = max(
            [
                int(i.split("_")[1])
                for i in os.listdir(
                os.path.join(
                    "tb_logs", os.path.join(cfg.run.reco_dataset, res_dir, "reconstruction")
                )
            )
            ]
        )
        premodel = load_premodel(
            cfg,
            dm_reco.train_dataset.graph_dims["nodes"],
            args,
            vocab_size,
            train_steps + 1,
            pad_token_idx,
            tokeniser,
        )


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
            max_seq_len=300,
            augment=False,
            num_buckets=24,
        )
        # print(dm)
        print("Finished datamodule.")
        train_steps = util.calc_train_steps(cfg, dm)
        print(f"Train steps: {train_steps}")

        print("Loading model...")
        premodel = load_premodel(
            cfg,
            dm.train_dataset.graph_dims["nodes"],
            args,
            vocab_size,
            train_steps + 1,
            pad_token_idx,
            tokeniser,
        )
        if task_name == "Vanilla regression":
            model = load_model(
                cfg,
                "regression",
                dm.train_dataset.graph_dims["nodes"],
                args,
                vocab_size,
                train_steps + 1,
                pad_token_idx,
                tokeniser,
            )
        else:
            model = model.load_from_checkpoint(
                os.path.join(
                    "tb_logs",
                    os.path.join(cfg.run.reco_dataset, res_dir, "reconstruction"),
                    f"version_{str(max_version_reco)}/checkpoints/{best_model_name_reco}",
                ),
                cfg=cfg,
                mode="regression",
                graph_dim=dm.train_dataset.graph_dims["nodes"],
                vocab_size=vocab_size,
                d_premodel=args.d_premodel,
                premodel=premodel,
                h_feedforward=cfg.layers.h_feedforward,
                lr=cfg.regression.optim.lr,
                weight_decay=cfg.regression.optim.weight_decay,
                activation="gelu",
                num_steps=train_steps + 1,
                dropout_p=cfg.regression.dropout,
                max_seq_len=args.max_seq_len,
                batch_size=cfg.regression.batch_size,
                epochs=cfg.regression.epochs,
                augment=args.augment
            )
        # pretraining was here

        print("Finished model.")
        # print(model)
        print("Building trainer...")
        trainer = build_trainer(cfg, "regression")
        print("Finished trainer.")

        print("Fitting data module to trainer")
        trainer.fit(model, dm)
        print("path logdir ", os.path.join(trainer.log_dir, "checkpoints"))
        print("log dir", os.listdir(os.path.join(trainer.log_dir, "checkpoints")))
        best_model_name = [
            i
            for i in os.listdir(os.path.join(trainer.log_dir, "checkpoints"))
            if i.startswith("epoch")
        ][0]

        max_version = max(
            [
                int(i.split("_")[1])
                for i in os.listdir(
                    os.path.join(
                        "tb_logs", os.path.join(cfg.run.regr_dataset, res_dir, task)
                    )
                )
            ]
        )
        print(
            os.path.join(
                "tb_logs",
                os.path.join(cfg.run.regr_dataset, res_dir, task),
                f"version_0/checkpoints/{best_model_name}",
            )
        )
        print("path to model is ", os.path.join(
                "tb_logs",
                os.path.join(cfg.run.regr_dataset, res_dir, task),
                f"version_{str(max_version)}/checkpoints/{best_model_name}",
            ))
        # model = load_model(
        #     cfg,
        #     "regression",
        #     dm.train_dataset.graph_dims["nodes"],
        #     args,
        #     vocab_size,
        #     train_steps + 1,
        #     pad_token_idx,
        #     tokeniser,
        # )
        model = model.load_from_checkpoint(
            os.path.join(
                "tb_logs",
                os.path.join(cfg.run.regr_dataset, res_dir, task),
                f"version_{str(max_version)}/checkpoints/{best_model_name}",
            ),
            cfg=cfg,
            mode="regression",
            graph_dim=dm.train_dataset.graph_dims["nodes"],
            vocab_size=vocab_size,
            d_premodel=args.d_premodel,
            premodel=premodel,
            h_feedforward=cfg.layers.h_feedforward,
            lr=cfg.regression.optim.lr,
            weight_decay=cfg.regression.optim.weight_decay,
            activation="gelu",
            num_steps=train_steps + 1,
            dropout_p=cfg.regression.dropout,
            max_seq_len=args.max_seq_len,
            batch_size=cfg.regression.batch_size,
            epochs=cfg.regression.epochs,
            augment=args.augment
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
    # save_results(cfg, model, metrics=metrics, task=task)
    logger.info(f"{cfg.run.regr_dataset} | {task_name} is DONE!\n\n")
    # save_path = "results_predictions.csv"
    # print(results)
    # results.to_csv(save_path, index=False)


if __name__ == "__main__":
    DEFAULT_vocab_path = "prop_bart_vocab.txt"
    DEFAULT_model_path = (
        "/data/user/mikhaillebedev/chemfusion/combined/step1000000.ckpt"
    )
    DEFAULT_SCHEDULE = "cycle"
    DEFAULT_AUGMENT = True
    DEFAULT_WARM_UP_STEPS = 300
    DEFAULT_TRAIN_TOKENS = None
    DEFAULT_NUM_BUCKETS = 24
    DEFAULT_LIMIT_VAL_BATCHES = 1.0
    DEFAULT_D_PREMODEL = 512
    DEFAULT_MAX_SEQ_LEN = 300
    DEFAULT_H_FEEDFORWARD = 2048
    DEFAULT_drp = 0.2
    DEFAULT_Hdrp = 0.4
    parser = argparse.ArgumentParser()

    parser.add_argument("--vocab_path", type=str, default=DEFAULT_vocab_path)
    parser.add_argument("--model_path", type=str, default=DEFAULT_model_path)
    parser.add_argument("--d_premodel", type=int, default=DEFAULT_D_PREMODEL)
    parser.add_argument("--h_feedforward", type=int, default=DEFAULT_H_FEEDFORWARD)
    parser.add_argument("--drp", type=int, default=DEFAULT_drp)
    parser.add_argument("--Hdrp", type=int, default=DEFAULT_Hdrp)
    parser.add_argument("--max_seq_len", type=int, default=DEFAULT_MAX_SEQ_LEN)
    parser.add_argument(
        "--chem_token_start_idx", type=int, default=util.DEFAULT_CHEM_TOKEN_START
    )
    parser.add_argument("--schedule", type=str, default=DEFAULT_SCHEDULE)
    parser.add_argument("--augment", type=str, default=DEFAULT_AUGMENT)
    parser.add_argument("--warm_up_steps", type=int, default=DEFAULT_WARM_UP_STEPS)
    parser.add_argument("--train_tokens", type=int, default=DEFAULT_TRAIN_TOKENS)
    parser.add_argument("--num_buckets", type=int, default=DEFAULT_NUM_BUCKETS)
    parser.add_argument(
        "--limit_val_batches", type=float, default=DEFAULT_LIMIT_VAL_BATCHES
    )

    args = parser.parse_args()
    main()
