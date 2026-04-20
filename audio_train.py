###
# Author: Kai Li
# Date: 2022-04-06 14:51:43
# Email: lk21@mails.tsinghua.edu.cn
# LastEditTime: 2022-06-05 14:51:15
###
import os
import sys
import yaml
import torch
from torch import Tensor
import argparse
import json
import look2hear.datas
import look2hear.models
import look2hear.system
import look2hear.losses
import look2hear.metrics
import look2hear.utils
from look2hear.system import make_optimizer
from dataclasses import dataclass
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, TQDMProgressBar
from pytorch_lightning.strategies.ddp import DDPStrategy
from rich import print, reconfigure
from collections.abc import MutableMapping
from look2hear.utils import print_only

import warnings

warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision("medium")  # tận dụng Tensor Core trên A100

parser = argparse.ArgumentParser()
parser.add_argument(
    "--conf_dir",
    default="local/conf.yml",
    help="Path to config yml file",
)
parser.add_argument(
    "--checkpoint_dir",
    default=None,
    help="Directory to save checkpoints (e.g. /content/drive/MyDrive/SPMamba_ckpt)",
)
parser.add_argument(
    "--resume",
    action="store_true",
    help="Resume training from last.ckpt in checkpoint_dir",
)

def main(config):
    print_only(
        "Instantiating datamodule <{}>".format(config["datamodule"]["data_name"])
    )
    datamodule: object = getattr(look2hear.datas, config["datamodule"]["data_name"])(
        **config["datamodule"]["data_config"]
    )
    datamodule.setup()

    train_loader, val_loader, test_loader = datamodule.make_loader
    
    # Define model and optimizer
    print_only(
        "Instantiating AudioNet <{}>".format(config["audionet"]["audionet_name"])
    )
    model = getattr(look2hear.models, config["audionet"]["audionet_name"])(
        sample_rate=config["datamodule"]["data_config"]["sample_rate"],
        **config["audionet"]["audionet_config"],
    )
    # import pdb; pdb.set_trace()
    print_only("Instantiating Optimizer <{}>".format(config["optimizer"]["optim_name"]))
    optimizer = make_optimizer(model.parameters(), **config["optimizer"])

    # Define scheduler
    scheduler = None
    if config["scheduler"]["sche_name"]:
        print_only(
            "Instantiating Scheduler <{}>".format(config["scheduler"]["sche_name"])
        )
        if config["scheduler"]["sche_name"] != "DPTNetScheduler":
            scheduler = getattr(torch.optim.lr_scheduler, config["scheduler"]["sche_name"])(
                optimizer=optimizer, **config["scheduler"]["sche_config"]
            )
        else:
            scheduler = {
                "scheduler": getattr(look2hear.system.schedulers, config["scheduler"]["sche_name"])(
                    optimizer, len(train_loader) // config["datamodule"]["data_config"]["batch_size"], 64
                ),
                "interval": "step",
            }

    # Set experiment directory: use --checkpoint_dir if provided, else default
    if config["main_args"].get("checkpoint_dir"):
        exp_dir = config["main_args"]["checkpoint_dir"]
    else:
        exp_dir = os.path.join(
            os.getcwd(), "Experiments", "checkpoint", config["exp"]["exp_name"]
        )
    config["main_args"]["exp_dir"] = exp_dir
    import time
    os.makedirs(exp_dir, exist_ok=True)
    time.sleep(2)  # Đợi Drive đồng bộ
    print_only(f"[Checkpoint Dir] {exp_dir}")
    print_only(f"[Checkpoint Dir exists] {os.path.isdir(exp_dir)}")
    conf_path = os.path.join(exp_dir, "conf.yml")
    with open(conf_path, "w") as outfile:
        yaml.safe_dump(config, outfile)

    # Define Loss function.
    print_only(
        "Instantiating Loss, Train <{}>, Val <{}>".format(
            config["loss"]["train"]["sdr_type"], config["loss"]["val"]["sdr_type"]
        )
    )
    loss_func = {
        "train": getattr(look2hear.losses, config["loss"]["train"]["loss_func"])(
            getattr(look2hear.losses, config["loss"]["train"]["sdr_type"]),
            **config["loss"]["train"]["config"],
        ),
        "val": getattr(look2hear.losses, config["loss"]["val"]["loss_func"])(
            getattr(look2hear.losses, config["loss"]["val"]["sdr_type"]),
            **config["loss"]["val"]["config"],
        ),
    }

    print_only("Instantiating System <{}>".format(config["training"]["system"]))
    system = getattr(look2hear.system, config["training"]["system"])(
        audio_model=model,
        loss_func=loss_func,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        scheduler=scheduler,
        config=config,
    )

    # Define callbacks
    print_only("Instantiating ModelCheckpoint")
    callbacks = []
    checkpoint_dir = os.path.join(exp_dir)

    # Bước 1: Tìm resume checkpoint TRƯỚC khi cleanup
    import glob
    ckpt_path = None
    if config["main_args"].get("resume"):
        last_ckpts = glob.glob(os.path.join(exp_dir, "last*.ckpt"))
        if last_ckpts:
            ckpt_path = max(last_ckpts, key=os.path.getmtime)
            print_only(f"Resuming from checkpoint: {ckpt_path}")
        else:
            print_only("[Warning] --resume set but no last*.ckpt found. Starting fresh.")

    # Bước 2: Xóa last*.ckpt cũ để Lightning tạo fresh last.ckpt (không bị -v1, -v2)
    # Nhưng GIỮ LẠI file đang dùng để resume (nếu có)
    for old_last in glob.glob(os.path.join(checkpoint_dir, "last*.ckpt")):
        if ckpt_path and os.path.abspath(old_last) == os.path.abspath(ckpt_path):
            print_only(f"[Cleanup] Skipped (resume target): {old_last}")
            continue
        os.remove(old_last)
        print_only(f"[Cleanup] Removed {old_last}")

    # Checkpoint strategy: đọc từ config
    save_every_n_steps = config["training"].get("save_every_n_steps", None)

    if save_every_n_steps:
        # T4/GPU yếu: lưu best mỗi epoch + backup mỗi N steps (chống mất khi Colab ngắt)
        checkpoint = ModelCheckpoint(
            checkpoint_dir,
            filename="{epoch}-best",
            monitor="val_loss/dataloader_idx_0",
            mode="min",
            save_top_k=5,
            verbose=True,
            save_last=False,
        )
        callbacks.append(checkpoint)

        step_checkpoint = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="backup-{epoch}-{step}",
            every_n_train_steps=save_every_n_steps,
            save_top_k=3,
            save_last=True,
        )
        callbacks.append(step_checkpoint)
    else:
        # H100/GPU mạnh: chỉ lưu best + last.ckpt mỗi epoch
        checkpoint = ModelCheckpoint(
            checkpoint_dir,
            filename="{epoch}-best",
            monitor="val_loss/dataloader_idx_0",
            mode="min",
            save_top_k=5,
            verbose=True,
            save_last=True,       # last.ckpt cập nhật mỗi epoch
        )
        callbacks.append(checkpoint)

    if config["training"]["early_stop"]:
        print_only("Instantiating EarlyStopping")
        callbacks.append(EarlyStopping(**config["training"]["early_stop"]))

    print_only("Instantiating TQDMProgressBar")
    callbacks.append(TQDMProgressBar(refresh_rate=10))

    # Don't ask GPU if they are not available.
    gpus = config["training"]["gpus"] if torch.cuda.is_available() else None
    distributed_backend = "cuda" if torch.cuda.is_available() else None

    # No external logger

    trainer = pl.Trainer(
        precision=config["training"].get("precision", "16-mixed"),
        max_epochs=config["training"]["epochs"],
        callbacks=callbacks,
        default_root_dir=exp_dir,
        devices=gpus,
        accelerator=distributed_backend,
        strategy=DDPStrategy(find_unused_parameters=True) if len(gpus) > 1 else "auto",
        limit_train_batches=1.0,  # Useful for fast experiment
        gradient_clip_val=5.0,
        accumulate_grad_batches=config["training"].get("accumulate_grad_batches", 1),
        logger=False,
        sync_batchnorm=True,
        # num_sanity_val_steps=0,
        # sync_batchnorm=True,
        # fast_dev_run=True,
    )
    # [NEW] Print essential config before training
    print_only("\n" + "="*40)
    print_only("=== TRAINING CONFIGURATION ===")
    print_only(f" - Model Params:  {sum(p.numel() for p in model.parameters())/1e6:.2f} M")
    print_only(f" - Batch Size:    {config['datamodule']['data_config']['batch_size']}")
    print_only(f" - Segment (s):   {config['datamodule']['data_config']['segment']}")
    print_only(f" - Sample Rate:   {config['datamodule']['data_config']['sample_rate']}")
    print_only(f" - Embd (C):      {config['audionet']['audionet_config']['emb_dim']}")
    print_only(f" - Layers (B):    {config['audionet']['audionet_config']['n_layers']}")
    print_only(f" - Grad Accums:   {config['training'].get('accumulate_grad_batches', 1)}")
    print_only("="*40 + "\n")

    trainer.fit(system, ckpt_path=ckpt_path)
    print_only("Finished Training")
    best_k = {k: v.item() for k, v in checkpoint.best_k_models.items()}
    with open(os.path.join(exp_dir, "best_k_models.json"), "w") as f:
        json.dump(best_k, f, indent=0)

    state_dict = torch.load(checkpoint.best_model_path)
    system.load_state_dict(state_dict=state_dict["state_dict"])
    system.cpu()

    to_save = system.audio_model.serialize()
    torch.save(to_save, os.path.join(exp_dir, "best_model.pth"))


if __name__ == "__main__":
    import yaml
    from pprint import pprint
    from look2hear.utils.parser_utils import (
        prepare_parser_from_dict,
        parse_args_as_dict,
    )

    args = parser.parse_args()
    with open(args.conf_dir) as f:
        def_conf = yaml.safe_load(f)
    parser = prepare_parser_from_dict(def_conf, parser=parser)

    arg_dic, plain_args = parse_args_as_dict(parser, return_plain_args=True)

    # Inject CLI-only args into config
    if args.checkpoint_dir:
        arg_dic["main_args"]["checkpoint_dir"] = args.checkpoint_dir
    if args.resume:
        arg_dic["main_args"]["resume"] = True

    main(arg_dic)
