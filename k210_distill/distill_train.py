#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Huấn luyện Chưng cất tri thức trực tuyến (Online Knowledge Distillation) cho K210.

Nén Teacher ``SPMamba`` (16 kHz, 6.04M params, ``epoch=91-best.ckpt``) xuống Student
siêu nhẹ ``K210SepUNet`` (8 kHz, ~235K params) trên cụm máy chủ NVIDIA DGX A100 (HUST).

Ví dụ chạy trên cụm SLURM HUST:
    python distill_train.py \\
        --conf_dir configs/distill-k210-a100.yml \\
        --h5_path ~/data/data.h5 \\
        --teacher_ckpt epoch=91-best.ckpt
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import warnings

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.strategies.ddp import DDPStrategy

import look2hear.datas
import look2hear.losses
import look2hear.models
import look2hear.system
from look2hear.system import load_teacher_model, make_optimizer
from look2hear.utils import print_only

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision("high")  # Tận dụng TensorFloat-32 trên NVIDIA A100

parser = argparse.ArgumentParser(description="Online Distillation SPMamba -> K210SepUNet")
parser.add_argument(
    "--conf_dir",
    default="configs/distill-k210-a100.yml",
    help="Đường dẫn tới file cấu hình YAML",
)
parser.add_argument(
    "--h5_path",
    default=None,
    help="Ghi đè đường dẫn file data.h5 (ví dụ: ~/data/data.h5)",
)
parser.add_argument(
    "--teacher_ckpt",
    default=None,
    help="Đường dẫn tới checkpoint Teacher SPMamba (mặc định đọc từ config: epoch=91-best.ckpt)",
)
parser.add_argument(
    "--no_teacher",
    action="store_true",
    help="Tắt Teacher SPMamba, chỉ huấn luyện Student trực tiếp trên nhãn thật 8 kHz",
)
parser.add_argument(
    "--checkpoint_dir",
    default=None,
    help="Thư mục lưu checkpoints",
)
parser.add_argument(
    "--resume",
    action="store_true",
    help="Tiếp tục huấn luyện từ last*.ckpt trong checkpoint_dir",
)
parser.add_argument(
    "--resume_ckpt",
    default=None,
    help="Đường dẫn tường minh tới checkpoint cần resume",
)
parser.add_argument(
    "--wandb",
    action="store_true",
    help="Bật ghi log Weights & Biases",
)
parser.add_argument(
    "--wandb_project",
    default=None,
    help="Tên project Weights & Biases",
)
parser.add_argument(
    "--wandb_run_name",
    default=None,
    help="Tên run Weights & Biases",
)
parser.add_argument(
    "--wandb_offline",
    action="store_true",
    help="Ghi log W&B ở chế độ offline (dùng khi compute node SLURM không ra Internet)",
)
parser.add_argument(
    "--fast_dev_run",
    action="store_true",
    help="Chạy thử 1 batch train + 1 batch val để kiểm tra luồng dữ liệu & GPU trước khi submit SLURM",
)


def build_logger(config, exp_dir):
    logger_config = config.get("logger", {})
    if not logger_config or not logger_config.get("enabled", False):
        print_only("Logger disabled")
        return False

    logger_name = str(logger_config.get("name", "wandb")).lower()
    if logger_name != "wandb":
        raise ValueError(f"Unsupported logger: {logger_name}")

    try:
        from pytorch_lightning.loggers import WandbLogger
    except ImportError as exc:
        raise RuntimeError(
            "W&B logging được bật nhưng chưa cài gói `wandb`. "
            "Chạy `pip install wandb` hoặc tắt logger trong config."
        ) from exc

    if logger_config.get("offline", False):
        os.environ.setdefault("WANDB_MODE", "offline")

    project = logger_config.get("project") or "spmamba-k210"
    run_name = logger_config.get("run_name") or config["exp"]["exp_name"]
    save_dir = logger_config.get("save_dir") or exp_dir
    os.makedirs(save_dir, exist_ok=True)
    log_model = logger_config.get("log_model", False)
    entity = logger_config.get("entity")

    kwargs = {
        "project": project,
        "name": run_name,
        "save_dir": save_dir,
        "log_model": log_model,
    }
    if entity:
        kwargs["entity"] = entity

    print_only(f"Instantiating WandbLogger project={project}, name={run_name}")
    return WandbLogger(**kwargs)


def main(config):
    print_only(
        "Instantiating datamodule <{}>".format(config["datamodule"]["data_name"])
    )
    datamodule = getattr(look2hear.datas, config["datamodule"]["data_name"])(
        **config["datamodule"]["data_config"]
    )
    datamodule.setup()
    train_loader, val_loader, test_loader = datamodule.make_loader

    # Khởi tạo Student Model (mặc định 8 kHz cho K210)
    audionet_cfg = dict(config["audionet"]["audionet_config"])
    student_sr = int(
        audionet_cfg.pop(
            "sample_rate",
            config.get("distill", {}).get(
                "student_sample_rate",
                config["datamodule"]["data_config"]["sample_rate"],
            ),
        )
    )
    print_only(
        f"Instantiating Student AudioNet <{config['audionet']['audionet_name']}> @ {student_sr} Hz"
    )
    model = getattr(look2hear.models, config["audionet"]["audionet_name"])(
        sample_rate=student_sr,
        **audionet_cfg,
    )

    # Khởi tạo Teacher Model (nếu bật chưng cất)
    distill_cfg = config.get("distill", {})
    init_ckpt = distill_cfg.get("init_ckpt")
    if init_ckpt and os.path.isfile(init_ckpt) and not config["main_args"].get("resume_ckpt"):
        print_only(f"[Warm-Start] Loading pretrained KPU Conv2D weights from <{init_ckpt}>")
        try:
            raw_init = torch.load(init_ckpt, map_location="cpu", weights_only=False)
        except TypeError:
            raw_init = torch.load(init_ckpt, map_location="cpu")
        sd = raw_init.get("state_dict", raw_init)
        clean_sd = {}
        for k, v in sd.items():
            if k.startswith("audio_model."):
                clean_sd[k[len("audio_model.") :]] = v
            elif not k.startswith("resampler.") and not k.startswith("pit_aligner."):
                clean_sd[k] = v
        missing, unexpected = model.load_state_dict(clean_sd, strict=False)
        print_only(
            f"[Warm-Start] Loaded {len(clean_sd) - len(unexpected)} tensors "
            f"(new RISC-V Attention/GRU tensors: {len(missing)})"
        )

    use_teacher = bool(distill_cfg.get("enabled", True)) and not config["main_args"].get(
        "no_teacher", False
    )
    teacher_model = None
    if use_teacher:
        teacher_ckpt = (
            config["main_args"].get("teacher_ckpt")
            or distill_cfg.get("teacher_ckpt")
            or "epoch=91-best.ckpt"
        )
        if not os.path.isfile(teacher_ckpt):
            raise FileNotFoundError(
                f"Khong tim thay checkpoint Teacher SPMamba tai '{teacher_ckpt}'. "
                "Hay kiem tra duong dan --teacher_ckpt hoac dung co --no_teacher."
            )
        print_only(f"Loading frozen Teacher SPMamba from <{teacher_ckpt}>")
        teacher_model = load_teacher_model(teacher_ckpt, map_location="cpu")
        t_params = sum(p.numel() for p in teacher_model.parameters()) / 1e6
        print_only(f"Teacher loaded successfully ({t_params:.2f} M params, frozen)")
    else:
        print_only("Training Student WITHOUT Teacher (--no_teacher or distill.enabled=False)")

    print_only("Instantiating Optimizer <{}>".format(config["optimizer"]["optim_name"]))
    optimizer = make_optimizer(model.parameters(), **config["optimizer"])

    scheduler = None
    if config["scheduler"]["sche_name"]:
        print_only(
            "Instantiating Scheduler <{}>".format(config["scheduler"]["sche_name"])
        )
        if config["scheduler"]["sche_name"] != "DPTNetScheduler":
            scheduler = getattr(
                torch.optim.lr_scheduler, config["scheduler"]["sche_name"]
            )(optimizer=optimizer, **config["scheduler"]["sche_config"])
        else:
            scheduler = {
                "scheduler": getattr(
                    look2hear.system.schedulers, config["scheduler"]["sche_name"]
                )(
                    optimizer,
                    len(train_loader)
                    // config["datamodule"]["data_config"]["batch_size"],
                    64,
                ),
                "interval": "step",
            }

    if config["main_args"].get("checkpoint_dir"):
        exp_dir = config["main_args"]["checkpoint_dir"]
    else:
        exp_dir = os.path.join(
            os.getcwd(), "Experiments", "checkpoint", config["exp"]["exp_name"]
        )
    config["main_args"]["exp_dir"] = exp_dir
    os.makedirs(exp_dir, exist_ok=True)
    print_only(f"[Checkpoint Dir] {exp_dir}")

    conf_path = os.path.join(exp_dir, "conf.yml")
    with open(conf_path, "w") as outfile:
        yaml.safe_dump(config, outfile)

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

    system_name = config["training"].get("system", "DistillLightningModule")
    print_only(f"Instantiating System <{system_name}>")
    system_cls = getattr(look2hear.system, system_name)
    system_kwargs = dict(
        audio_model=model,
        loss_func=loss_func,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        scheduler=scheduler,
        config=config,
    )
    if system_name == "DistillLightningModule":
        system_kwargs["teacher_model"] = teacher_model
    system = system_cls(**system_kwargs)

    callbacks = []
    checkpoint_dir = os.path.join(exp_dir)

    ckpt_path = None
    resume_ckpt = config["main_args"].get("resume_ckpt")
    if resume_ckpt:
        if os.path.isfile(resume_ckpt):
            ckpt_path = resume_ckpt
            print_only(f"Resuming from explicit checkpoint: {ckpt_path}")
        else:
            print_only(f"[Warning] --resume_ckpt not found: {resume_ckpt}")
    if ckpt_path is None and config["main_args"].get("resume"):
        last_ckpts = glob.glob(os.path.join(exp_dir, "last*.ckpt"))
        if last_ckpts:
            ckpt_path = max(last_ckpts, key=os.path.getmtime)
            print_only(f"Resuming from last checkpoint: {ckpt_path}")
        else:
            all_ckpts = glob.glob(os.path.join(exp_dir, "*.ckpt"))
            if all_ckpts:
                ckpt_path = max(all_ckpts, key=os.path.getmtime)
                print_only(
                    f"[Warning] no last*.ckpt found. Resuming from newest: {ckpt_path}"
                )
            else:
                print_only("[Warning] --resume set but no checkpoints found. Starting fresh.")

    for old_last in glob.glob(os.path.join(checkpoint_dir, "last*.ckpt")):
        if ckpt_path and os.path.abspath(old_last) == os.path.abspath(ckpt_path):
            continue
        os.remove(old_last)

    checkpoint = ModelCheckpoint(
        checkpoint_dir,
        filename="{epoch}-best",
        monitor="val_loss/dataloader_idx_0",
        mode="min",
        save_top_k=5,
        verbose=True,
        save_last=True,
    )
    callbacks.append(checkpoint)

    if config["training"].get("early_stop"):
        callbacks.append(EarlyStopping(**config["training"]["early_stop"]))
    callbacks.append(TQDMProgressBar(refresh_rate=10))

    gpus = config["training"]["gpus"] if torch.cuda.is_available() else 1
    accelerator = "cuda" if torch.cuda.is_available() else "cpu"
    fast_dev_run = bool(config["main_args"].get("fast_dev_run", False))
    trainer_logger = False if fast_dev_run else build_logger(config, exp_dir)

    trainer = pl.Trainer(
        precision=config["training"].get("precision", "bf16-mixed")
        if torch.cuda.is_available()
        else 32,
        max_epochs=config["training"]["epochs"],
        callbacks=callbacks,
        default_root_dir=exp_dir,
        devices=gpus,
        accelerator=accelerator,
        strategy=DDPStrategy(find_unused_parameters=False)
        if isinstance(gpus, list) and len(gpus) > 1
        else "auto",
        limit_train_batches=1.0,
        gradient_clip_val=5.0,
        accumulate_grad_batches=config["training"].get("accumulate_grad_batches", 1),
        logger=trainer_logger,
        log_every_n_steps=config["training"].get("log_every_n_steps", 25),
        sync_batchnorm=isinstance(gpus, list) and len(gpus) > 1,
        fast_dev_run=fast_dev_run,
    )

    stu_params = sum(p.numel() for p in model.parameters())
    print_only("\n" + "=" * 54)
    print_only("=== K210 STUDENT DISTILLATION CONFIGURATION ===")
    print_only(f" - Student Model:   {config['audionet']['audionet_name']}")
    print_only(f" - Student Params:  {stu_params:,} ({stu_params / 1e3:.1f} K ~ {stu_params / 1024:.1f} KB INT8)")
    print_only(f" - Source SR (H5):  {config['datamodule']['data_config']['sample_rate']} Hz")
    print_only(f" - Student SR:      {student_sr} Hz")
    print_only(f" - Teacher Active:  {teacher_model is not None}")
    print_only(f" - Batch Size:      {config['datamodule']['data_config']['batch_size']}")
    print_only(f" - Segment (s):     {config['datamodule']['data_config']['segment']}")
    print_only("=" * 54 + "\n")

    trainer.fit(system, ckpt_path=ckpt_path)
    print_only("Finished Training")

    if fast_dev_run or not checkpoint.best_model_path:
        return

    best_k = {k: v.item() for k, v in checkpoint.best_k_models.items()}
    with open(os.path.join(exp_dir, "best_k_models.json"), "w") as f:
        json.dump(best_k, f, indent=2)

    try:
        state_dict = torch.load(checkpoint.best_model_path, map_location="cpu", weights_only=False)
    except TypeError:
        state_dict = torch.load(checkpoint.best_model_path, map_location="cpu")
    system.load_state_dict(state_dict=state_dict["state_dict"])
    system.cpu()

    to_save = system.audio_model.serialize()
    best_pth = os.path.join(exp_dir, "best_model.pth")
    torch.save(to_save, best_pth)
    print_only(f"Saved serialized Student model to: {best_pth}")


def build_config_from_args(args) -> dict:
    with open(args.conf_dir, encoding="utf-8") as f:
        arg_dic = yaml.safe_load(f) or {}

    arg_dic.setdefault("main_args", {})
    arg_dic["main_args"]["conf_dir"] = args.conf_dir

    if args.h5_path:
        arg_dic["datamodule"]["data_config"]["h5_path"] = os.path.expanduser(args.h5_path)
    if args.teacher_ckpt:
        arg_dic["main_args"]["teacher_ckpt"] = os.path.expanduser(args.teacher_ckpt)
    if args.no_teacher:
        arg_dic["main_args"]["no_teacher"] = True
    if args.checkpoint_dir:
        arg_dic["main_args"]["checkpoint_dir"] = os.path.expanduser(args.checkpoint_dir)
    if args.resume:
        arg_dic["main_args"]["resume"] = True
    if args.resume_ckpt:
        arg_dic["main_args"]["resume_ckpt"] = os.path.expanduser(args.resume_ckpt)
    if args.wandb:
        arg_dic.setdefault("logger", {})
        arg_dic["logger"]["enabled"] = True
        arg_dic["logger"]["name"] = "wandb"
    if args.wandb_project:
        arg_dic.setdefault("logger", {})
        arg_dic["logger"]["project"] = args.wandb_project
    if args.wandb_run_name:
        arg_dic.setdefault("logger", {})
        arg_dic["logger"]["run_name"] = args.wandb_run_name
    if args.wandb_offline:
        arg_dic.setdefault("logger", {})
        arg_dic["logger"]["offline"] = True
    if args.fast_dev_run:
        arg_dic["main_args"]["fast_dev_run"] = True

    return arg_dic


if __name__ == "__main__":
    cli_args = parser.parse_args()
    config_dic = build_config_from_args(cli_args)
    main(config_dic)

