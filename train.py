from transformer import Transformer
from utils import (
    train_step, 
    preprossessing, 
    save_checkpoint, 
    load_last_checkpoint
)
from data import Dataset_for_summerisation

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.amp import GradScaler
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from tqdm import tqdm

import os
import yaml
import mlflow
import mlflow.pytorch

# Config
with open("../config/base.yaml", "r") as f:
    cfg = yaml.safe_load(f)

device = torch.device(
    cfg["training"]["device"] if torch.cuda.is_available() else "cpu"
)
print(f"Device : {device}")
if device.type == "cuda":
    print(f"GPU    : {torch.cuda.get_device_name(0)}")
    print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Tokenizer / Data 
tokenizer = AutoTokenizer.from_pretrained(
    cfg["tokenizer"]["pretrained_model_name"]
)


cfg["model"]["vocab_size"] = tokenizer.vocab_size

raw_data = load_dataset(cfg["dataset"]["path"], cfg["dataset"]["name"])


data_processed = [
    preprossessing(row) for row in raw_data[cfg["dataset"]["split"]]
]

dataset = Dataset_for_summerisation(tokenizer, data_processed)

data_loader = DataLoader(
    dataset,
    batch_size=cfg["data"]["batch_size"],
    shuffle=True,
    num_workers=2,
    pin_memory=True,           
    persistent_workers=True,
)

#  Modèle 
model = Transformer(**cfg["model"]).to(device)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Paramètres : {n_params / 1e6:.1f} M")

# Optimizer / Scheduler / Scaler 
optimizer = AdamW(
    model.parameters(),
    lr=cfg["optimizer"]["lr"],
    weight_decay=cfg["optimizer"]["weight_decay"],
)

warmup_steps = cfg["scheduler"]["warmup_steps"]
total_steps  = cfg["scheduler"]["total_steps"]

warmup_sched = LinearLR(
    optimizer,
    start_factor=0.01,
    end_factor=1.0,
    total_iters=warmup_steps,
)
cosine_sched = CosineAnnealingLR(
    optimizer,
    T_max=total_steps - warmup_steps,
    eta_min=1e-6,
)
scheduler = SequentialLR(
    optimizer,
    schedulers=[warmup_sched, cosine_sched],
    milestones=[warmup_steps],
)

use_amp = cfg["training"]["mixed_precision"] and device.type == "cuda"
scaler  = GradScaler(enabled=use_amp)

criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)


checkpoint_dir = cfg["training"]["checkpoint_dir"]
os.makedirs(checkpoint_dir, exist_ok=True)



start_epoch, last_loss = load_last_checkpoint(checkpoint_dir, model, optimizer, scheduler, scaler, device)

n_epochs    = cfg["training"]["n_epochs"]
clip        = cfg["training"]["clip"]
accum_steps = cfg["training"]["gradient_accumulation_steps"]
ckpt_freq   = cfg["training"]["checkpoint_freq"]

# MLflow 
mlflow.set_experiment("transformer_summarisation")

with mlflow.start_run(run_name=f"run_from_epoch_{start_epoch}"):

    
    mlflow.log_params({
        "embedding_dim":    cfg["model"]["embedding_dim"],
        "n_heads":          cfg["model"]["n_heads"],
        "n_layer":          cfg["model"]["n_layer"],
        "n_params_M":       round(n_params / 1e6, 2),
        "batch_size":       cfg["data"]["batch_size"],
        "accum_steps":      accum_steps,
        "effective_batch":  cfg["data"]["batch_size"] * accum_steps,
        "lr":               cfg["optimizer"]["lr"],
        "weight_decay":     cfg["optimizer"]["weight_decay"],
        "warmup_steps":     warmup_steps,
        "total_steps":      total_steps,
        "mixed_precision":  use_amp,
        "clip":             clip,
        "tokenizer":        cfg["tokenizer"]["pretrained_model_name"],
    })

    epoch_bar = tqdm(
        range(start_epoch, n_epochs),
        desc="Epochs",
        unit="epoch",
        initial=start_epoch,
        total=n_epochs,
    )

    for epoch in epoch_bar:

        #  Reset stats mémoire 
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        
        loss = train_step(
            model,
            data_loader,
            optimizer,
            scheduler,
            criterion,
            scaler,
            clip,
            accum_steps,
            use_amp,
            device,
        )

        # Stats mémoire 
        mem_gb = 0.0
        if device.type == "cuda":
            mem_gb = torch.cuda.max_memory_allocated() / 1e9

        current_lr = scheduler.get_last_lr()[0]

        epoch_bar.set_postfix({
            "loss": f"{loss:.4f}",
            "lr":   f"{current_lr:.2e}",
            "VRAM": f"{mem_gb:.1f}GB",
        })

        # ── Log MLflow 
        mlflow.log_metrics(
            {
                "train_loss": loss,
                "learning_rate": current_lr,
                "vram_gb": mem_gb,
            },
            step=epoch,
        )

        #Checkpoint
        if (epoch + 1) % ckpt_freq == 0 or (epoch + 1) == n_epochs:
            ckpt_path = save_checkpoint(checkpoint_dir, epoch, loss,  model, optimizer, scheduler, scaler)
            mlflow.log_artifact(ckpt_path)

    #  Sauvegarde finale du modèle dans MLflow 
    mlflow.pytorch.log_model(model, artifact_path="model")
    print("\n Entraînement terminé. Modèle logué dans MLflow.")