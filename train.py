from transformer import Transformer
from utils import train_step,preprossessing
from data import Dataset_for_summerisation

import torch.nn as nn
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import dataloader
from torch.optim import AdamW
import tqdm

import os
import yaml

with open("../config/base.yaml", "r") as f :
    cfg = yaml.safe_load(f)


model = Transformer(**cfg["model"])
optimiser = AdamW(model.parameters(), **cfg["optimizer"])
tokenizer = AutoTokenizer.from_pretrained(**cfg["optimizer"])
data = load_dataset(**cfg["data"])

data_prossesse  = data.map(lambda row : preprossessing(row))

dataset = Dataset_for_summerisation(tokenizer, data)

data_loader = dataloader(
    data,
    batch_size = cfg["batch_size"],
    suffle = True,
    num_worker = 4
    )


for  epoch in range() :

    


