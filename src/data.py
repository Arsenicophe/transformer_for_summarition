import torch.nn as nn
import torch
from torch.utils.data import  Dataset



class Dataset_for_summerisation(Dataset):

    def __init__(self, tokenizer, data ):
        super().__init__()
        self.tokenizer = tokenizer
        self.data = data

    def __len__(self):
        return len(self.data)
    

    def __getitem__(self, index):
        src, tgt = self.data[index]

        enc = self.tokenizer(

            src,

            padding="max_length",

            truncation=True,

            max_length=32,

            return_tensors="pt"

        )

        dec = self.tokenizer(

            tgt,

            padding="max_length",

            truncation=True,

            max_length=32,

            return_tensors="pt"

        )

        return {

            "enc_input_ids": enc["input_ids"].squeeze(0),

            "enc_mask": enc["attention_mask"].squeeze(0),

            "dec_input_ids": dec["input_ids"].squeeze(0),

            "dec_mask": dec["attention_mask"].squeeze(0),

        }