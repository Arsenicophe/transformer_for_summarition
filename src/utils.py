import numpy as np
import torch

def encodage_positionnel( d_model, positions) :
   
    position = np.arange(positions)[:, np.newaxis]
    k = np.arange(d_model)[np.newaxis, :]
    i = k // 2
    
    # initialize a matrix angle_rads of all the angles 
    angle_rates = 1 / np.power(10000, (2 * i) / np.float32(d_model))
    angle_rads = position * angle_rates
  
    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
  
    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
    pos_encoding = angle_rads[np.newaxis, ...]
    
    return torch.cast(pos_encoding, dtype=torch.float32)

def preprossessing(data) :
    pass

def train_step(model, batch, optimizer, criterion):

    src = batch["enc_input_ids"]
    enc_mask = (batch["enc_mask"] == 0)

    tgt = batch["dec_input_ids"]
    dec_mask = (batch["dec_mask"] == 0)
    
    tgt_dec = tgt[:, :-1]
    tgt_cible = tgt[:, 1:]

    dec_mask = dec_mask[:, :-1]

    seq_len = tgt_dec.size(1)

    causal_mask = torch.triu(
        torch.ones(seq_len, seq_len, device=tgt_dec.device),
        diagonal=1
    ).bool()

    model.train()

    optimizer.zero_grad()

    logits = model(
        src,
        tgt_dec,
        enc_mask,
        dec_mask,
        causal_mask
    )

    loss = criterion(
        logits.reshape(-1, logits.size(-1)),
        tgt_cible.reshape(-1)
    )

    loss.backward()
    optimizer.step()

    return loss.item()

    

