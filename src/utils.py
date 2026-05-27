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
    
    return torch.from_numpy(pos_encoding, dtype=torch.float32)

def preprossessing(data) :
    pass

def train_step(
    model,
    data_loader,
    optimizer,
    scheduler,
    criterion,
    scaler,
    clip,
    accum_steps,
    mixed_precision,
    device,
):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
 
    batch_bar = tqdm(data_loader, desc="  batches", leave=False, unit="batch")
 
    for step, batch in enumerate(batch_bar):
 
        # ── Préparation des tenseurs ──────────────────────────────────────────
        src      = batch["enc_input_ids"].to(device)
        enc_mask = (batch["enc_mask"] == 0).to(device)   # True = à ignorer
 
        tgt      = batch["dec_input_ids"].to(device)
        dec_mask = (batch["dec_mask"] == 0).to(device)
 
        # Teacher forcing : décalage d'un token
        tgt_dec    = tgt[:, :-1]         # entrée décodeur  [BOS … t-1]
        tgt_cible  = tgt[:, 1:]          # cible            [t1  … EOS]
        dec_mask   = dec_mask[:, :-1]
 
        seq_len = tgt_dec.size(1)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device), diagonal=1
        ).bool()
 
        
        with torch.autocast(device_type=device.type, enabled=mixed_precision):
            logits = model(src, tgt_dec, enc_mask, dec_mask, causal_mask)
            loss   = criterion(
                logits.reshape(-1, logits.size(-1)),
                tgt_cible.reshape(-1)
            ) / accum_steps   # normaliser pour l'accumulation
 
       
        scaler.scale(loss).backward()
 
        
        last_step = (step + 1) == len(data_loader)
        if (step + 1) % accum_steps == 0 or last_step:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
 
        total_loss += loss.item() * accum_steps
        batch_bar.set_postfix({"loss": f"{loss.item() * accum_steps:.4f}"})
 
    return total_loss / len(data_loader)