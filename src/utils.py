import numpy as np
import torch
import tqdm
import os

def encodage_positionnel( d_model, positions) :
   
    position = np.arange(positions)[:, np.newaxis]
    k = np.arange(d_model)[np.newaxis, :]
    i = k // 2
    
    angle_rates = 1 / np.power(10000, (2 * i) / np.float32(d_model))
    angle_rads = position * angle_rates
  
    
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
  
    
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
        enc_mask = (batch["enc_mask"] == 0).to(device)   
 
        tgt      = batch["dec_input_ids"].to(device)
        dec_mask = (batch["dec_mask"] == 0).to(device)
 
        # Teacher forcing : décalage d'un token
        tgt_dec    = tgt[:, :-1]        
        tgt_cible  = tgt[:, 1:]          
        dec_mask   = dec_mask[:, :-1]
 
        seq_len = tgt_dec.size(1)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device), diagonal=1
        ).bool()
 
        # ── Forward (mixed precision) ─────────────────────────────────────────
        with torch.autocast(device_type=device.type, enabled=mixed_precision):
            logits = model(src, tgt_dec, enc_mask, dec_mask, causal_mask)
            loss   = criterion(
                logits.reshape(-1, logits.size(-1)),
                tgt_cible.reshape(-1)
            ) / accum_steps   # normaliser pour l'accumulation
 
        # ── Backward ─────────────────────────────────────────────────────────
        scaler.scale(loss).backward()
 
        # ── Mise à jour tous les accum_steps (ou au dernier batch) ───────────
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

def save_checkpoint(checkpoint_dir, epoch, loss,  model, optimizer, scheduler, scaler):
    path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch:03d}.pt")
    torch.save(
        {
            "epoch":                epoch,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict":    scaler.state_dict(),
            "loss":                 loss,
        },
        path,
    )
    tqdm.write(f"  Checkpoint sauvé → {path}")
    return path


def load_last_checkpoint(checkpoint_dir, model, optimizer, scheduler, scaler, device):
    """Charge le dernier checkpoint dispo et retourne l'epoch de reprise."""
    files = sorted(
        f for f in os.listdir(checkpoint_dir) if f.endswith(".pt")
    )
    if not files:
        print("  Aucun checkpoint trouvé — entraînement from scratch.")
        return 0, None

    path = os.path.join(checkpoint_dir, files[-1])
    ckpt = torch.load(path, map_location=device)

    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    scaler.load_state_dict(ckpt["scaler_state_dict"])

    start = ckpt["epoch"] + 1
    print(
        f"  ▶ Reprise depuis {path} "
        f"(epoch {ckpt['epoch']}, loss {ckpt['loss']:.4f})"
    )
    return start, ckpt["loss"]




import torch


def next_word(
    model,
    tokenizer,
    encoder_input,
    output,
    device,
    max_length=128
):

    # Encoder
    enc = tokenizer(
        encoder_input,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_length
    )

    enc_input_ids = enc["input_ids"].to(device)

    enc_mask = (
        enc["attention_mask"] == 0
    ).to(device)

    # Decoder
    dec_input_ids = output.to(device)

    dec_mask = (
        dec_input_ids == tokenizer.pad_token_id
    )

    # Causal mask
    seq_len = dec_input_ids.size(1)

    causal_mask = torch.triu(
        torch.ones(
            seq_len,
            seq_len,
            device=device
        ),
        diagonal=1
    ).bool()

    # Forward
    predictions = model(
        enc_input_ids,
        dec_input_ids,
        enc_mask,
        dec_mask,
        causal_mask,
    )

    # Last token logits
    predictions = predictions[:, -1, :]

    # Greedy decoding
    predicted_id = torch.argmax(
        predictions,
        dim=-1
    )

    return predicted_id


import torch


def summarize(
    model,
    tokenizer,
    input_document,
    device,
    max_len=128
):

    model.eval()

    with torch.no_grad():

        # Start token
        
        output = torch.tensor(
        [[tokenizer.bos_token_id]],
        device=device
        )

        # Autoregressive generation
        for _ in range(max_len):

            predicted_id = next_word(
                model,
                tokenizer,
                input_document,
                output,
                device,
                max_len
            )

            # Append predicted token
            output = torch.cat(
                [output, predicted_id.unsqueeze(1)],
                dim=1
            )

            # Stop at EOS
            if predicted_id.item() == tokenizer.eos_token_id:
                break

        # Decode
        summary = tokenizer.decode(
            output[0],
            skip_special_tokens=True
        )

    return summary