import numpy as np
import torch
from tqdm import tqdm
import os
from rouge_score import rouge_scorer


# ── Encodage positionnel ──────────────────────────────────────────────────────

def encodage_positionnel(d_model, positions):
    position    = np.arange(positions)[:, np.newaxis]
    k           = np.arange(d_model)[np.newaxis, :]
    i           = k // 2
    angle_rates = 1 / np.power(10000, (2 * i) / np.float32(d_model))
    angle_rads  = position * angle_rates
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return torch.from_numpy(angle_rads[np.newaxis, ...]).float()


# ── Train step ────────────────────────────────────────────────────────────────

def train_step(model, data_loader, optimizer, scheduler, criterion,
               scaler, clip, accum_steps, mixed_precision, device):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    batch_bar = tqdm(data_loader, desc="  batches", leave=False, unit="batch")

    for step, batch in enumerate(batch_bar):
        src      = batch["enc_input_ids"].to(device)
        enc_mask = (batch["enc_mask"] == 0).to(device)
        tgt      = batch["dec_input_ids"].to(device)

        tgt_dec   = tgt[:, :-1]
        tgt_cible = tgt[:, 1:]

        seq_len     = tgt_dec.size(1)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

        with torch.autocast(device_type=device.type, enabled=mixed_precision):
            logits = model(src, tgt_dec, enc_mask, causal_mask)
            loss   = criterion(logits.reshape(-1, logits.size(-1)), tgt_cible.reshape(-1)) / accum_steps

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


# ── Checkpoint ────────────────────────────────────────────────────────────────

def save_checkpoint(checkpoint_dir, epoch, loss, model, optimizer, scheduler, scaler):
    path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch:03d}.pt")
    torch.save({
        "epoch":                epoch,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict":    scaler.state_dict(),
        "loss":                 loss,
    }, path)
    tqdm.write(f"  ✅ Checkpoint sauvé → {path}")
    return path


def load_last_checkpoint(checkpoint_dir, model, optimizer, scheduler, scaler, device):
    files = sorted(f for f in os.listdir(checkpoint_dir) if f.endswith(".pt"))
    if not files:
        print("  Aucun checkpoint — entraînement from scratch.")
        return 0, None

    path = os.path.join(checkpoint_dir, files[-1])
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    scaler.load_state_dict(ckpt["scaler_state_dict"])
    print(f"  ▶ Reprise depuis {path} (epoch {ckpt['epoch']}, loss {ckpt['loss']:.4f})")
    return ckpt["epoch"] + 1, ckpt["loss"]


# ── Génération ────────────────────────────────────────────────────────────────

def summarize(model, tokenizer, input_document, device, max_len=128):
    """
    Génération autoregressive — l'encodeur est calculé UNE SEULE FOIS
    puis réutilisé à chaque step de décodage.
    """
    model.eval()
    with torch.no_grad():

        # Encoder — 1 seul appel pour tout le processus de génération
        enc        = tokenizer(input_document, return_tensors="pt",
                               truncation=True, padding=True, max_length=512)
        enc_ids    = enc["input_ids"].to(device)
        enc_mask   = (enc["attention_mask"] == 0).to(device)
        context    = model.encoder(enc_ids, enc_mask)   # FIX : calculé une fois

        # Décodage autorégressif
        output = torch.tensor([[tokenizer.bos_token_id]], device=device)

        for _ in range(max_len):
            seq_len     = output.size(1)
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

            decoded    = model.decoder(output, context, enc_mask, causal_mask)
            logits     = model.lm_head(decoded)
            next_token = logits[:, -1, :].argmax(dim=-1)

            output = torch.cat([output, next_token.unsqueeze(1)], dim=1)

            if next_token.item() == tokenizer.eos_token_id:
                break

        return tokenizer.decode(output[0], skip_special_tokens=True)


# ── Eval step ─────────────────────────────────────────────────────────────────

def eval_step(model, data_loader, criterion, tokenizer, device,
              mixed_precision, num_samples_rouge=64):
    """
    - test_loss : cross-entropy sur tout le split (teacher forcing)
    - ROUGE     : calculé sur num_samples_rouge exemples via génération réelle
                  (autoregressive) pour avoir des scores représentatifs
    """
    model.eval()
    total_loss = 0.0
    r1_scores, r2_scores, rL_scores = [], [], []
    decoded_count = 0

    scorer    = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    batch_bar = tqdm(data_loader, desc="  eval", leave=False, unit="batch")

    with torch.no_grad():
        for batch in batch_bar:
            src      = batch["enc_input_ids"].to(device)
            enc_mask = (batch["enc_mask"] == 0).to(device)
            tgt      = batch["dec_input_ids"].to(device)
            tgt_dec  = tgt[:, :-1]
            tgt_cible = tgt[:, 1:]

            seq_len     = tgt_dec.size(1)
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

            # Loss en teacher forcing (rapide)
            with torch.autocast(device_type=device.type, enabled=mixed_precision):
                logits = model(src, tgt_dec, enc_mask, causal_mask)
                loss   = criterion(logits.reshape(-1, logits.size(-1)), tgt_cible.reshape(-1))

            total_loss += loss.item()
            batch_bar.set_postfix({"eval_loss": f"{loss.item():.4f}"})

            # ROUGE via génération autoregressive réelle
            if decoded_count < num_samples_rouge:
                for i in range(src.size(0)):
                    if decoded_count >= num_samples_rouge:
                        break
                    src_text = tokenizer.decode(src[i].tolist(), skip_special_tokens=True)
                    ref_text = tokenizer.decode(tgt_cible[i].tolist(), skip_special_tokens=True)
                    pred_text = summarize(model, tokenizer, src_text, device)

                    s = scorer.score(ref_text, pred_text)
                    r1_scores.append(s["rouge1"].fmeasure)
                    r2_scores.append(s["rouge2"].fmeasure)
                    rL_scores.append(s["rougeL"].fmeasure)
                    decoded_count += 1

    return {
        "test_loss": total_loss / len(data_loader),
        "rouge1":    float(np.mean(r1_scores)) if r1_scores else 0.0,
        "rouge2":    float(np.mean(r2_scores)) if r2_scores else 0.0,
        "rougeL":    float(np.mean(rL_scores)) if rL_scores else 0.0,
    }