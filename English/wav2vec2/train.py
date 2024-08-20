import json, random
from pathlib import Path

import numpy as np

from tqdm import tqdm

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

import jiwer

from tokenizer import Tokenizer, manual_seed
from dataset import LibriSpeech, collate_fn
from alignment import sample_alignment_ones, sample_alignment
from diffusion import Diffuser
from model import DNAR



def train_1(train_loader, diffuser, model, optimizer, scheduler, scaler, epoch, device):
    model.train()

    optimizer.zero_grad()

    loss_dm = torch.tensor(0.0)

    train_loss = []
    train_loss_dm = []

    train_bar = tqdm(train_loader, desc=f'[ Train | {epoch + 1:03d}/120 ]', ncols=0)

    for step, (features, attention_mask, input_lengths, texts, target_lengths, logits_ctc, alignments) in enumerate(train_bar):
        features = features.to(device)
        attention_mask = attention_mask.to(device)
        input_lengths = input_lengths.to(device)
        texts = texts.to(device)
        target_lengths = target_lengths.to(device)

        with torch.autocast(device):
            hidden_states = model.encoder(features, attention_mask)

            logits = model.encoder.ctc(hidden_states)
            logits = F.log_softmax(logits, dim=-1)

            loss = F.ctc_loss(logits.transpose(0, 1), texts, input_lengths, target_lengths)

        skip = (epoch < 30 or random.random() < 0.9)

        if not skip:
            logits = logits.cpu()
            texts = texts.cpu()

            logits_ctc = diffuser.split(logits.detach(), input_lengths)
            texts_ctc = torch.split(texts, target_lengths.tolist(), dim=0)

            alignments = sample_alignment_ones(logits_ctc, texts_ctc)
            alignments = alignments.to(device)

            x_0 = alignments
            x_0_onehot = diffuser.one_hot(x_0)

            timesteps = torch.randint(1, diffuser.timesteps + 1, [alignments.size(0)], device=device)

            x_t_prob = diffuser.forward_from_x_0(x_0_onehot, timesteps)

            x_t = diffuser.sample(x_t_prob)
            x_t_onehot = diffuser.one_hot(x_t)

            x_t_mask = (x_t == diffuser.classes - 1)

            with torch.autocast(device):
                logits = model.decoder(hidden_states.detach(), attention_mask, x_t, timesteps)
                logits = F.softmax(logits, dim=-1)

                x_0_pred_prob = torch.cat([logits, torch.zeros_like(logits[:, :, :1])], dim=-1)

                loss_dm = diffuser.dfba(
                    x_t_onehot, x_0_pred_prob, timesteps, x_t_mask, input_lengths, texts, target_lengths, logits_ctc
                )

        if skip:
            scaler.scale(loss).backward()
        else:
            scaler.scale(loss + loss_dm).backward()

        if (step + 1) % 4 == 0:
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        train_loss.append(loss.item())

        if not skip:
            train_loss_dm.append(loss_dm.item())

        train_bar.set_postfix({
            'loss': f'{loss.item():.4f} {loss_dm.item():.4f}',
            'mean': f'{np.mean(train_loss):.4f} {np.mean(train_loss_dm):.4f}',
            'lr': f'{scheduler.get_last_lr()[0]:.2e}'
        })

    torch.save(model.state_dict(), f'checkpoints/model_{epoch + 1:03d}.pt')

    torch.cuda.empty_cache()



def train_2(train_loader, diffuser, model, optimizer, scheduler, scaler, epoch, device):
    model.encoder.eval()
    model.decoder.train()

    train_loss = []

    train_bar = tqdm(train_loader, desc=f'[ Train | {epoch + 1:03d}/120 ]', ncols=0)

    for features, attention_mask, input_lengths, texts, target_lengths, logits_ctc, alignments in train_bar:
        features = features.to(device)
        attention_mask = attention_mask.to(device)
        alignments = alignments.to(device)

        x_0 = alignments
        x_0_onehot = diffuser.one_hot(x_0)

        timesteps = torch.randint(1, diffuser.timesteps + 1, [alignments.size(0)], device=device)

        x_t_prob = diffuser.forward_from_x_0(x_0_onehot, timesteps)

        x_t = diffuser.sample(x_t_prob)
        x_t_onehot = diffuser.one_hot(x_t)

        x_t_mask = (x_t == diffuser.classes - 1)

        with torch.autocast(device):
            with torch.no_grad():
                hidden_states = model.encoder(features, attention_mask)

            logits = model.decoder(hidden_states, attention_mask, x_t, timesteps)
            logits = F.softmax(logits, dim=-1)

            x_0_pred_prob = torch.cat([logits, torch.zeros_like(logits[:, :, :1])], dim=-1)

            loss = diffuser.dfba(
                x_t_onehot, x_0_pred_prob, timesteps, x_t_mask, input_lengths, texts, target_lengths, logits_ctc
            )

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        train_loss.append(loss.item())

        train_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'mean': f'{np.mean(train_loss):.4f}',
            'lr': f'{scheduler.get_last_lr()[0]:.2e}'
        })

    torch.save(model.state_dict(), f'checkpoints/model_{epoch + 1:03d}.pt')

    torch.cuda.empty_cache()



@torch.inference_mode()
def valid(tokenizer, valid_loader, diffuser, model, epoch, device):
    model.eval()

    texts_true = []
    texts_pred = []

    valid_bar = tqdm(valid_loader, desc=f'[ Valid | {epoch + 1:03d}/120 ]', ncols=0)

    for features, attention_mask, input_lengths, texts, target_lengths in valid_bar:
        features = features.to(device)
        attention_mask = attention_mask.to(device)

        with torch.autocast(device):
            hidden_states = model.encoder(features, attention_mask)

        x_t = torch.full([hidden_states.size(0), hidden_states.size(1)], diffuser.classes - 1, device=device)
        x_t_onehot = diffuser.one_hot(x_t)

        for timestep in range(diffuser.timesteps, 0, -1):
            timesteps = torch.full([x_t.size(0)], timestep, device=device)

            x_t_mask = (x_t == diffuser.classes - 1)

            with torch.autocast(device):
                logits = model.decoder(hidden_states, attention_mask, x_t, timesteps)

            x_0_pred = torch.where(x_t_mask, torch.argmax(logits, dim=-1), x_t)
            x_0_pred_onehot = diffuser.one_hot(x_0_pred)

            x_t_prob = diffuser.reverse(x_t_onehot, x_0_pred_onehot, timesteps)

            x_t = diffuser.sample(x_t_prob)
            x_t_onehot = diffuser.one_hot(x_t)

        texts_true += [tokenizer.decode(text) for text in torch.split(texts, target_lengths.tolist(), dim=0)]
        texts_pred += tokenizer.batch_decode_ctc(x_t, attention_mask)

        valid_bar.set_postfix({
            'wer': f'{jiwer.wer(texts_true, texts_pred):.2%}'
        })

    torch.cuda.empty_cache()

    return jiwer.wer(texts_true, texts_pred)



def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = Tokenizer()

    train_set = LibriSpeech(
        'datasets/train-clean-100.csv',
        tokenizer,
        True
    )
    train_loader = DataLoader(
        train_set,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        collate_fn=lambda batch: collate_fn(batch, True),
        pin_memory=True,
        drop_last=True
    )

    valid_set = LibriSpeech(
        'datasets/dev-other.csv',
        tokenizer,
        False
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        collate_fn=lambda batch: collate_fn(batch, False),
        pin_memory=True
    )

    diffuser = Diffuser(200, len(tokenizer) + 1, device=device)

    model = DNAR().to(device)

    optimizer_1 = torch.optim.Adam(
        model.parameters(),
        lr=5e-5,
        betas=[0.9, 0.98],
        weight_decay=1e-8
    )

    iters = len(train_loader) // 4

    scheduler_1 = torch.optim.lr_scheduler.SequentialLR(
        optimizer_1, [
            torch.optim.lr_scheduler.LinearLR(
                optimizer_1, start_factor=0.01, end_factor=1, total_iters=2 * iters
            ),
            torch.optim.lr_scheduler.LinearLR(
                optimizer_1, start_factor=1, end_factor=0.2, total_iters=28 * iters
            ),
            torch.optim.lr_scheduler.LinearLR(
                optimizer_1, start_factor=0.2, end_factor=0.1, total_iters=60 * iters
            )
        ],
        milestones=[
            2 * iters,
            30 * iters
        ]
    )

    scaler_1 = torch.cuda.amp.GradScaler()

    optimizer_2 = torch.optim.Adam(
        model.decoder.parameters(),
        lr=5e-5,
        betas=[0.9, 0.98],
        weight_decay=1e-8
    )

    iters = len(train_loader)

    scheduler_2 = torch.optim.lr_scheduler.SequentialLR(
        optimizer_2, [
            torch.optim.lr_scheduler.LinearLR(
                optimizer_2, start_factor=0.01, end_factor=1, total_iters=2 * iters
            ),
            torch.optim.lr_scheduler.LinearLR(
                optimizer_2, start_factor=1, end_factor=0.2, total_iters=28 * iters
            )
        ],
        milestones=[
            2 * iters
        ]
    )

    scaler_2 = torch.cuda.amp.GradScaler()

    path = Path('checkpoints/log')
    path.mkdir(parents=True, exist_ok=True)

    manual_seed()

    log = []

    for epoch in range(90):
        train_1(train_loader, diffuser, model, optimizer_1, scheduler_1, scaler_1, epoch, device)

        if epoch < 30:
            wer = 1.0
        else:
            wer = valid(tokenizer, valid_loader, diffuser, model, epoch, device)

        log.append(wer)

    train_set.logits, train_set.alignments = sample_alignment(tokenizer, train_set.df, model.encoder, device)

    for epoch in range(90, 120):
        train_2(train_loader, diffuser, model, optimizer_2, scheduler_2, scaler_2, epoch, device)

        wer = valid(tokenizer, valid_loader, diffuser, model, epoch, device)

        log.append(wer)

    with open('checkpoints/log/model_log.json', mode='w') as fp:
        json.dump(log, fp, indent=4)



if __name__ == '__main__':
    main()
