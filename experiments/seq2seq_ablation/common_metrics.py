import numpy as np
import torch
from sklearn.metrics import accuracy_score

from tools import wer


SPECIAL_TOKEN_IDS = {0, 1, 2}


def compute_batch_acc(target_flat: torch.Tensor, pred_flat: torch.Tensor) -> float:
    return accuracy_score(
        target_flat.detach().cpu().numpy(),
        pred_flat.detach().cpu().numpy(),
    )


def compute_batch_wer(target_flat: torch.Tensor, pred_flat: torch.Tensor, batch_size: int):
    pred_seq = pred_flat.view(-1, batch_size).permute(1, 0).tolist()
    tgt_seq = target_flat.view(-1, batch_size).permute(1, 0).tolist()
    wers = []
    for i in range(batch_size):
        pred_clean = [x for x in pred_seq[i] if x not in SPECIAL_TOKEN_IDS]
        tgt_clean = [x for x in tgt_seq[i] if x not in SPECIAL_TOKEN_IDS]
        if len(tgt_clean) == 0:
            continue
        wers.append(wer(tgt_clean, pred_clean))
    if not wers:
        return 0.0, wers
    return float(np.mean(wers)), wers


def compute_epoch_acc(all_targets, all_preds) -> float:
    if len(all_targets) == 0:
        return 0.0
    target = torch.cat(all_targets, dim=0).detach().cpu().numpy()
    pred = torch.cat(all_preds, dim=0).detach().cpu().numpy()
    return accuracy_score(target, pred)


def clean_target_for_ctc(target_batch: torch.Tensor):
    target_lengths = []
    non_empty_targets = []
    for sample in target_batch:
        cleaned = sample[(sample != 0) & (sample != 1) & (sample != 2)]
        target_lengths.append(int(cleaned.numel()))
        if cleaned.numel() > 0:
            non_empty_targets.append(cleaned)
    target_lengths = torch.tensor(target_lengths, dtype=torch.long, device=target_batch.device)
    if non_empty_targets:
        ctc_targets = torch.cat(non_empty_targets, dim=0)
    else:
        ctc_targets = torch.empty((0,), dtype=torch.long, device=target_batch.device)
    return ctc_targets, target_lengths

