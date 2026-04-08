import os

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .common_metrics import (
    clean_target_for_ctc,
    compute_batch_acc,
    compute_batch_wer,
    compute_epoch_acc,
)


def _forward_model(model, imgs, target, pose_feat, teacher_forcing_ratio, use_pose, use_ctc_aux):
    if use_ctc_aux:
        outputs, ctc_logits = model(imgs, target, teacher_forcing_ratio=teacher_forcing_ratio)
        return outputs, ctc_logits
    if use_pose:
        outputs = model(
            imgs,
            target,
            pose_feat=pose_feat,
            teacher_forcing_ratio=teacher_forcing_ratio,
        )
        return outputs, None
    outputs = model(imgs, target, teacher_forcing_ratio=teacher_forcing_ratio)
    return outputs, None


def run_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    epoch_idx,
    logger,
    writer,
    mode="train",
    log_interval=100,
    clip=1.0,
    teacher_forcing_ratio=0.5,
    use_pose=False,
    use_ctc_aux=False,
    ctc_loss_fn=None,
    ctc_weight=0.0,
):
    is_train = mode == "train"
    model.train() if is_train else model.eval()

    losses = []
    ctc_losses = []
    all_targets = []
    all_preds = []
    all_wers = []

    iterator = tqdm(
        enumerate(dataloader),
        total=len(dataloader),
        desc=f"{mode.capitalize()} Epoch {epoch_idx + 1:03d}",
        leave=True,
        dynamic_ncols=True,
    )

    for batch_idx, batch in iterator:
        imgs, target, _, pose_feat = batch
        imgs = imgs.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        pose_feat = pose_feat.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            outputs, ctc_logits = _forward_model(
                model=model,
                imgs=imgs,
                target=target,
                pose_feat=pose_feat,
                teacher_forcing_ratio=teacher_forcing_ratio if is_train else 0.0,
                use_pose=use_pose,
                use_ctc_aux=use_ctc_aux,
            )

            output_dim = outputs.shape[-1]
            outputs_flat = outputs[1:].reshape(-1, output_dim)
            target_flat = target.permute(1, 0)[1:].reshape(-1)
            ce_loss = criterion(outputs_flat, target_flat)
            total_loss = ce_loss

            ctc_loss_val = None
            if use_ctc_aux:
                log_probs = ctc_logits.log_softmax(-1).permute(1, 0, 2)
                input_lengths = torch.full(
                    (imgs.size(0),),
                    fill_value=ctc_logits.size(1),
                    dtype=torch.long,
                    device=device,
                )
                ctc_targets, target_lengths = clean_target_for_ctc(target)
                ctc_loss_val = ctc_loss_fn(log_probs, ctc_targets, input_lengths, target_lengths)
                total_loss = total_loss + float(ctc_weight) * ctc_loss_val
                ctc_losses.append(float(ctc_loss_val.detach().cpu().item()))

            if is_train:
                total_loss.backward()
                if clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()

        pred_flat = outputs_flat.argmax(dim=1)
        score = compute_batch_acc(target_flat, pred_flat)
        batch_wer, wers = compute_batch_wer(target_flat, pred_flat, imgs.size(0))

        losses.append(float(total_loss.detach().cpu().item()))
        all_targets.append(target_flat.detach().cpu())
        all_preds.append(pred_flat.detach().cpu())
        all_wers.extend(wers)

        if (batch_idx + 1) % log_interval == 0:
            if use_ctc_aux and ctc_loss_val is not None:
                logger.info(
                    "epoch %3d | iteration %5d | Loss %.6f | CTC %.6f | Acc %.2f%% | WER %.2f%%",
                    epoch_idx + 1,
                    batch_idx + 1,
                    float(total_loss.detach().cpu().item()),
                    float(ctc_loss_val.detach().cpu().item()),
                    score * 100,
                    batch_wer,
                )
            else:
                logger.info(
                    "epoch %3d | iteration %5d | Loss %.6f | Acc %.2f%% | WER %.2f%%",
                    epoch_idx + 1,
                    batch_idx + 1,
                    float(total_loss.detach().cpu().item()),
                    score * 100,
                    batch_wer,
                )

        iterator.set_postfix(
            loss="{:.4f}".format(float(total_loss.detach().cpu().item())),
            acc="{:.2f}%".format(score * 100),
            wer="{:.2f}%".format(batch_wer),
        )

    epoch_loss = float(np.mean(losses)) if losses else 0.0
    epoch_acc = float(compute_epoch_acc(all_targets, all_preds))
    epoch_wer = float(np.mean(all_wers)) if all_wers else 0.0
    epoch_ctc = float(np.mean(ctc_losses)) if ctc_losses else 0.0

    writer.add_scalars("Loss", {mode: epoch_loss}, epoch_idx + 1)
    writer.add_scalars("Accuracy", {mode: epoch_acc}, epoch_idx + 1)
    writer.add_scalars("WER", {mode: epoch_wer}, epoch_idx + 1)
    if use_ctc_aux:
        writer.add_scalars("Loss_CTC", {mode: epoch_ctc}, epoch_idx + 1)

    if use_ctc_aux:
        logger.info(
            "Average %s Loss of Epoch %d: %.6f | CTC: %.6f | Acc: %.2f%% | WER %.2f%%",
            mode.capitalize(),
            epoch_idx + 1,
            epoch_loss,
            epoch_ctc,
            epoch_acc * 100,
            epoch_wer,
        )
    else:
        logger.info(
            "Average %s Loss of Epoch %d: %.6f | Acc: %.2f%% | WER %.2f%%",
            mode.capitalize(),
            epoch_idx + 1,
            epoch_loss,
            epoch_acc * 100,
            epoch_wer,
        )

    return {
        "loss": epoch_loss,
        "acc": epoch_acc,
        "wer": epoch_wer,
        "ctc_loss": epoch_ctc,
    }


def save_checkpoint(model, model_path, epoch_idx, save_each_epoch=True):
    os.makedirs(model_path, exist_ok=True)
    state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    last_path = os.path.join(model_path, f"continuous_slr_epoch{epoch_idx + 1:03d}.pth")
    if save_each_epoch:
        torch.save(state, last_path)
    return state, last_path

