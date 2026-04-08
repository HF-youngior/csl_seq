#!/usr/bin/env python3
import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from experiments.seq2seq_ablation.common_args import add_base_train_args
from experiments.seq2seq_ablation.common_logger import append_summary_row, create_logger_and_writer
from experiments.seq2seq_ablation.common_setup import build_dataloaders, prepare_device_and_seed
from experiments.seq2seq_ablation.common_train import run_one_epoch, save_checkpoint
from experiments.seq2seq_ablation.model_b_ctcaux import build_model


def parse_args():
    parser = argparse.ArgumentParser(description="Ablation B: Seq2Seq + lightweight CTC auxiliary loss.")
    add_base_train_args(
        parser=parser,
        default_model_path="./models/ablation_b_ctcaux",
        exp_name="ablation_b_ctcaux",
    )
    parser.add_argument("--ctc_aux_weight", type=float, default=0.2)
    parser.add_argument("--ctc_blank_id", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    device = prepare_device_and_seed(args)
    logger, writer, log_path, run_path, _ = create_logger_and_writer(args.exp_name)

    logger.info("Starting experiment: %s", args.exp_name)
    logger.info("Device: %s", device)
    logger.info("Random seed: %d", args.random_seed)
    logger.info("Unified VAC WER is disabled in this entry.")
    logger.info("CTC aux: weight=%.3f blank_id=%d", args.ctc_aux_weight, args.ctc_blank_id)
    logger.info("Args: %s", vars(args))

    train_set, val_set, test_set, train_loader, val_loader, test_loader = build_dataloaders(args, use_pose=False)
    logger.info("Dataset size: train=%d, val=%d, test=%d", len(train_set), len(val_set), len(test_set))
    logger.info("Output dim: %d", train_set.output_dim)

    model = build_model(
        output_dim=train_set.output_dim,
        enc_hid_dim=args.enc_hid_dim,
        dec_hid_dim=args.dec_hid_dim,
        emb_dim=args.emb_dim,
        dropout=args.dropout,
        device=device,
    )
    if torch.cuda.device_count() > 1:
        logger.info("Using %d GPUs", torch.cuda.device_count())
        model = nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss()
    ctc_loss_fn = nn.CTCLoss(blank=args.ctc_blank_id, zero_infinity=True)
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    best_val_wer = float("inf")
    best_epoch = 0
    final_val_wer = 0.0
    final_test_wer = 0.0

    for epoch in range(args.epochs):
        run_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch_idx=epoch,
            logger=logger,
            writer=writer,
            mode="train",
            log_interval=args.log_interval,
            clip=args.clip,
            teacher_forcing_ratio=args.teacher_forcing_ratio,
            use_pose=False,
            use_ctc_aux=True,
            ctc_loss_fn=ctc_loss_fn,
            ctc_weight=args.ctc_aux_weight,
        )
        val_metrics = run_one_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            optimizer=None,
            device=device,
            epoch_idx=epoch,
            logger=logger,
            writer=writer,
            mode="validation",
            log_interval=args.log_interval,
            clip=args.clip,
            teacher_forcing_ratio=0.0,
            use_pose=False,
            use_ctc_aux=True,
            ctc_loss_fn=ctc_loss_fn,
            ctc_weight=args.ctc_aux_weight,
        )
        final_val_wer = val_metrics["wer"]

        state, ckpt_path = save_checkpoint(
            model=model,
            model_path=args.model_path,
            epoch_idx=epoch,
            save_each_epoch=args.save_each_epoch,
        )
        logger.info("Epoch %d checkpoint saved: %s", epoch + 1, ckpt_path)

        if val_metrics["wer"] < best_val_wer:
            best_val_wer = val_metrics["wer"]
            best_epoch = epoch + 1
            best_path = os.path.join(args.model_path, "best_model.pth")
            torch.save(state, best_path)
            logger.info("Best model updated: %s | val_WER=%.2f%%", best_path, best_val_wer)

    best_path = os.path.join(args.model_path, "best_model.pth")
    if os.path.isfile(best_path):
        best_state = torch.load(best_path, map_location=device)
        if hasattr(model, "module"):
            model.module.load_state_dict(best_state)
        else:
            model.load_state_dict(best_state)
        logger.info("Loaded best checkpoint for final test: %s", best_path)

    test_metrics = run_one_epoch(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        optimizer=None,
        device=device,
        epoch_idx=args.epochs,
        logger=logger,
        writer=writer,
        mode="test",
        log_interval=args.log_interval,
        clip=args.clip,
        teacher_forcing_ratio=0.0,
        use_pose=False,
        use_ctc_aux=True,
        ctc_loss_fn=ctc_loss_fn,
        ctc_weight=args.ctc_aux_weight,
    )
    final_test_wer = test_metrics["wer"]
    logger.info("Final Test WER (best-val checkpoint): %.2f%%", final_test_wer)

    writer.close()

    append_summary_row(
        csv_path=args.results_csv,
        row={
            "exp_name": args.exp_name,
            "seed": args.random_seed,
            "best_epoch": best_epoch,
            "best_val_wer": f"{best_val_wer:.4f}",
            "final_val_wer": f"{final_val_wer:.4f}",
            "test_wer": f"{final_test_wer:.4f}",
            "log_path": log_path,
            "run_path": run_path,
            "model_path": args.model_path,
        },
    )
    logger.info("Finished. Summary appended to %s", args.results_csv)


if __name__ == "__main__":
    main()
