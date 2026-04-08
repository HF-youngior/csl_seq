import csv
import logging
import os
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def create_logger_and_writer(exp_name: str):
    ensure_dir("log")
    ensure_dir("runs")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = os.path.join("log", f"{exp_name}_{timestamp}.log")
    run_path = os.path.join("runs", f"{exp_name}_{timestamp}")

    logger = logging.getLogger(exp_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    formatter = logging.Formatter("%(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    writer = SummaryWriter(run_path)
    return logger, writer, log_path, run_path, timestamp


def _upgrade_summary_schema_if_needed(csv_path: str, fieldnames):
    if not os.path.isfile(csv_path) or os.path.getsize(csv_path) == 0:
        return

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        old_fields = reader.fieldnames or []
        rows = list(reader)

    if old_fields == fieldnames:
        return

    needs_upgrade = any(name not in old_fields for name in fieldnames)
    if not needs_upgrade:
        return

    for row in rows:
        for name in fieldnames:
            row.setdefault(name, "")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def append_summary_row(csv_path: str, row: dict) -> None:
    folder = os.path.dirname(csv_path)
    if folder:
        ensure_dir(folder)

    fieldnames = [
        "exp_name",
        "seed",
        "best_epoch",
        "best_val_wer",
        "final_val_wer",
        "test_wer",
        "log_path",
        "run_path",
        "model_path",
    ]

    _upgrade_summary_schema_if_needed(csv_path, fieldnames)
    write_header = not os.path.isfile(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
