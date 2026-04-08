import os
import pdb
import sys
import copy
import time
import traceback
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from evaluation.slr_eval.wer_calculation import evaluate


def _format_float(value):
    if value is None or np.isnan(value):
        return "nan"
    return "{:.8f}".format(value)


def _resolve_gt_path(dataset_info, mode):
    gt_file = "{}-{}.stm".format(dataset_info["evaluation_prefix"], mode)
    candidates = []
    gt_dir = dataset_info.get("evaluation_gt_dir", None)
    if gt_dir:
        candidates.append(os.path.join(gt_dir, gt_file))
    candidates.append(os.path.join(dataset_info["evaluation_dir"], gt_file))
    repo_root = os.path.abspath(os.path.dirname(__file__))
    candidates.append(os.path.join(repo_root, "csl100", gt_file))

    seen = set()
    for candidate in candidates:
        abs_path = os.path.abspath(candidate)
        if abs_path in seen:
            continue
        seen.add(abs_path)
        if os.path.isfile(abs_path):
            return abs_path
    raise FileNotFoundError(
        "Groundtruth file not found for mode '{}'. Checked: {}".format(mode, sorted(seen))
    )


def _read_key_stats(path):
    keys = []
    line_count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            line_count += 1
            parts = stripped.split()
            if parts:
                keys.append(parts[0])
    key_set = set(keys)
    return {
        "line_count": line_count,
        "key_count": len(keys),
        "uniq_key_count": len(key_set),
        "key_set": key_set,
    }


def _check_eval_key_alignment(recoder, dataset_info, mode, ctm_path):
    gt_path = _resolve_gt_path(dataset_info, mode)
    ctm_stats = _read_key_stats(ctm_path)
    stm_stats = _read_key_stats(gt_path)

    recoder.print_log(
        "Eval key check [{}]: GT={}, CTM lines={}, CTM keys={}, CTM uniq_keys={}, "
        "STM lines={}, STM keys={}, STM uniq_keys={}".format(
            mode,
            gt_path,
            ctm_stats["line_count"],
            ctm_stats["key_count"],
            ctm_stats["uniq_key_count"],
            stm_stats["line_count"],
            stm_stats["key_count"],
            stm_stats["uniq_key_count"],
        )
    )

    only_ctm = sorted(ctm_stats["key_set"] - stm_stats["key_set"])
    only_stm = sorted(stm_stats["key_set"] - ctm_stats["key_set"])
    if only_ctm or only_stm:
        recoder.print_log(
            "Eval key mismatch [{}]: only_in_ctm={}, only_in_stm={}".format(
                mode, len(only_ctm), len(only_stm)
            )
        )
        if only_ctm:
            recoder.print_log("only_in_ctm sample: {}".format(", ".join(only_ctm[:20])))
        if only_stm:
            recoder.print_log("only_in_stm sample: {}".format(", ".join(only_stm[:20])))
        raise RuntimeError("CTM/STM key mismatch for mode '{}'; abort evaluation.".format(mode))


def seq_train(loader, model, optimizer, device, epoch_idx, recoder, train_cfg=None):
    model.train()
    train_cfg = train_cfg or {}
    clip_grad = float(train_cfg.get("clip_grad", 5.0))
    invalid_warn_ratio = float(train_cfg.get("invalid_warn_ratio", 0.05))
    max_valid_loss = float(train_cfg.get("max_valid_loss", 1e4))
    min_valid_loss = float(train_cfg.get("min_valid_loss", -1e-3))
    sanitize_grad = bool(train_cfg.get("sanitize_grad", True))

    loss_value = []
    valid_batch_count = 0
    invalid_batch_count = 0
    invalid_grad_batch_count = 0
    sanitized_grad_batch_count = 0
    zero_loss_batch_count = 0
    clr = [group['lr'] for group in optimizer.optimizer.param_groups]
    batch_time_window = []
    for batch_idx, data in enumerate(tqdm(loader)):
        batch_start_time = time.time()
        # Backward-compatible zero_grad for custom optimizer wrappers / older torch.
        try:
            optimizer.zero_grad(set_to_none=True)
        except TypeError:
            optimizer.zero_grad()
        vid = device.data_to_device(data[0])
        vid_lgt = device.data_to_device(data[1])
        label = device.data_to_device(data[2])
        label_lgt = device.data_to_device(data[3])
        ret_dict = model(vid, vid_lgt, label=label, label_lgt=label_lgt)
        loss = model.criterion_calculation(ret_dict, label, label_lgt)
        # Evaluate loss validity on CPU scalar to avoid device-side numeric quirks.
        loss_scalar = float(loss.detach().cpu().item())
        invalid_reason = None
        if not np.isfinite(loss_scalar):
            invalid_reason = "non-finite"
        elif loss_scalar < min_valid_loss:
            invalid_reason = "negative"
        elif loss_scalar > max_valid_loss:
            invalid_reason = "too-large"

        if invalid_reason is not None:
            invalid_batch_count += 1
            recoder.print_log(
                '\tEpoch: {}, Batch({}/{}) invalid loss ({:.8f}, reason={}) - skip this batch.'
                .format(epoch_idx, batch_idx, len(loader), loss_scalar, invalid_reason)
            )
            del ret_dict, loss, vid, vid_lgt, label, label_lgt
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            batch_time_window.append(time.time() - batch_start_time)
            continue
        if abs(loss_scalar) < 1e-12:
            zero_loss_batch_count += 1
        loss.backward()
        had_non_finite_grad = False
        for p in model.parameters():
            if p.grad is None:
                continue
            finite_mask = torch.isfinite(p.grad)
            if not finite_mask.all():
                had_non_finite_grad = True
                if sanitize_grad:
                    p.grad.data = torch.nan_to_num(p.grad.data, nan=0.0, posinf=0.0, neginf=0.0)
                else:
                    invalid_grad_batch_count += 1
                    invalid_batch_count += 1
                    recoder.print_log(
                        '\tEpoch: {}, Batch({}/{}) invalid gradient (non-finite) - skip optimizer step.'
                        .format(epoch_idx, batch_idx, len(loader))
                    )
                    optimizer.zero_grad()
                    break
        else:
            if had_non_finite_grad and sanitize_grad:
                sanitized_grad_batch_count += 1
                recoder.print_log(
                    '\tEpoch: {}, Batch({}/{}) gradient had non-finite values -> sanitized to 0.'
                    .format(epoch_idx, batch_idx, len(loader))
                )
            grad_finite = True
            for p in model.parameters():
                if p.grad is None:
                    continue
                if not torch.isfinite(p.grad).all():
                    grad_finite = False
                    break
            if not grad_finite:
                invalid_grad_batch_count += 1
                invalid_batch_count += 1
                recoder.print_log(
                    '\tEpoch: {}, Batch({}/{}) invalid gradient (still non-finite after sanitize) - skip optimizer step.'
                    .format(epoch_idx, batch_idx, len(loader))
                )
                optimizer.zero_grad()
                continue
            if clip_grad > 0:
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
            valid_batch_count += 1
            loss_value.append(loss_scalar)
            if batch_idx % recoder.log_interval == 0:
                batch_time_window.append(time.time() - batch_start_time)
                avg_batch_time = float(np.mean(batch_time_window[-max(1, recoder.log_interval):]))
                recoder.print_log(
                    '\tEpoch: {}, Batch({}/{}) done. Loss: {:.8f}  lr:{:.6f}  batch_time:{:.3f}s'
                        .format(epoch_idx, batch_idx, len(loader), loss_scalar, clr[0], avg_batch_time))
            else:
                batch_time_window.append(time.time() - batch_start_time)
            continue
    optimizer.scheduler.step()

    invalid_ratio = invalid_batch_count / max(1, len(loader))
    if len(loss_value) > 0:
        losses = np.asarray(loss_value, dtype=np.float64)
        finite_losses = losses[np.isfinite(losses)]
        if len(finite_losses) > 0:
            mean_loss = float(np.mean(finite_losses))
            median_loss = float(np.median(finite_losses))
            p95_loss = float(np.percentile(finite_losses, 95))
        else:
            mean_loss = np.nan
            median_loss = np.nan
            p95_loss = np.nan
    else:
        mean_loss = np.nan
        median_loss = np.nan
        p95_loss = np.nan

    recoder.print_log(
        '\tEpoch {} stats: valid_batches={}, invalid_ctc_batches={}, invalid_ratio={:.2%}, '
        'zero_loss_batches={}, sanitized_grad_batches={}, invalid_grad_batches={}, '
        'valid_loss_mean={}, valid_loss_median={}, valid_loss_p95={}.'
        .format(
            epoch_idx,
            valid_batch_count,
            invalid_batch_count,
            invalid_ratio,
            zero_loss_batch_count,
            sanitized_grad_batch_count,
            invalid_grad_batch_count,
            _format_float(mean_loss),
            _format_float(median_loss),
            _format_float(p95_loss),
        )
    )
    if invalid_ratio > invalid_warn_ratio:
        recoder.print_log(
            '\tWarning: invalid batch ratio {:.2%} is above threshold {:.2%}.'
            .format(invalid_ratio, invalid_warn_ratio)
        )
    recoder.print_log('\tMean training loss: {}.'.format(_format_float(mean_loss)))
    return loss_value


def seq_eval(cfg, loader, model, device, mode, epoch, work_dir, recoder,
             evaluate_tool="python", report_auxiliary=False):
    model.eval()
    total_sent = []
    total_info = []
    total_conv_sent = []
    stat = {i: [0, 0] for i in range(len(loader.dataset.dict))}
    for batch_idx, data in enumerate(tqdm(loader)):
        recoder.record_timer("device")
        vid = device.data_to_device(data[0])
        vid_lgt = device.data_to_device(data[1])
        label = device.data_to_device(data[2])
        label_lgt = device.data_to_device(data[3])
        with torch.no_grad():
            ret_dict = model(
                vid,
                vid_lgt,
                label=label,
                label_lgt=label_lgt,
                return_conv_decode=bool(report_auxiliary),
            )

        total_info += [file_name.split("|")[0] for file_name in data[-1]]
        total_sent += ret_dict['recognized_sents']
        if report_auxiliary:
            total_conv_sent += ret_dict['conv_sents']
    python_eval = True if evaluate_tool == "python" else False
    token_unit = "auto"
    groundtruth_dir = None
    if hasattr(cfg, "eval"):
        token_unit = cfg.eval.get("token_unit", "auto")
    if hasattr(cfg, "dataset_info"):
        groundtruth_dir = cfg.dataset_info.get("evaluation_gt_dir", None)
    work_prefix = os.path.join(work_dir, "")
    hypothesis_path = os.path.join(work_dir, "output-hypothesis-{}.ctm".format(mode))
    write2file(hypothesis_path, total_info, total_sent)
    if report_auxiliary:
        conv_hypothesis_path = os.path.join(work_dir, "output-hypothesis-{}-conv.ctm".format(mode))
        write2file(conv_hypothesis_path, total_info, total_conv_sent)
    _check_eval_key_alignment(recoder, cfg.dataset_info, mode, hypothesis_path)
    try:
        if report_auxiliary:
            evaluate(
                prefix=work_prefix, mode=mode, output_file="output-hypothesis-{}-conv.ctm".format(mode),
                evaluate_dir=cfg.dataset_info['evaluation_dir'],
                evaluate_prefix=cfg.dataset_info['evaluation_prefix'],
                output_dir="epoch_{}_result/".format(epoch),
                python_evaluate=python_eval,
                token_unit=token_unit,
                groundtruth_dir=groundtruth_dir,
            )
        lstm_ret = evaluate(
            prefix=work_prefix, mode=mode, output_file="output-hypothesis-{}.ctm".format(mode),
            evaluate_dir=cfg.dataset_info['evaluation_dir'],
            evaluate_prefix=cfg.dataset_info['evaluation_prefix'],
            output_dir="epoch_{}_result/".format(epoch),
            python_evaluate=python_eval,
            triplet=bool(report_auxiliary),
            token_unit=token_unit,
            groundtruth_dir=groundtruth_dir,
        )
    except Exception:
        recoder.print_log("Evaluation failed:\n{}".format(traceback.format_exc()))
        raise
    recoder.print_log(f"Epoch {epoch}, {mode} {lstm_ret: 2.2f}%", os.path.join(work_dir, "{}.txt".format(mode)))
    return lstm_ret


def seq_feature_generation(loader, model, device, mode, work_dir, recoder):
    model.eval()

    src_path = os.path.abspath(os.path.join(work_dir, mode))
    tgt_path = os.path.abspath(f"./features/{mode}")
    if not os.path.exists("./features/"):
    	os.makedirs("./features/")

    if os.path.islink(tgt_path):
        curr_path = os.readlink(tgt_path)
        if work_dir[1:] in curr_path and os.path.isabs(curr_path):
            return
        else:
            os.unlink(tgt_path)
    else:
        if os.path.exists(src_path) and len(loader.dataset) == len(os.listdir(src_path)):
            os.symlink(src_path, tgt_path)
            return

    for batch_idx, data in tqdm(enumerate(loader)):
        recoder.record_timer("device")
        vid = device.data_to_device(data[0])
        vid_lgt = device.data_to_device(data[1])
        with torch.no_grad():
            ret_dict = model(vid, vid_lgt, return_conv_decode=False)
        if not os.path.exists(src_path):
            os.makedirs(src_path)
        start = 0
        for sample_idx in range(len(vid)):
            end = start + data[3][sample_idx]
            filename = f"{src_path}/{data[-1][sample_idx].split('|')[0]}_features.npy"
            save_file = {
                "label": data[2][start:end],
                "features": ret_dict['framewise_features'][sample_idx][:, :vid_lgt[sample_idx]].T.cpu().detach(),
            }
            np.save(filename, save_file)
            start = end
        assert end == len(data[2])
    os.symlink(src_path, tgt_path)


def write2file(path, info, output):
    with open(path, "w", encoding="utf-8") as filereader:
        for sample_idx, sample in enumerate(output):
            for word_idx, word in enumerate(sample):
                filereader.writelines(
                    "{} 1 {:.2f} {:.2f} {}\n".format(info[sample_idx],
                                                     word_idx * 1.0 / 100,
                                                     (word_idx + 1) * 1.0 / 100,
                                                     word[0]))
