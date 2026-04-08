#!/usr/bin/env python3
import argparse
import os
import sys
from collections import OrderedDict

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def _str2bool(value):
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def _read_corpus_char_vocab(corpus_path):
    if not os.path.isfile(corpus_path):
        raise FileNotFoundError("corpus file not found: {}".format(corpus_path))

    token_to_id = OrderedDict()
    token_to_id["<pad>"] = 0
    token_to_id["<sos>"] = 1
    token_to_id["<eos>"] = 2
    corpus_tokens = OrderedDict()

    with open(corpus_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            sentence_id = parts[0]
            sentence = "".join(parts[1:])
            tokens = [token_to_id["<sos>"]]
            for ch in sentence:
                if ch not in token_to_id:
                    token_to_id[ch] = len(token_to_id)
                tokens.append(token_to_id[ch])
            tokens.append(token_to_id["<eos>"])
            corpus_tokens[sentence_id] = tokens

    if len(corpus_tokens) == 0:
        raise RuntimeError("No valid samples loaded from corpus: {}".format(corpus_path))

    max_len = max(len(v) for v in corpus_tokens.values())
    for sentence_id, tokens in corpus_tokens.items():
        if len(tokens) < max_len:
            corpus_tokens[sentence_id] = tokens + [token_to_id["<pad>"]] * (max_len - len(tokens))

    id_to_token = {idx: token for token, idx in token_to_id.items() if idx >= 3}
    return token_to_id, id_to_token, max_len


def _load_split_info(info_path):
    if not os.path.isfile(info_path):
        raise FileNotFoundError("info npy not found: {}".format(info_path))
    info = np.load(info_path, allow_pickle=True).item()
    sample_ids = sorted([k for k in info.keys() if isinstance(k, int)])
    if len(sample_ids) == 0:
        raise RuntimeError("No integer sample ids found in info file: {}".format(info_path))
    samples = []
    for sid in sample_ids:
        sample = info[sid]
        if "fileid" not in sample or "folder" not in sample:
            raise KeyError("Sample {} missing 'fileid' or 'folder'.".format(sid))
        samples.append(sample)
    return samples


def _sample_video_frames(video_path, target_frames):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    if len(frames) == 0:
        raise RuntimeError("Decoded 0 frames: {}".format(video_path))

    if len(frames) > target_frames:
        sample_idx = np.linspace(0, len(frames) - 1, target_frames).astype(np.int32).tolist()
        frames = [frames[i] for i in sample_idx]
    elif len(frames) < target_frames:
        frames.extend([frames[-1]] * (target_frames - len(frames)))
    return frames


class VACInfoSeq2SeqDataset(Dataset):
    def __init__(self, samples, dataset_root, frames, sample_size):
        self.samples = samples
        self.dataset_root = dataset_root
        self.frames = int(frames)
        self.transform = transforms.Compose(
            [
                transforms.Resize([sample_size, sample_size]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        fileid = str(sample["fileid"])
        rel_path = str(sample["folder"]).replace("/", os.sep)
        video_path = os.path.join(self.dataset_root, rel_path)
        frames = _sample_video_frames(video_path, self.frames)

        tensors = []
        for frame in frames:
            image = Image.fromarray(frame)
            tensors.append(self.transform(image))
        video = torch.stack(tensors, dim=0).permute(1, 0, 2, 3)
        return video, fileid


def _collate_batch(batch):
    videos, fileids = list(zip(*batch))
    return torch.stack(videos, dim=0), list(fileids)


def _strip_module_prefix(state_dict):
    new_state = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith("module."):
            new_state[key[len("module."):]] = value
        else:
            new_state[key] = value
    return new_state


def _load_seq2seq_model(args, vocab_size, device):
    from models.Seq2Seq import Decoder, Encoder, Seq2Seq

    encoder = Encoder(lstm_hidden_size=args.enc_hid_dim, arch=args.encoder_arch).to(device)
    decoder = Decoder(
        output_dim=vocab_size,
        emb_dim=args.emb_dim,
        enc_hid_dim=args.enc_hid_dim,
        dec_hid_dim=args.dec_hid_dim,
        dropout=args.dropout,
    ).to(device)
    model = Seq2Seq(encoder=encoder, decoder=decoder, device=device).to(device)

    raw_state = torch.load(args.checkpoint, map_location=device)
    state_dict = raw_state["model_state_dict"] if isinstance(raw_state, dict) and "model_state_dict" in raw_state else raw_state
    state_dict = _strip_module_prefix(state_dict)
    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Checkpoint incompatible.\nmissing_keys={}\nunexpected_keys={}".format(
                incompatible.missing_keys[:20], incompatible.unexpected_keys[:20]
            )
        )
    model.eval()
    return model


def _ids_to_tokens(ids, id_to_token):
    tokens = []
    for token_id in ids:
        token = id_to_token.get(int(token_id), "")
        if token:
            tokens.append(token)
    return tokens


def _decode_predictions(model, loader, device, decode_len, pad_id, sos_id, id_to_token):
    all_fileids = []
    all_predictions = []
    with torch.no_grad():
        for videos, fileids in tqdm(loader, desc="seq2seq export"):
            videos = videos.to(device, non_blocking=True)
            batch_size = videos.size(0)
            target = torch.full((batch_size, decode_len), pad_id, dtype=torch.long, device=device)
            target[:, 0] = sos_id
            outputs = model(videos, target, teacher_forcing_ratio=0.0)
            pred_ids = outputs.argmax(dim=2).permute(1, 0).cpu().tolist()
            for seq in pred_ids:
                # step-0 output is not produced by decoder loop; drop it.
                cleaned = [int(tid) for tid in seq[1:] if int(tid) not in (0, 1, 2)]
                all_predictions.append(_ids_to_tokens(cleaned, id_to_token))
            all_fileids.extend(fileids)
    return all_fileids, all_predictions


def _write_ctm(output_path, fileids, predictions, empty_token):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for fileid, tokens in zip(fileids, predictions):
            write_tokens = tokens if len(tokens) > 0 else [empty_token]
            for idx, token in enumerate(write_tokens):
                f.write(
                    "{} 1 {:.2f} {:.2f} {}\n".format(
                        fileid,
                        idx * 1.0 / 100.0,
                        (idx + 1) * 1.0 / 100.0,
                        token,
                    )
                )


def _read_key_stats(path):
    key_list = []
    line_count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            line_count += 1
            parts = text.split()
            if parts:
                key_list.append(parts[0])
    key_set = set(key_list)
    return {
        "line_count": line_count,
        "key_count": len(key_list),
        "uniq_key_count": len(key_set),
        "key_set": key_set,
    }


def _resolve_eval_paths(vac_root, eval_dir, eval_gt_dir):
    evaluation_dir = eval_dir
    if not os.path.isabs(evaluation_dir):
        evaluation_dir = os.path.join(vac_root, evaluation_dir)
    evaluation_gt_dir = eval_gt_dir
    if not os.path.isabs(evaluation_gt_dir):
        evaluation_gt_dir = os.path.join(vac_root, evaluation_gt_dir)
    return os.path.abspath(evaluation_dir), os.path.abspath(evaluation_gt_dir)


def _check_key_alignment(ctm_path, split, evaluation_prefix, evaluation_gt_dir):
    gt_path = os.path.join(evaluation_gt_dir, "{}-{}.stm".format(evaluation_prefix, split))
    if not os.path.isfile(gt_path):
        raise FileNotFoundError("Groundtruth STM not found: {}".format(gt_path))
    ctm_stats = _read_key_stats(ctm_path)
    stm_stats = _read_key_stats(gt_path)
    print(
        "Eval key check [{}]: GT={}, CTM lines={}, CTM keys={}, CTM uniq_keys={}, "
        "STM lines={}, STM keys={}, STM uniq_keys={}".format(
            split,
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
        raise RuntimeError(
            "CTM/STM key mismatch. only_in_ctm={} only_in_stm={} sample_ctm={} sample_stm={}".format(
                len(only_ctm), len(only_stm), only_ctm[:10], only_stm[:10]
            )
        )
    return gt_path


def _run_vac_eval(ctm_path, split, evaluation_dir, evaluation_prefix, evaluation_gt_dir):
    vac_root = os.path.abspath(os.path.join(evaluation_dir, os.pardir, os.pardir))
    if vac_root not in sys.path:
        sys.path.insert(0, vac_root)
    from evaluation.slr_eval.wer_calculation import evaluate as vac_evaluate

    output_file = os.path.basename(ctm_path)
    work_prefix = os.path.join(os.path.dirname(ctm_path), "")
    wer_value = vac_evaluate(
        prefix=work_prefix,
        mode=split,
        evaluate_dir=evaluation_dir,
        evaluate_prefix=evaluation_prefix,
        output_file=output_file,
        output_dir=None,
        python_evaluate=True,
        token_unit="char",
        groundtruth_dir=evaluation_gt_dir,
    )
    print("VAC unified WER [{}]: {:0.2f}%".format(split, wer_value))
    return wer_value


def parse_args():
    parser = argparse.ArgumentParser(description="Export Seq2Seq predictions as VAC-compatible CTM.")
    parser.add_argument("--checkpoint", type=str, required=True, help="path to seq2seq .pth checkpoint")
    parser.add_argument("--dataset-root", type=str, required=True, help="CSL color dataset root")
    parser.add_argument("--corpus-path", type=str, required=True, help="corpus.txt path")
    parser.add_argument("--split", type=str, default="dev", choices=["train", "dev", "test", "dev_fast"])
    parser.add_argument("--vac-root", type=str, default="", help="VAC_CSLR-main root")
    parser.add_argument("--info-path", type=str, default="", help="override split info npy path")
    parser.add_argument("--output-ctm", type=str, default="", help="output ctm path")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-worker", type=int, default=2)
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--sample-size", type=int, default=128)
    parser.add_argument("--encoder-arch", type=str, default="resnet18")
    parser.add_argument("--enc-hid-dim", type=int, default=512)
    parser.add_argument("--dec-hid-dim", type=int, default=512)
    parser.add_argument("--emb-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--max-decode-len", type=int, default=0, help="0 means use corpus max length")
    parser.add_argument("--empty-token", type=str, default="<eps>")
    parser.add_argument("--run-eval", type=_str2bool, default=False, help="run VAC WER after export")
    parser.add_argument("--evaluation-dir", type=str, default="./evaluation/slr_eval")
    parser.add_argument("--evaluation-prefix", type=str, default="csl100-groundtruth")
    parser.add_argument("--evaluation-gt-dir", type=str, default="./csl100")
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    vac_root = os.path.abspath(args.vac_root) if args.vac_root else os.path.join(root_dir, "VAC_CSLR-main")
    if not os.path.isdir(vac_root):
        raise NotADirectoryError("VAC root does not exist: {}".format(vac_root))

    info_path = args.info_path.strip()
    if not info_path:
        info_path = os.path.join(vac_root, "csl100", "{}_info.npy".format(args.split))
    info_path = os.path.abspath(info_path)

    output_ctm = args.output_ctm.strip()
    if not output_ctm:
        output_ctm = os.path.join(
            vac_root,
            "work_dir",
            "seq2seq_unified_eval",
            "output-hypothesis-{}.ctm".format(args.split),
        )
    output_ctm = os.path.abspath(output_ctm)

    evaluation_dir, evaluation_gt_dir = _resolve_eval_paths(
        vac_root, args.evaluation_dir, args.evaluation_gt_dir
    )

    os.makedirs(os.path.dirname(output_ctm), exist_ok=True)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    print("checkpoint={}".format(os.path.abspath(args.checkpoint)))
    print("dataset_root={}".format(os.path.abspath(args.dataset_root)))
    print("corpus_path={}".format(os.path.abspath(args.corpus_path)))
    print("split={}".format(args.split))
    print("info_path={}".format(info_path))
    print("output_ctm={}".format(output_ctm))
    print("device={}".format(device))

    token_to_id, id_to_token, corpus_max_len = _read_corpus_char_vocab(os.path.abspath(args.corpus_path))
    decode_len = int(args.max_decode_len) if int(args.max_decode_len) > 0 else int(corpus_max_len)
    samples = _load_split_info(info_path)
    dataset = VACInfoSeq2SeqDataset(
        samples=samples,
        dataset_root=os.path.abspath(args.dataset_root),
        frames=int(args.frames),
        sample_size=int(args.sample_size),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_worker),
        pin_memory=device.type == "cuda",
        collate_fn=_collate_batch,
    )

    sys.path.insert(0, root_dir)
    sys.path.insert(0, vac_root)
    model = _load_seq2seq_model(args, vocab_size=len(token_to_id), device=device)
    fileids, predictions = _decode_predictions(
        model=model,
        loader=loader,
        device=device,
        decode_len=decode_len,
        pad_id=token_to_id["<pad>"],
        sos_id=token_to_id["<sos>"],
        id_to_token=id_to_token,
    )
    _write_ctm(output_ctm, fileids, predictions, args.empty_token)
    print("CTM exported: {} (samples={})".format(output_ctm, len(fileids)))

    _check_key_alignment(
        ctm_path=output_ctm,
        split=args.split,
        evaluation_prefix=args.evaluation_prefix,
        evaluation_gt_dir=evaluation_gt_dir,
    )

    if _str2bool(args.run_eval):
        _run_vac_eval(
            ctm_path=output_ctm,
            split=args.split,
            evaluation_dir=evaluation_dir,
            evaluation_prefix=args.evaluation_prefix,
            evaluation_gt_dir=evaluation_gt_dir,
        )


if __name__ == "__main__":
    main()
