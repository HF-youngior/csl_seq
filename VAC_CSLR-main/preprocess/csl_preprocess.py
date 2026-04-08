import argparse
import os
import random
import re
from collections import OrderedDict, defaultdict

import cv2
import numpy as np
from tqdm import tqdm


VIDEO_NAME_PATTERN = re.compile(
    r"^(?P<signer>P\d{2})_s\d+_\d+_(?P<repeat>\d+)_color\.avi$",
    re.IGNORECASE,
)


def str2bool(value):
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def load_corpus(corpus_path):
    if not os.path.isfile(corpus_path):
        raise FileNotFoundError("corpus.txt not found: {}".format(corpus_path))
    corpus = {}
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            sent_id = parts[0]
            label = "".join(parts[1:])
            if sent_id in corpus:
                raise ValueError("Duplicate sentence id '{}' at line {}".format(sent_id, line_idx))
            corpus[sent_id] = label
    if len(corpus) == 0:
        raise ValueError("No valid sentence found in corpus: {}".format(corpus_path))
    return corpus


def parse_video_name(filename):
    matched = VIDEO_NAME_PATTERN.match(filename)
    if not matched:
        return None
    signer = matched.group("signer").upper()
    repeat = int(matched.group("repeat"))
    return signer, repeat


def to_char_spaced_label(label):
    return " ".join([ch for ch in label if ch.strip()])


def scan_dataset(dataset_root, corpus):
    sentence_dirs = sorted(
        [d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d))]
    )
    samples_by_sentence = OrderedDict()
    bad_samples = []

    for sentence_id in tqdm(sentence_dirs, desc="scan dataset"):
        if sentence_id not in corpus:
            bad_samples.append("Sentence {} not found in corpus".format(sentence_id))
            continue
        sentence_dir = os.path.join(dataset_root, sentence_id)
        video_files = sorted([f for f in os.listdir(sentence_dir) if f.lower().endswith(".avi")])
        label = corpus[sentence_id]
        sentence_samples = []

        for video_file in video_files:
            parsed = parse_video_name(video_file)
            if parsed is None:
                bad_samples.append("Invalid video name: {}/{}".format(sentence_id, video_file))
                continue
            signer, repeat = parsed
            rel_path = "{}/{}".format(sentence_id, video_file)
            abs_path = os.path.join(dataset_root, rel_path)
            if not os.path.isfile(abs_path):
                bad_samples.append("Missing video file: {}".format(abs_path))
                continue
            fileid = "{}_{}_{}".format(sentence_id, signer, repeat)
            sentence_samples.append(
                {
                    "sentence_id": sentence_id,
                    "fileid": fileid,
                    "folder": rel_path,
                    "signer": signer,
                    "label": label,
                    "num_frames": int(0),
                    "original_info": "{}|{}|{}|{}".format(fileid, rel_path, signer, label),
                }
            )
        samples_by_sentence[sentence_id] = sentence_samples
    return samples_by_sentence, bad_samples


def split_signer_holdout_5000x2(samples_by_sentence):
    train, dev, test = [], [], []
    for sentence_id in sorted(samples_by_sentence.keys()):
        for sample in samples_by_sentence[sentence_id]:
            signer_num = int(sample["signer"][1:])
            if signer_num <= 40:
                train.append(sample)
            else:
                dev.append(sample)
                test.append(sample)
    return train, dev, test


def split_paper_random_8_2_seeded(samples_by_sentence, split_seed, eval_pool_split="mirror"):
    train, dev, test = [], [], []
    for sentence_id in sorted(samples_by_sentence.keys()):
        sentence_samples = list(samples_by_sentence[sentence_id])
        if len(sentence_samples) != 250:
            raise RuntimeError(
                "paper_random_8_2_seeded expects 250 videos per sentence. "
                "Found {} for sentence {}.".format(len(sentence_samples), sentence_id)
            )
        rng = random.Random(split_seed + int(sentence_id))
        rng.shuffle(sentence_samples)
        train.extend(sentence_samples[:200])
        eval_pool = sentence_samples[200:250]
        if eval_pool_split == "mirror":
            dev.extend(eval_pool)
            test.extend(eval_pool)
        elif eval_pool_split == "disjoint_half":
            if len(eval_pool) != 50:
                raise RuntimeError(
                    "disjoint_half expects 50 eval samples per sentence, got {} for {}.".format(
                        len(eval_pool), sentence_id
                    )
                )
            dev.extend(eval_pool[:25])
            test.extend(eval_pool[25:50])
        else:
            raise ValueError("Unsupported eval_pool_split: {}".format(eval_pool_split))
    return train, dev, test


def build_dev_fast(dev_samples, per_sentence=10, split_seed=42):
    grouped = defaultdict(list)
    for sample in dev_samples:
        grouped[sample["sentence_id"]].append(sample)

    dev_fast = []
    for sentence_id in sorted(grouped.keys()):
        sentence_group = list(grouped[sentence_id])
        if len(sentence_group) < per_sentence:
            raise RuntimeError(
                "dev_fast requires at least {} samples per sentence, got {} for sentence {}.".format(
                    per_sentence, len(sentence_group), sentence_id
                )
            )
        rng = random.Random(split_seed * 13 + int(sentence_id))
        rng.shuffle(sentence_group)
        dev_fast.extend(sentence_group[:per_sentence])
    return dev_fast


def list_to_info_dict(samples):
    info_dict = OrderedDict()
    for idx, sample in enumerate(samples):
        info_dict[idx] = {
            "fileid": sample["fileid"],
            "folder": sample["folder"],
            "signer": sample["signer"],
            "label": sample["label"],
            "num_frames": int(sample.get("num_frames", 0)),
            "original_info": "{}|{}|{}|{}".format(
                sample["fileid"],
                sample["folder"],
                sample["signer"],
                sample["label"],
            ),
        }
    return info_dict


def generate_gt_stm(info, save_path):
    with open(save_path, "w", encoding="utf-8") as f:
        for _, sample in info.items():
            char_label = to_char_spaced_label(sample["label"])
            f.write(
                "{} 1 {} 0.0 1.79769e+308 {}\n".format(
                    sample["fileid"],
                    sample["signer"],
                    char_label,
                )
            )


def build_gloss_dict(train_info):
    stats = {}
    for _, sample in train_info.items():
        for ch in sample["label"]:
            if not ch.strip():
                continue
            stats[ch] = stats.get(ch, 0) + 1
    gloss_dict = {}
    for idx, ch in enumerate(sorted(stats.keys()), start=1):
        gloss_dict[ch] = [idx, stats[ch]]
    return gloss_dict


def save_bad_samples(output_dir, mode, bad_samples):
    if len(bad_samples) == 0:
        return
    bad_path = os.path.join(output_dir, "{}_bad_samples.txt".format(mode))
    with open(bad_path, "w", encoding="utf-8") as f:
        for line in bad_samples:
            f.write(line + "\n")


def sample_video_frames(video_path, target_frames):
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
        sample_index = np.linspace(0, len(frames) - 1, target_frames).astype(np.int32).tolist()
        frames = [frames[i] for i in sample_index]
    elif len(frames) < target_frames:
        frames.extend([frames[-1]] * (target_frames - len(frames)))
    return np.asarray(frames, dtype=np.uint8)


def export_frame_cache(samples, dataset_root, cache_dir, cache_frames, strict_exists=True, overwrite=False):
    unique_samples = OrderedDict()
    for sample in samples:
        unique_samples[sample["fileid"]] = sample

    os.makedirs(cache_dir, exist_ok=True)
    failures = []
    cached = 0
    skipped = 0

    for fileid, sample in tqdm(unique_samples.items(), desc="export frame cache"):
        cache_path = os.path.join(cache_dir, "{}.npy".format(fileid))
        if os.path.isfile(cache_path) and not overwrite:
            skipped += 1
            continue
        video_path = os.path.join(dataset_root, sample["folder"].replace("/", os.sep))
        try:
            frames = sample_video_frames(video_path, cache_frames)
            np.save(cache_path, frames)
            cached += 1
        except Exception as exc:
            failures.append("cache export failed {}: {}".format(fileid, exc))

    if failures and strict_exists:
        preview = "\n".join(failures[:20])
        raise RuntimeError(
            "Frame cache export failed on {} samples.\n{}".format(len(failures), preview)
        )
    print("cache export: cached={}, skipped_existing={}, failed={}".format(cached, skipped, len(failures)))
    return failures


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess CSL100 for VAC-CSLR by scanning real video files."
    )
    parser.add_argument("--dataset", type=str, default="csl100", help="output prefix")
    parser.add_argument(
        "--dataset-root",
        type=str,
        required=True,
        help="path to CSL100 color directory, e.g. /root/autodl-tmp/SLR_dataset/color",
    )
    parser.add_argument(
        "--corpus-path",
        type=str,
        required=True,
        help="path to corpus.txt",
    )
    parser.add_argument(
        "--split-mode",
        type=str,
        default="signer_holdout_5000x2",
        choices=["signer_holdout_5000x2", "paper_random_8_2_seeded"],
        help="dataset split mode",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="random seed for random split modes",
    )
    parser.add_argument(
        "--eval-pool-split",
        type=str,
        default="mirror",
        choices=["mirror", "disjoint_half"],
        help="for paper_random_8_2_seeded: mirror uses same dev/test pool; "
             "disjoint_half splits eval pool into 25/25 per sentence.",
    )
    parser.add_argument(
        "--dev-fast-per-sentence",
        type=int,
        default=10,
        help="samples per sentence for dev_fast split",
    )
    parser.add_argument(
        "--strict-exists",
        type=str,
        default="true",
        choices=["true", "false"],
        help="whether to fail when invalid/missing samples are found",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../csl100",
        help="output directory for npy/stm/gloss_dict",
    )
    parser.add_argument(
        "--export-frame-cache",
        type=str,
        default="false",
        choices=["true", "false"],
        help="whether to export sampled frame cache as .npy files",
    )
    parser.add_argument(
        "--cache-frames",
        type=int,
        default=64,
        choices=[32, 48, 64],
        help="number of frames per cached sample",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="",
        help="cache output directory; default is <output-dir>/frame_cache_<cache-frames>",
    )
    parser.add_argument(
        "--cache-overwrite",
        type=str,
        default="false",
        choices=["true", "false"],
        help="overwrite existing cache files",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    strict_exists = str2bool(args.strict_exists)
    export_frame_cache_flag = str2bool(args.export_frame_cache)
    cache_overwrite = str2bool(args.cache_overwrite)
    dataset_root = os.path.abspath(args.dataset_root)
    corpus_path = os.path.abspath(args.corpus_path)
    output_dir = os.path.abspath(args.output_dir)

    if not os.path.isdir(dataset_root):
        raise NotADirectoryError("dataset-root does not exist: {}".format(dataset_root))
    os.makedirs(output_dir, exist_ok=True)

    corpus = load_corpus(corpus_path)
    print("dataset_root={}".format(dataset_root))
    print("corpus_path={}".format(corpus_path))
    print("output_dir={}".format(output_dir))
    print("split_mode={}".format(args.split_mode))
    print("split_seed={}".format(args.split_seed))
    print("eval_pool_split={}".format(args.eval_pool_split))
    print("strict_exists={}".format(strict_exists))
    print("export_frame_cache={}".format(export_frame_cache_flag))

    scanned_samples, bad_samples = scan_dataset(dataset_root, corpus)
    save_bad_samples(output_dir, "scan", bad_samples)
    if strict_exists and bad_samples:
        preview = "\n".join(bad_samples[:20])
        raise RuntimeError("Found {} invalid samples while scanning.\n{}".format(len(bad_samples), preview))

    if args.split_mode == "signer_holdout_5000x2":
        train_samples, dev_samples, test_samples = split_signer_holdout_5000x2(scanned_samples)
    elif args.split_mode == "paper_random_8_2_seeded":
        train_samples, dev_samples, test_samples = split_paper_random_8_2_seeded(
            scanned_samples, args.split_seed, args.eval_pool_split
        )
    else:
        raise ValueError("Unsupported split mode: {}".format(args.split_mode))

    dev_fast_samples = build_dev_fast(
        dev_samples,
        per_sentence=args.dev_fast_per_sentence,
        split_seed=args.split_seed,
    )

    split_samples = OrderedDict(
        train=train_samples,
        dev=dev_samples,
        test=test_samples,
        dev_fast=dev_fast_samples,
    )

    split_infos = OrderedDict()
    for mode, samples in split_samples.items():
        info = list_to_info_dict(samples)
        split_infos[mode] = info
        np.save(os.path.join(output_dir, "{}_info.npy".format(mode)), info)
        generate_gt_stm(
            info,
            os.path.join(output_dir, "{}-groundtruth-{}.stm".format(args.dataset, mode)),
        )
        print("{} samples: {}".format(mode, len(info)))

    gloss_dict = build_gloss_dict(split_infos["train"])
    np.save(os.path.join(output_dir, "gloss_dict.npy"), gloss_dict)
    print("gloss_dict size: {}".format(len(gloss_dict)))

    if export_frame_cache_flag:
        cache_dir = args.cache_dir.strip()
        if not cache_dir:
            cache_dir = os.path.join(output_dir, "frame_cache_{}".format(args.cache_frames))
        cache_dir = os.path.abspath(cache_dir)
        cache_failures = export_frame_cache(
            samples=train_samples + dev_samples + test_samples,
            dataset_root=dataset_root,
            cache_dir=cache_dir,
            cache_frames=args.cache_frames,
            strict_exists=strict_exists,
            overwrite=cache_overwrite,
        )
        save_bad_samples(output_dir, "cache_export", cache_failures)
        print("cache_dir={}".format(cache_dir))
