import os
import cv2
import time
import torch
import warnings
import numpy as np
import torch.utils.data as data

from utils import video_augmentation

warnings.simplefilter(action="ignore", category=FutureWarning)
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


def _resolve_info_path(mode):
    candidates = [
        os.path.join(".", "csl100", "{}_info.npy".format(mode)),
        os.path.join("..", "csl100", "{}_info.npy".format(mode)),
    ]
    for path in candidates:
        abs_path = os.path.abspath(path)
        if os.path.isfile(abs_path):
            return abs_path
    raise FileNotFoundError(
        "Cannot find {}_info.npy. Checked: {}".format(mode, [os.path.abspath(p) for p in candidates])
    )


class CSLFeeder(data.Dataset):
    def __init__(
        self,
        prefix,
        gloss_dict,
        drop_ratio=1,
        num_gloss=-1,
        mode="train",
        transform_mode=True,
        datatype="video",
        frames=96,
        use_cache=False,
        cache_dir="",
        cache_fallback_video=True,
        strict_exists=True,
        strict_label=True,
        **kwargs
    ):
        del drop_ratio, num_gloss, kwargs
        self.mode = mode
        self.prefix = prefix
        self.dict = gloss_dict
        self.data_type = datatype
        self.frames = int(frames) if int(frames) > 0 else 96
        self.use_cache = bool(use_cache)
        self.cache_fallback_video = bool(cache_fallback_video)
        self.transform_mode = "train" if transform_mode else "test"
        self.strict_exists = bool(strict_exists)
        self.strict_label = bool(strict_label)

        info_path = _resolve_info_path(mode)
        self.inputs_list = np.load(info_path, allow_pickle=True).item()
        self.sample_ids = sorted([k for k in self.inputs_list.keys() if isinstance(k, int)])
        if len(self.sample_ids) == 0:
            raise RuntimeError("No valid sample ids loaded from {}".format(info_path))
        default_cache_dir = os.path.join(os.path.dirname(info_path), "frame_cache_{}".format(self.frames))
        self.cache_dir = os.path.abspath(cache_dir) if str(cache_dir).strip() else os.path.abspath(default_cache_dir)
        self.cache_total = len(self.sample_ids)
        self.cache_existing = 0
        if self.use_cache:
            for sample_id in self.sample_ids:
                sample = self.inputs_list[sample_id]
                if os.path.isfile(self._cache_path(sample["fileid"])):
                    self.cache_existing += 1

        self._validate_samples()
        print(mode, len(self))
        if self.use_cache:
            print(
                "[CSLFeeder:{}] cache coverage: {}/{} ({:.2%}), cache_dir={}, fallback_video={}".format(
                    mode,
                    self.cache_existing,
                    self.cache_total,
                    self.cache_existing / max(1, self.cache_total),
                    self.cache_dir,
                    self.cache_fallback_video,
                )
            )
        self.data_aug = self.transform()
        print("")

    def _cache_path(self, fileid):
        return os.path.join(self.cache_dir, "{}.npy".format(fileid))

    def _validate_samples(self):
        bad_items = []
        for sample_id in self.sample_ids:
            sample = self.inputs_list[sample_id]
            if "folder" not in sample:
                bad_items.append("sample {} missing 'folder'".format(sample_id))
                continue
            video_path = os.path.join(self.prefix, sample["folder"].replace("/", os.sep))
            cache_path = self._cache_path(sample["fileid"])
            cache_exists = os.path.isfile(cache_path)
            if self.use_cache and not cache_exists and not self.cache_fallback_video and self.strict_exists:
                bad_items.append("sample {} missing cache file: {}".format(sample_id, cache_path))
            if self.strict_exists and not os.path.isfile(video_path) and (not self.use_cache or not cache_exists):
                bad_items.append("sample {} missing file: {}".format(sample_id, video_path))

        if self.strict_exists and bad_items:
            preview = "\n".join(bad_items[:20])
            raise RuntimeError(
                "Found {} invalid samples in {} split.\n{}".format(len(bad_items), self.mode, preview)
            )

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        if self.data_type == "video":
            input_data, label, sample = self.read_video(sample_id)
            input_data, label = self.normalize(input_data, label)
            return input_data, torch.LongTensor(label), self._normalized_original_info(sample)
        input_data, label = self.read_features(sample_id)
        return input_data, label, self._normalized_original_info(self.inputs_list[sample_id])

    def _normalized_original_info(self, sample):
        fileid = str(sample.get("fileid", "")).strip()
        folder = str(sample.get("folder", "")).strip()
        signer = str(sample.get("signer", "")).strip()
        label = str(sample.get("label", "")).strip()

        original_info = str(sample.get("original_info", "")).strip()
        parts = original_info.split("|") if original_info else []
        if len(parts) >= 1 and parts[0] == fileid:
            return original_info
        return "{}|{}|{}|{}".format(fileid, folder, signer, label)

    def _label_to_ids(self, label_text, sample_id):
        char_tokens = [ch for ch in label_text if ch.strip()]
        ids = []
        missing = []
        for ch in char_tokens:
            if ch in self.dict:
                ids.append(self.dict[ch][0])
            else:
                missing.append(ch)

        if self.strict_label and missing:
            uniq = "".join(sorted(set(missing)))
            raise KeyError(
                "Sample {} contains {} chars missing from gloss_dict: {}".format(
                    sample_id, len(set(missing)), uniq
                )
            )
        return ids

    def read_video(self, sample_id):
        sample = self.inputs_list[sample_id]
        if self.use_cache:
            cache_path = self._cache_path(sample["fileid"])
            if os.path.isfile(cache_path):
                frames = np.load(cache_path, allow_pickle=False)
                if len(frames) > self.frames:
                    sample_index = np.linspace(0, len(frames) - 1, self.frames).astype(np.int32).tolist()
                    frames = frames[sample_index]
                elif len(frames) < self.frames:
                    if len(frames) == 0:
                        raise RuntimeError("Empty cache file: {}".format(cache_path))
                    pad = self.frames - len(frames)
                    frames = np.concatenate([frames, np.repeat(frames[-1:], pad, axis=0)], axis=0)
                label_ids = self._label_to_ids(str(sample.get("label", "")), sample_id)
                return frames, label_ids, sample
            if not self.cache_fallback_video:
                raise FileNotFoundError("Cache miss and fallback disabled: {}".format(cache_path))

        video_rel = sample["folder"].replace("/", os.sep)
        video_path = os.path.join(self.prefix, video_rel)
        cap = cv2.VideoCapture(video_path)
        frames = []

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()

        if len(frames) == 0:
            msg = "Decoded 0 frames: {}".format(video_path)
            if self.strict_exists:
                raise RuntimeError(msg)
            # Keep data flow alive for non-strict mode.
            frames = [np.zeros((224, 224, 3), dtype=np.uint8)]
        else:
            if len(frames) > self.frames:
                sample_index = np.linspace(0, len(frames) - 1, self.frames).astype(np.int32).tolist()
                frames = [frames[i] for i in sample_index]
            elif len(frames) < self.frames:
                pad = self.frames - len(frames)
                frames.extend([frames[-1]] * pad)

        label_ids = self._label_to_ids(str(sample.get("label", "")), sample_id)
        return frames, label_ids, sample

    def read_features(self, sample_id):
        sample = self.inputs_list[sample_id]
        data = np.load(
            os.path.join(".", "features", self.mode, "{}_features.npy".format(sample["fileid"])),
            allow_pickle=True,
        ).item()
        return data["features"], data["label"]

    def normalize(self, video, label, file_id=None):
        video, label = self.data_aug(video, label, file_id)
        video = video.float() / 127.5 - 1
        return video, label

    def transform(self):
        if self.transform_mode == "train":
            print("Apply training transform.")
            return video_augmentation.Compose(
                [
                    video_augmentation.RandomCrop(224),
                    video_augmentation.RandomHorizontalFlip(0.5),
                    video_augmentation.ToTensor(),
                    video_augmentation.TemporalRescale(0.2),
                ]
            )
        print("Apply testing transform.")
        return video_augmentation.Compose(
            [
                video_augmentation.CenterCrop(224),
                video_augmentation.ToTensor(),
            ]
        )

    @staticmethod
    def collate_fn(batch):
        batch = [item for item in sorted(batch, key=lambda x: len(x[0]), reverse=True)]
        video, label, info = list(zip(*batch))
        if len(video[0].shape) > 3:
            max_len = len(video[0])
            video_length = torch.LongTensor([np.ceil(len(vid) / 4.0) * 4 + 12 for vid in video])
            left_pad = 6
            right_pad = int(np.ceil(max_len / 4.0)) * 4 - max_len + 6
            max_len = max_len + left_pad + right_pad
            padded_video = [
                torch.cat(
                    (
                        vid[0][None].expand(left_pad, -1, -1, -1),
                        vid,
                        vid[-1][None].expand(max_len - len(vid) - left_pad, -1, -1, -1),
                    ),
                    dim=0,
                )
                for vid in video
            ]
            padded_video = torch.stack(padded_video)
        else:
            max_len = len(video[0])
            video_length = torch.LongTensor([len(vid) for vid in video])
            padded_video = [
                torch.cat(
                    (
                        vid,
                        vid[-1][None].expand(max_len - len(vid), -1),
                    ),
                    dim=0,
                )
                for vid in video
            ]
            padded_video = torch.stack(padded_video).permute(0, 2, 1)
        label_length = torch.LongTensor([len(lab) for lab in label])
        if max(label_length) == 0:
            return padded_video, video_length, [], [], info
        padded_label = []
        for lab in label:
            padded_label.extend(lab)
        padded_label = torch.LongTensor(padded_label)
        return padded_video, video_length, padded_label, label_length, info

    def __len__(self):
        return len(self.sample_ids)

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time
