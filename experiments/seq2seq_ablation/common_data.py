import os
from typing import Optional

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from dataset import _load_info_samples, _read_char_corpus, _sample_video_frames


def resolve_info_path(vac_root: str, explicit_path: str, split_name: str) -> str:
    if explicit_path:
        return explicit_path
    return os.path.join(vac_root, "csl100", f"{split_name}_info.npy")


def build_transform(sample_size: int):
    return transforms.Compose(
        [
            transforms.Resize([sample_size, sample_size]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )


def _extract_pose_array(loaded):
    if isinstance(loaded, np.lib.npyio.NpzFile):
        if len(loaded.files) == 0:
            return None
        return loaded[loaded.files[0]]

    if isinstance(loaded, np.ndarray) and loaded.dtype == object:
        try:
            obj = loaded.item()
        except Exception:
            obj = None
        if isinstance(obj, dict):
            for key in ["pose", "feature", "feat", "landmarks"]:
                if key in obj:
                    return np.asarray(obj[key])
    if isinstance(loaded, dict):
        for key in ["pose", "feature", "feat", "landmarks"]:
            if key in loaded:
                return np.asarray(loaded[key])
        return None
    return np.asarray(loaded)


def _load_pose_feature(
    pose_root: str,
    fileid: str,
    sentence_id: str,
    pose_dim: int,
) -> (np.ndarray, bool):
    if not pose_root:
        return np.zeros((pose_dim,), dtype=np.float32), False

    candidates = [
        os.path.join(pose_root, f"{fileid}.npy"),
        os.path.join(pose_root, f"{fileid}.npz"),
        os.path.join(pose_root, sentence_id, f"{fileid}.npy"),
        os.path.join(pose_root, sentence_id, f"{fileid}.npz"),
    ]
    for path in candidates:
        if not os.path.isfile(path):
            continue
        loaded = np.load(path, allow_pickle=True)
        arr = _extract_pose_array(loaded)
        if arr is None:
            continue
        flat = np.asarray(arr, dtype=np.float32).reshape(-1)
        if pose_dim > 0:
            if flat.shape[0] > pose_dim:
                flat = flat[:pose_dim]
            elif flat.shape[0] < pose_dim:
                flat = np.pad(flat, (0, pose_dim - flat.shape[0]), mode="constant")
        return flat, True
    return np.zeros((pose_dim,), dtype=np.float32), False


class CSLCharInfoWithAux(Dataset):
    def __init__(
        self,
        dataset_root: str,
        corpus_path: str,
        info_path: str,
        frames: int = 32,
        transform=None,
        pose_root: str = "",
        pose_dim: int = 0,
        use_pose: bool = False,
    ):
        super().__init__()
        self.dataset_root = dataset_root
        self.frames = int(frames)
        self.transform = transform
        self.samples = _load_info_samples(info_path)
        self.dict, self.corpus, self.max_length, self.unknown = _read_char_corpus(corpus_path)
        self.output_dim = len(self.dict)
        self.pose_root = pose_root
        self.pose_dim = int(max(0, pose_dim))
        self.use_pose = bool(use_pose)
        self.missing_pose = 0

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        rel_path = str(sample["folder"]).replace("/", os.sep)
        video_path = os.path.join(self.dataset_root, rel_path)
        images = _sample_video_frames(video_path, self.frames, self.transform)

        fileid = str(sample["fileid"])
        sentence_id = fileid.split("_")[0]
        tokens = torch.LongTensor(self.corpus[sentence_id])

        if self.use_pose and self.pose_dim > 0:
            pose_feat, found = _load_pose_feature(
                pose_root=self.pose_root,
                fileid=fileid,
                sentence_id=sentence_id,
                pose_dim=self.pose_dim,
            )
            if not found:
                self.missing_pose += 1
            pose_tensor = torch.from_numpy(pose_feat)
        else:
            pose_tensor = torch.zeros((0,), dtype=torch.float32)

        return images, tokens, fileid, pose_tensor


def collate_with_meta(batch):
    images = torch.stack([item[0] for item in batch], dim=0)
    tokens = torch.stack([item[1] for item in batch], dim=0)
    fileids = [item[2] for item in batch]
    pose = torch.stack([item[3] for item in batch], dim=0)
    return images, tokens, fileids, pose

