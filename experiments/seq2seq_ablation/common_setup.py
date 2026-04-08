import os

import torch
from torch.utils.data import DataLoader

from .common_data import (
    CSLCharInfoWithAux,
    build_transform,
    collate_with_meta,
    resolve_info_path,
)
from .common_seed import set_random_seed


def prepare_device_and_seed(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_random_seed(args.random_seed, deterministic=args.deterministic)
    return device


def build_dataloaders(args, use_pose=False, pose_root="", pose_dim=0):
    if not args.use_char_level:
        raise NotImplementedError("Ablation scripts currently support char-level only.")
    if not args.use_vac_split:
        raise NotImplementedError("Ablation scripts currently require use_vac_split=true.")

    vac_root = os.path.abspath(args.vac_root)
    train_info = os.path.abspath(resolve_info_path(vac_root, args.train_info_path, "train"))
    val_info = os.path.abspath(resolve_info_path(vac_root, args.val_info_path, "dev"))
    test_info = os.path.abspath(resolve_info_path(vac_root, args.test_info_path, "test"))
    transform = build_transform(args.sample_size)

    train_set = CSLCharInfoWithAux(
        dataset_root=args.data_path,
        corpus_path=args.corpus_path,
        info_path=train_info,
        frames=args.sample_duration,
        transform=transform,
        pose_root=pose_root,
        pose_dim=pose_dim,
        use_pose=use_pose,
    )
    val_set = CSLCharInfoWithAux(
        dataset_root=args.data_path,
        corpus_path=args.corpus_path,
        info_path=val_info,
        frames=args.sample_duration,
        transform=transform,
        pose_root=pose_root,
        pose_dim=pose_dim,
        use_pose=use_pose,
    )
    test_set = CSLCharInfoWithAux(
        dataset_root=args.data_path,
        corpus_path=args.corpus_path,
        info_path=test_info,
        frames=args.sample_duration,
        transform=transform,
        pose_root=pose_root,
        pose_dim=pose_dim,
        use_pose=use_pose,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_with_meta,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_with_meta,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_with_meta,
    )
    return train_set, val_set, test_set, train_loader, val_loader, test_loader
