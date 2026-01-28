# data_utils.py
"""
Data loading utilities for Sub-network 1.

DDP upgrade:
- use DistributedSampler for train_loader when distributed=True
"""

import os
import glob
import pandas as pd

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    AsChannelFirstd,
    RandAdjustContrastd,
    RandRotate90d,
    RandFlipd,
    RandZoomd,
    EnsureTyped,
)
from monai.data import Dataset, DataLoader, list_data_collate


def build_file_dicts(args):
    train_dir = args.train_dir
    test_dir = args.test_dir
    val_dir = args.val_dir

    train_img = sorted(glob.glob(os.path.join(train_dir, "Patient_*.nii.gz")))
    val_img = sorted(glob.glob(os.path.join(val_dir, "Patient_*.nii.gz")))
    test_img = sorted(glob.glob(os.path.join(test_dir, "Patient_*.nii.gz")))

    train_os = pd.read_csv(args.train_csv)
    test_os = pd.read_csv(args.test_csv)
    val_os = pd.read_csv(args.val_csv)

    train_os["ID"] = train_os["ID"].astype(str)
    val_os["ID"] = val_os["ID"].astype(str)
    test_os["ID"] = test_os["ID"].astype(str)

    train_files = [
        {
            "input": in_img,
            "OS_status": df1,
            "OS_time": df2,
            "ID": df3,
            "PFS_status": df4,
            "PFS_time": df5,
            "PVTT": df6,
            "LungMet": df7,
            "BoneMet": df8,
            "Up to seven": df9,
            "LNMet": df10,
            "Age": df11,
            "Gender": df12,
            "Child_Pugh": df13,
            "HBV": df14,
            "Stage": df15,
        }
        for in_img, df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12, df13, df14, df15 in zip(
            train_img,
            train_os["OS_status"],
            train_os["OS_time"],
            train_os["ID"],
            train_os["PFS_status"],
            train_os["PFS_time"],
            train_os["PVTT"],
            train_os["LungMet"],
            train_os["BoneMet"],
            train_os["Up to seven"],
            train_os["LNMet"],
            train_os["Age"],
            train_os["Male"],
            train_os["Child_Pugh"],
            train_os["HBV"],
            train_os["BCLC"],
        )
    ]

    val_files = [
        {
            "input": in_img,
            "OS_status": df1,
            "OS_time": df2,
            "ID": df3,
            "PFS_status": df4,
            "PFS_time": df5,
            "PVTT": df6,
            "LungMet": df7,
            "BoneMet": df8,
            "Up to seven": df9,
            "LNMet": df10,
            "Age": df11,
            "Gender": df12,
            "Child_Pugh": df13,
            "HBV": df14,
            "Stage": df15,
        }
        for in_img, df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12, df13, df14, df15 in zip(
            val_img,
            val_os["OS_status"],
            val_os["OS_time"],
            val_os["ID"],
            val_os["PFS_status"],
            val_os["PFS_time"],
            val_os["PVTT"],
            val_os["LungMet"],
            val_os["BoneMet"],
            val_os["Up to seven"],
            val_os["LNMet"],
            val_os["Age"],
            val_os["Male"],
            val_os["Child_Pugh"],
            val_os["HBV"],
            val_os["BCLC"],
        )
    ]

    test_files = [
        {
            "input": in_img,
            "OS_status": df1,
            "OS_time": df2,
            "ID": df3,
            "PFS_status": df4,
            "PFS_time": df5,
            "PVTT": df6,
            "LungMet": df7,
            "BoneMet": df8,
            "Up to seven": df9,
            "LNMet": df10,
            "Age": df11,
            "Gender": df12,
            "Child_Pugh": df13,
            "HBV": df14,
            "Stage": df15,
        }
        for in_img, df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12, df13, df14, df15 in zip(
            test_img,
            test_os["OS_status"],
            test_os["OS_time"],
            test_os["ID"],
            test_os["PFS_status"],
            test_os["PFS_time"],
            test_os["PVTT"],
            test_os["LungMet"],
            test_os["BoneMet"],
            test_os["Up to seven"],
            test_os["LNMet"],
            test_os["Age"],
            test_os["Male"],
            test_os["Child_Pugh"],
            test_os["HBV"],
            test_os["BCLC"],
        )
    ]

    return train_files, val_files, test_files


def build_file_dicts_only_imgs(args):
    test_dir = args.test_dir
    test_img = sorted(glob.glob(os.path.join(test_dir, "Patient_*.nii.gz")))
    test_files = [{"input": in_img} for in_img in test_img]
    return test_files


def build_transforms(rand_p):
    train_transforms = Compose(
        [
            LoadImaged(keys=["input"]),
            EnsureChannelFirstd(keys=["input"], channel_dim=0),
            RandAdjustContrastd(keys=["input"], prob=rand_p),
            RandRotate90d(keys=["input"], prob=rand_p, spatial_axes=(0, 2)),
            RandRotate90d(keys=["input"], prob=rand_p, spatial_axes=(0, 1)),
            RandFlipd(keys=["input"], prob=rand_p),
            RandZoomd(keys="input", prob=rand_p),
            EnsureTyped(keys=["input"]),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["input"]),
            AsChannelFirstd(keys=["input"], channel_dim=0),
            EnsureTyped(keys=["input"]),
        ]
    )
    return train_transforms, val_transforms


def create_dataloaders(
    train_files,
    val_files,
    test_files,
    train_transforms,
    val_transforms,
    args,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
):
    train_ds = Dataset(data=train_files, transform=train_transforms)
    val_ds = Dataset(data=val_files, transform=val_transforms)
    test_ds = Dataset(data=test_files, transform=val_transforms)

    train_sampler = None
    if distributed:
        from torch.utils.data.distributed import DistributedSampler

        train_sampler = DistributedSampler(
            train_ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=False,
        )

    if args.best_model_name == "multitask_subnet3":
        train_loader = DataLoader(
            train_ds,
            batch_size=args.train_batch,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=args.num_workers_train,
            collate_fn=list_data_collate,
            pin_memory=True,
            drop_last=True,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=args.train_batch,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=args.num_workers_train,
            collate_fn=list_data_collate,
            pin_memory=True,
        )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.val_batch,
        shuffle=False,
        num_workers=args.num_workers_val,
        collate_fn=list_data_collate,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.test_batch,
        shuffle=False,
        num_workers=args.num_workers_test,
        collate_fn=list_data_collate,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, train_ds, val_ds, test_ds
