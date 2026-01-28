# run_subnet3.py
"""
Independent inference script for Sub-network 3.

This reproduces the behavior of the original `if model_run:` block:
- load PFS model and run inference to get PFS risk scores and bottleneck features
- load OS model and run inference to get OS risk scores
- save combined risks and features to CSV
"""

import os
import warnings

import torch
import pandas as pd
import torch.multiprocessing

from models.models_subnet3 import AutoEncoder_New
from data_utils import build_file_dicts_only_imgs, build_transforms
from eval_utils import model_run_gpu_ae

warnings.filterwarnings("ignore")
torch.multiprocessing.set_sharing_strategy("file_system")


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Independent inference for Sub-network 3.")

    # GPU
    parser.add_argument("--gpu", type=str, default="3", help="CUDA_VISIBLE_DEVICES id")
    parser.add_argument("--val_batch", type=int, default=16)
    parser.add_argument("--num_workers_val", type=int, default=10)

    # Paths (only test_dir)
    parser.add_argument(
        "--test_dir",
        type=str,
        default="../samples/Input_Test/",
    )

    # Model paths
    parser.add_argument(
        "--pfs_model_path",
        type=str,
        default="../weights/multitask_subnet3_PFS.pth",
    )
    parser.add_argument(
        "--os_model_path",
        type=str,
        default="../weights/multitask_subnet3_OS.pth",
    )

    parser.add_argument("--drop_rate", type=float, default=0.25)
    parser.add_argument("--best_model_name", type=str, default="multitask_subnet3")

    return parser.parse_args()


def main():
    args = parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.cuda.empty_cache()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("------------------- Step 1: load data --------------------------")
    test_files = build_file_dicts_only_imgs(args)

    print("------------------- Step 2: define transforms --------------------------")
    _, val_transforms = build_transforms(rand_p=0.0)

    model = AutoEncoder_New(
        dimensions=3,
        in_channels=3,
        out_channels=1,
        num_res_units=2,
        channels=(32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        inter_channels=[256, 256],
        inter_dilations=[2, 2],
        dropout=args.drop_rate,
    ).to(device)

    print("------------------- Run PFS model --------------------------")
    model.load_state_dict(torch.load(args.pfs_model_path))
    _, pfs_risk, bn_feature = model_run_gpu_ae(model, test_files, val_transforms, args)

    print("------------------- Run OS model --------------------------")
    model.load_state_dict(torch.load(args.os_model_path))
    os_risk, _, _ = model_run_gpu_ae(model, test_files, val_transforms, args)

    pred_save = torch.cat((os_risk, pfs_risk), 1)
    pred_save = pd.DataFrame(pred_save.cpu().numpy())
    bn_feature = pd.DataFrame(bn_feature.cpu().numpy())

    os.makedirs("outputs", exist_ok=True)
    risk_filename = f"{args.best_model_name}_sindependent_subnet3.csv"
    feat_filename = f"{args.best_model_name}_sfeature_independent_subnet3.csv"

    pred_save.to_csv(f"./outputs/{risk_filename}", index=False)
    bn_feature.to_csv(f"./outputs/{feat_filename}", index=False)

    print("Independent predictions and features saved to outputs/:")
    print(f"  {risk_filename}")
    print(f"  {feat_filename}")


if __name__ == "__main__":
    main()
