# run_subnet1.py
"""
Independent inference script for Sub-network 1.

This reproduces the behavior of the original `if model_run:` block:
- load PFS model and run inference to get PFS risk scores
- load OS model and run inference to get OS risk scores
- save combined risks to CSV
"""

import os
import warnings

import torch
import pandas as pd

import torch.multiprocessing

from models.models import EfficientNet
from data_utils import build_file_dicts_only_imgs, build_transforms
from eval_utils import model_run_gpu

warnings.filterwarnings("ignore")
torch.multiprocessing.set_sharing_strategy("file_system")


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Independent inference for Sub-network 1.")

    # GPU
    parser.add_argument("--gpu", type=str, default="1", help="CUDA_VISIBLE_DEVICES id")
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
        default="../weights/multitask_subnet1_PFS.pth",
    )
    parser.add_argument(
        "--os_model_path",
        type=str,
        default="../weights/multitask_subnet1_OS.pth",
    )

    parser.add_argument("--best_model_name", type=str, default="multitask_subnet1")

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

    model = EfficientNet().to(device)

    print("------------------- Run PFS model --------------------------")
    model.load_state_dict(torch.load(args.pfs_model_path))
    _, pfs_risk = model_run_gpu(model, test_files, val_transforms, args)

    print("------------------- Run OS model --------------------------")
    model.load_state_dict(torch.load(args.os_model_path))
    os_risk, _ = model_run_gpu(model, test_files, val_transforms, args)

    pred_save = torch.cat((os_risk, pfs_risk), 1)
    pred_save = pred_save.cpu().numpy()
    pred_save = pd.DataFrame(pred_save)
    os.makedirs("outputs", exist_ok=True)
    pred_save.to_csv("./outputs/" + args.best_model_name + "_sindependent.csv", index=False)

    print("Independent predictions saved:", "outputs/" + args.best_model_name + "_sindependent.csv")


if __name__ == "__main__":
    main()
