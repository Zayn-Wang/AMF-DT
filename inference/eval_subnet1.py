# eval_subnet1.py
"""
Evaluation script for Sub-network 1.

This reproduces the behavior of the original `if model_pred:` block:
- load PFS model, evaluate train/val/test
- load OS model, evaluate train/val/test
- save combined predictions and labels to CSV files
"""

import os
import warnings

import torch
import pandas as pd

import torch.multiprocessing

from models.models import EfficientNet
from data_utils import build_file_dicts, build_transforms
from eval_utils import model_eval_gpu

warnings.filterwarnings("ignore")
torch.multiprocessing.set_sharing_strategy("file_system")


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate Sub-network 1 and save CSV predictions.")

    # GPU
    parser.add_argument("--gpu", type=str, default="1", help="CUDA_VISIBLE_DEVICES id")
    parser.add_argument("--val_batch", type=int, default=16)
    parser.add_argument("--num_workers_val", type=int, default=10)

    # Paths
    parser.add_argument(
        "--train_dir",
        type=str,
        default="../samples/Input_Train/",
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        default="../samples/Input_Test/",
    )
    parser.add_argument(
        "--val_dir",
        type=str,
        default="../samples/Input_Val/",
    )

    parser.add_argument(
        "--train_csv",
        type=str,
        default="../samples/train_events.csv",
    )
    parser.add_argument(
        "--test_csv",
        type=str,
        default="../samples/test_events.csv",
    )
    parser.add_argument(
        "--val_csv",
        type=str,
        default="../samples/valid_events.csv",
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
    train_files, val_files, test_files = build_file_dicts(args)

    print("------------------- Step 2: define transforms --------------------------")
    _, val_transforms = build_transforms(rand_p=0.0)  # no random augmentation for eval

    model = EfficientNet().to(device)

    print("------------------- Evaluate PFS model --------------------------")
    model.load_state_dict(torch.load(args.pfs_model_path))
    _, pfs_risk_train, os_train, ostime_train, pfs_train, pfstime_train, ID_train, age_train = model_eval_gpu(
        model, train_files, val_transforms, args
    )
    _, pfs_risk_val, os_val, ostime_val, pfs_val, pfstime_val, ID_val, age_val = model_eval_gpu(
        model, val_files, val_transforms, args
    )
    _, pfs_risk_test, os_test, ostime_test, pfs_test, pfstime_test, ID_test, age_test = model_eval_gpu(
        model, test_files, val_transforms, args
    )

    print("------------------- Evaluate OS model --------------------------")
    model.load_state_dict(torch.load(args.os_model_path))
    os_risk_train, _, _, _, _, _, _, _ = model_eval_gpu(model, train_files, val_transforms, args)
    os_risk_val, _, _, _, _, _, _, _ = model_eval_gpu(model, val_files, val_transforms, args)
    os_risk_test, _, _, _, _, _, _, _ = model_eval_gpu(model, test_files, val_transforms, args)

    # reshape as original
    os_train, ostime_train, pfs_train, pfstime_train, ID_train = (
        os_train.unsqueeze(1),
        ostime_train.unsqueeze(1),
        pfs_train.unsqueeze(1),
        pfstime_train.unsqueeze(1),
        ID_train.unsqueeze(1),
    )
    os_val, ostime_val, pfs_val, pfstime_val, ID_val = (
        os_val.unsqueeze(1),
        ostime_val.unsqueeze(1),
        pfs_val.unsqueeze(1),
        pfstime_val.unsqueeze(1),
        ID_val.unsqueeze(1),
    )
    os_test, ostime_test, pfs_test, pfstime_test, ID_test = (
        os_test.unsqueeze(1),
        ostime_test.unsqueeze(1),
        pfs_test.unsqueeze(1),
        pfstime_test.unsqueeze(1),
        ID_test.unsqueeze(1),
    )

    pred_train_save = torch.cat(
        (os_risk_train, os_train, ostime_train, pfs_risk_train, pfs_train, pfstime_train, ID_train, age_train), 1
    )
    pred_val_save = torch.cat(
        (os_risk_val, os_val, ostime_val, pfs_risk_val, pfs_val, pfstime_val, ID_val, age_val), 1
    )
    pred_test_save = torch.cat(
        (os_risk_test, os_test, ostime_test, pfs_risk_test, pfs_test, pfstime_test, ID_test, age_test), 1
    )

    pred_train_save = pred_train_save.cpu().numpy()
    pred_val_save = pred_val_save.cpu().numpy()
    pred_test_save = pred_test_save.cpu().numpy()

    pred_train_save = pd.DataFrame(pred_train_save)
    pred_val_save = pd.DataFrame(pred_val_save)
    pred_test_save = pd.DataFrame(pred_test_save)

    os.makedirs("outputs", exist_ok=True)
    pred_train_save.to_csv("./outputs/" + args.best_model_name + "_strain.csv", index=False)
    pred_val_save.to_csv("./outputs/" + args.best_model_name + "_sval.csv", index=False)
    pred_test_save.to_csv("./outputs/" + args.best_model_name + "_stest.csv", index=False)

    print("CSV files saved:", "outputs/" + args.best_model_name + "_strain/sval/stest.csv")


if __name__ == "__main__":
    main()
