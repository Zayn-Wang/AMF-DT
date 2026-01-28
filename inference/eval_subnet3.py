# eval_subnet3.py
"""
Evaluation script for Sub-network 3.

This reproduces the behavior of the original `if model_pred:` block:
- load PFS model, evaluate train/val/test, save predictions and bottleneck features
- load OS model, evaluate train/val/test
"""

import os
import warnings

import torch
import pandas as pd
import torch.multiprocessing

from models.models_subnet3 import AutoEncoder_New
from data_utils import build_file_dicts, build_transforms
from eval_utils import model_eval_gpu_ae

warnings.filterwarnings("ignore")
torch.multiprocessing.set_sharing_strategy("file_system")


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate Sub-network 3 and save CSV predictions/features.")

    # GPU
    parser.add_argument("--gpu", type=str, default="3", help="CUDA_VISIBLE_DEVICES id")
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
    train_files, val_files, test_files = build_file_dicts(args)

    print("------------------- Step 2: define transforms --------------------------")
    _, val_transforms = build_transforms(rand_p=0.0)  # no random augmentation for eval

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

    print("------------------- Evaluate PFS model --------------------------")
    model.load_state_dict(torch.load(args.pfs_model_path))
    (
        _,
        pfs_risk_train,
        os_train,
        ostime_train,
        pfs_train,
        pfstime_train,
        ID_train,
        age_train,
        bn_feature_train,
    ) = model_eval_gpu_ae(model, train_files, val_transforms, args)
    (
        _,
        pfs_risk_val,
        os_val,
        ostime_val,
        pfs_val,
        pfstime_val,
        ID_val,
        age_val,
        bn_feature_val,
    ) = model_eval_gpu_ae(model, val_files, val_transforms, args)
    (
        _,
        pfs_risk_test,
        os_test,
        ostime_test,
        pfs_test,
        pfstime_test,
        ID_test,
        age_test,
        bn_feature_test,
    ) = model_eval_gpu_ae(model, test_files, val_transforms, args)

    print("------------------- Evaluate OS model --------------------------")
    model.load_state_dict(torch.load(args.os_model_path))
    os_risk_train, _, _, _, _, _, _, _, _ = model_eval_gpu_ae(model, train_files, val_transforms, args)
    os_risk_val, _, _, _, _, _, _, _, _ = model_eval_gpu_ae(model, val_files, val_transforms, args)
    os_risk_test, _, _, _, _, _, _, _, _ = model_eval_gpu_ae(model, test_files, val_transforms, args)

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

    pred_train_save = pd.DataFrame(pred_train_save.cpu().numpy())
    pred_val_save = pd.DataFrame(pred_val_save.cpu().numpy())
    pred_test_save = pd.DataFrame(pred_test_save.cpu().numpy())

    # bottleneck features (ID + feature vector)
    pred_f_train = torch.cat((ID_train, bn_feature_train), 1)
    pred_f_val = torch.cat((ID_val, bn_feature_val), 1)
    pred_f_test = torch.cat((ID_test, bn_feature_test), 1)

    pred_f_train = pd.DataFrame(pred_f_train.cpu().numpy())
    pred_f_val = pd.DataFrame(pred_f_val.cpu().numpy())
    pred_f_test = pd.DataFrame(pred_f_test.cpu().numpy())

    os.makedirs("outputs", exist_ok=True)
    pred_train_save.to_csv(f"./outputs/{args.best_model_name}_strain.csv", index=False)
    pred_val_save.to_csv(f"./outputs/{args.best_model_name}_sval.csv", index=False)
    pred_test_save.to_csv(f"./outputs/{args.best_model_name}_stest.csv", index=False)

    pred_f_train.to_csv(f"./outputs/{args.best_model_name}_sfeature_train.csv", index=False)
    pred_f_val.to_csv(f"./outputs/{args.best_model_name}_sfeature_val.csv", index=False)
    pred_f_test.to_csv(f"./outputs/{args.best_model_name}_sfeature_test.csv", index=False)

    print("CSV files saved to outputs/:")
    print(
        f"  {args.best_model_name}_strain.csv, {args.best_model_name}_sval.csv, {args.best_model_name}_stest.csv\n"
        f"  {args.best_model_name}_sfeature_train.csv, {args.best_model_name}_sfeature_val.csv, {args.best_model_name}_sfeature_test.csv"
    )


if __name__ == "__main__":
    main()
