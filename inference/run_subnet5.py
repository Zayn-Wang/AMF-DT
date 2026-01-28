# run_subnet5.py
"""
Independent inference script for Sub-network 5 (M3t).

This corresponds to original `if model_run:`:
- load PFS model, run on test cohort (image-only)
- load OS model, run on test cohort
- save OS & PFS risk scores as CSV
"""

import os
import warnings

import torch
import pandas as pd
import torch.multiprocessing

from models.models_m3t import M3t
from data_utils import build_file_dicts_only_imgs, build_transforms
from eval_utils import model_run_gpu_m3t

warnings.filterwarnings("ignore")
torch.multiprocessing.set_sharing_strategy("file_system")


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Independent inference for Sub-network 5.")

    # GPU
    parser.add_argument("--gpu", type=str, default="1", help="CUDA_VISIBLE_DEVICES id")
    parser.add_argument("--val_batch", type=int, default=4)
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
        default="../weights/multitask_subnet5_PFS.pth",
    )
    parser.add_argument(
        "--os_model_path",
        type=str,
        default="../weights/multitask_subnet5_OS.pth",
    )

    parser.add_argument("--drop_rate", type=float, default=0.1)
    parser.add_argument("--best_model_name", type=str, default="multitask_subnet5")

    return parser.parse_args()


def main():
    args = parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.cuda.empty_cache()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("------------------- Step 1: load data (image-only) --------------------------")
    test_files = build_file_dicts_only_imgs(args)

    print("------------------- Step 2: define transforms --------------------------")
    _, val_transforms = build_transforms(rand_p=0.0)

    model = M3t(args.drop_rate).to(device)

    print("------------------- Run PFS model --------------------------")
    model.load_state_dict(torch.load(args.pfs_model_path))
    _, pfs_risk = model_run_gpu_m3t(model, test_files, val_transforms, args)

    print("------------------- Run OS model --------------------------")
    model.load_state_dict(torch.load(args.os_model_path))
    os_risk, _ = model_run_gpu_m3t(model, test_files, val_transforms, args)

    pred_save = torch.cat((os_risk, pfs_risk), dim=1)
    pred_save = pd.DataFrame(pred_save.cpu().numpy())

    os.makedirs("outputs", exist_ok=True)
    risk_filename = f"{args.best_model_name}_sindependent_subnet5.csv"
    pred_save.to_csv(f"./outputs/{risk_filename}", index=False)

    print("Independent predictions saved to outputs/:")
    print(f"  {risk_filename}")


if __name__ == "__main__":
    main()
