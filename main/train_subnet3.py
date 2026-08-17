# train_subnet3.py
"""
Training script for Sub-network 3 (AutoEncoder-based survival model).

Multi-GPU support:
- If launched with torchrun (DDP): use DistributedDataParallel
- Else if multiple GPUs visible: use DataParallel
- Else: single GPU/CPU

Compatibility fix:
- torchinfo is optional (skip summary if not installed)

Notes:
- In DDP mode, only rank0 does printing/tensorboard/save to avoid duplication.
- DataLoader uses DistributedSampler (implemented in data_utils.create_dataloaders) when distributed=True.
"""

import os
import warnings
from datetime import datetime

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from tqdm import trange

import torch.multiprocessing
from monai.utils import set_determinism  # kept for compatibility

from models.models_subnet3 import AutoEncoder_New
from metrics import cox_log_rank, CIndex_lifeline, MultiLabel_Acc
from losses import surv_loss, MultiTaskLossWrapper4
from data_utils import build_file_dicts, build_transforms, create_dataloaders

warnings.filterwarnings("ignore")
torch.multiprocessing.set_sharing_strategy("file_system")


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Train Sub-network 3 (AutoEncoder-based survival model)")

    # GPU settings
    parser.add_argument("--gpu", type=str, default="3", help="CUDA_VISIBLE_DEVICES id list, e.g. 0 or 0,1,2")

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

    # Hyperparameters
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lr_decay", type=float, default=0.1)
    parser.add_argument("--drop_rate", type=float, default=0.25)
    parser.add_argument("--rand_p", type=float, default=0.35)
    parser.add_argument("--max_epochs", type=int, default=250)
    parser.add_argument("--train_batch", type=int, default=16)
    parser.add_argument("--val_batch", type=int, default=16)
    parser.add_argument("--test_batch", type=int, default=16)
    parser.add_argument("--skip_epoch_model", type=int, default=40)
    parser.add_argument("--n_loss", type=int, default=4)

    # DataLoader workers
    parser.add_argument("--num_workers_train", type=int, default=4)
    parser.add_argument("--num_workers_val", type=int, default=4)
    parser.add_argument("--num_workers_test", type=int, default=4)

    # Misc
    parser.add_argument("--verbose", action="store_true", help="print detailed training logs")
    parser.add_argument("--best_model_name", type=str, default="multitask_subnet3")

    return parser.parse_args()


def _ddp_env_available() -> bool:
    # torchrun will set these envs
    return ("RANK" in os.environ) and ("WORLD_SIZE" in os.environ) and (int(os.environ.get("WORLD_SIZE", "1")) > 1)


def _get_rank_world():
    if _ddp_env_available():
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        return rank, world_size, local_rank
    return 0, 1, 0


def _is_rank0(rank: int) -> bool:
    return rank == 0


def _unwrap_model(m: torch.nn.Module) -> torch.nn.Module:
    # Works for DataParallel and DDP
    return m.module if hasattr(m, "module") else m


def _try_model_summary(model: torch.nn.Module, input_tensor_shape):
    """
    torchinfo is optional. If not installed, skip.
    """
    try:
        from torchinfo import summary  # optional dependency
        summary(model, input_size=input_tensor_shape)
    except ModuleNotFoundError:
        print("[INFO] torchinfo not installed; skip model summary.")
    except Exception as e:
        print("[INFO] model summary skipped due to exception:", str(e))


def main():
    args = parse_args()

    # Environment settings
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.cuda.empty_cache()

    verbose = args.verbose

    # DDP init (if torchrun)
    use_ddp = _ddp_env_available()
    rank, world_size, local_rank = _get_rank_world()

    if use_ddp:
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def r0_print(*msg):
        if _is_rank0(rank):
            print(*msg)

    r0_print("------------------- Step 1: load data --------------------------")
    train_files, val_files, test_files = build_file_dicts(args)

    r0_print("------------------- Step 2: define transforms --------------------------")
    train_transforms, val_transforms = build_transforms(args.rand_p)

    r0_print("------------------- Step 3: define dataloaders --------------------------")
    (
        train_loader,
        val_loader,
        test_loader,
        train_ds,
        val_ds,
        test_ds,
    ) = create_dataloaders(
        train_files,
        val_files,
        test_files,
        train_transforms,
        val_transforms,
        args,
        distributed=use_ddp,
        rank=rank,
        world_size=world_size,
    )

    r0_print("------------------- Step 4: define NET --------------------------")
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

    # Multi-GPU wrapping
    if use_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )
        r0_print(f"[DDP] world_size={world_size}, rank={rank}, local_rank={local_rank}, device={device}")
    else:
        n_gpus = torch.cuda.device_count()
        if device.type == "cuda" and n_gpus > 1:
            model = torch.nn.DataParallel(model)
            r0_print(f"[DataParallel] using {n_gpus} GPUs (CUDA_VISIBLE_DEVICES={args.gpu})")
        else:
            r0_print(f"[Single GPU/CPU] device={device}")

    loss_func = MultiTaskLossWrapper4(args.n_loss).to(device)
    loss_MSE = nn.MSELoss().to(device)
    loss_AE = nn.L1Loss().to(device)
    loss_BCE = nn.BCEWithLogitsLoss().to(device)

    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=args.lr_decay)

    # Visualize network structure (rank0 only; torchinfo optional)
    if _is_rank0(rank):
        try:
            examples = iter(train_loader)
            example_data = next(examples)
            base_model = _unwrap_model(model)
            _try_model_summary(base_model, example_data["input"].shape)
        except Exception as e:
            r0_print("[INFO] Could not run model summary:", str(e))

    r0_print("------------------- Step 5: model training --------------------------")

    writer = None
    if _is_rank0(rank):
        time_stamp = "{0:%Y-%m-%d-T%H-%M-%S/}".format(datetime.now()) + args.best_model_name
        writer = SummaryWriter(log_dir="runs/" + time_stamp)

    val_interval = 1
    best_metric_os = -1
    best_metric_pfs = -1
    best_metric_epoch = -1
    best_metric_os_epoch = -1

    epoch_loss_values = []
    val_epoch_loss_values = []
    test_epoch_loss_values = []

    t = trange(args.max_epochs, desc="AE survival -- epoch 0, avg loss: inf", leave=True) if _is_rank0(rank) else range(args.max_epochs)

    for epoch in t:
        # DDP: set epoch for sampler (shuffle changes each epoch)
        if use_ddp and hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        model.train()
        epoch_loss = 0.0
        step = 0

        # rank0 only metric accumulation
        train_OS = None
        train_OStime = None
        train_OSpred = None
        train_PFS = None
        train_PFStime = None
        train_PFSpred = None
        train_label = None
        train_labelpred = None

        if _is_rank0(rank) and hasattr(t, "set_description"):
            t.set_description(f"epoch {epoch + 1} started")

        for batch_data in train_loader:
            inputs = batch_data["input"].to(device, non_blocking=True)
            t_OS = batch_data["OS_status"].to(device, non_blocking=True)
            t_OStime = batch_data["OS_time"].to(device, non_blocking=True)
            t_PFS = batch_data["PFS_status"].to(device, non_blocking=True)
            t_PFStime = batch_data["PFS_time"].to(device, non_blocking=True)

            t_Age = batch_data["Age"].to(device, non_blocking=True)
            t_Gender = batch_data["Gender"].to(device, non_blocking=True)
            t_Child_Pugh = batch_data["Child_Pugh"].to(device, non_blocking=True)
            t_HBV = batch_data["HBV"].to(device, non_blocking=True)
            t_LMet = batch_data["PVTT"].to(device, non_blocking=True)
            t_AMet = batch_data["LungMet"].to(device, non_blocking=True)
            t_BoMet = batch_data["BoneMet"].to(device, non_blocking=True)
            t_BrMet = batch_data["Up to seven"].to(device, non_blocking=True)
            t_LNMet = batch_data["LNMet"].to(device, non_blocking=True)
            t_stage = batch_data["Stage"].to(device, non_blocking=True)

            t_Gender, t_Child_Pugh, t_HBV = (
                t_Gender.unsqueeze(1),
                t_Child_Pugh.unsqueeze(1),
                t_HBV.unsqueeze(1),
            )
            t_LMet, t_AMet, t_BoMet, t_BrMet, t_LNMet = (
                t_LMet.unsqueeze(1),
                t_AMet.unsqueeze(1),
                t_BoMet.unsqueeze(1),
                t_BrMet.unsqueeze(1),
                t_LNMet.unsqueeze(1),
            )
            t_stage = t_stage.unsqueeze(1)

            t_label = torch.cat(
                (t_Gender, t_Child_Pugh, t_HBV, t_LMet, t_AMet, t_BoMet, t_BrMet, t_LNMet, t_stage), 1
            )

            optimizer.zero_grad(set_to_none=True)

            if verbose and _is_rank0(rank):
                print("inputs shape", inputs.shape)

            output1, output2, output3, output4, output5, _ = model(inputs)

            # rank0: accumulate for epoch-level metrics (same as original behavior)
            if _is_rank0(rank):
                if step == 0:
                    train_OS = t_OS
                    train_OStime = t_OStime
                    train_OSpred = output1
                    train_PFS = t_PFS
                    train_PFStime = t_PFStime
                    train_PFSpred = output2
                    train_label = t_label
                    train_labelpred = output4
                else:
                    train_OS = torch.cat([train_OS, t_OS])
                    train_OStime = torch.cat([train_OStime, t_OStime])
                    train_OSpred = torch.cat([train_OSpred, output1])
                    train_PFS = torch.cat([train_PFS, t_PFS])
                    train_PFStime = torch.cat([train_PFStime, t_PFStime])
                    train_PFSpred = torch.cat([train_PFSpred, output2])
                    train_label = torch.cat((train_label, t_label), 0)
                    train_labelpred = torch.cat((train_labelpred, output4), 0)

            # L1 regularization (use base model params)
            l1_reg = None
            for W in _unwrap_model(model).parameters():
                if l1_reg is None:
                    l1_reg = torch.abs(W).sum()
                else:
                    l1_reg = l1_reg + torch.abs(W).sum()

            t_loss_os = surv_loss(t_OS, t_OStime, output1)
            t_loss_pfs = surv_loss(t_PFS, t_PFStime, output2)

            # auxiliary losses (keep your original logic)
            t_loss_age = loss_MSE(output3, t_Age.unsqueeze(1).float().log_() / 4.75)
            t_loss_label = loss_BCE(output4, t_label.float())
            t_loss_img = loss_AE(output5, inputs)

            t_loss = loss_func(t_loss_pfs, t_loss_age, t_loss_label, t_loss_img) * torch.log10(l1_reg) * t_loss_os

            if verbose and _is_rank0(rank):
                print(f"\n training epoch: {epoch}, step: {step}")
                print(f"\n os loss: {t_loss_os:.4f}, pfs loss: {t_loss_pfs:.4f}")
                print(f"\n age loss: {t_loss_age:.4f}, label loss: {t_loss_label:.4f}")
                print(f"\n AE loss: {t_loss_img:.4f}, L1 loss: {l1_reg:.4f}, total loss: {t_loss:.4f}")

            step += 1
            t_loss.backward()
            optimizer.step()
            epoch_loss += float(t_loss.item())

            # rank0 only tensorboard
            if writer is not None:
                epoch_len = len(train_ds) // train_loader.batch_size
                writer.add_scalar("train 5 overall loss: step", t_loss.item(), epoch_len * epoch + step)
                writer.add_scalar("train 6 os loss: step", t_loss_os.item(), epoch_len * epoch + step)
                writer.add_scalar("train 7 pfs loss: step", t_loss_pfs.item(), epoch_len * epoch + step)
                writer.add_scalar("train 8 age loss: step", t_loss_age.item(), epoch_len * epoch + step)
                writer.add_scalar("train 9 label loss: step", t_loss_label.item(), epoch_len * epoch + step)
                writer.add_scalar("train 0 AE loss: step", t_loss_img.item(), epoch_len * epoch + step)
                writer.add_scalar("train 10 L1 loss: step", l1_reg.item(), epoch_len * epoch + step)

        # DDP barrier
        if use_ddp:
            torch.distributed.barrier()

        # end of epoch - training metrics (rank0 only)
        if _is_rank0(rank):
            with torch.no_grad():
                epoch_loss /= max(step, 1)
                epoch_loss_values.append(epoch_loss)

                t_pvalue_OS = cox_log_rank(train_OSpred, train_OS, train_OStime)
                print("t_pvalue_OS", t_pvalue_OS)
                t_cindex_OS = CIndex_lifeline(train_OSpred, train_OS, train_OStime)
                print("t_cindex_OS", t_cindex_OS)
                writer.add_scalar("train 1 overall log rank OS: epoch", float(t_pvalue_OS), epoch)
                writer.add_scalar("train 2 overall c-index OS: epoch", float(t_cindex_OS), epoch)

                t_pvalue_PFS = cox_log_rank(train_PFSpred, train_PFS, train_PFStime)
                print("t_pvalue_PFS", t_pvalue_PFS)
                t_cindex_PFS = CIndex_lifeline(train_PFSpred, train_PFS, train_PFStime)
                print("t_cindex_PFS", t_cindex_PFS)
                writer.add_scalar("train 3 overall log rank PFS: epoch", float(t_pvalue_PFS), epoch)
                writer.add_scalar("train 4 overall c-index PFS: epoch", float(t_cindex_PFS), epoch)

                t_label_pred = train_labelpred >= 0.0
                t_acc = MultiLabel_Acc(t_label_pred, train_label)
                writer.add_scalar("train 11  Gender accuracy: epoch", float(t_acc[0]), epoch)
                writer.add_scalar("train 12  Child_Pugh accuracy: epoch", float(t_acc[1]), epoch)
                writer.add_scalar("train 13  HBV accuracy: epoch", float(t_acc[2]), epoch)
                writer.add_scalar("train 14  Liver Met accuracy: epoch", float(t_acc[3]), epoch)
                writer.add_scalar("train 15  Ad Met accuracy: epoch", float(t_acc[4]), epoch)
                writer.add_scalar("train 16  Bone Met accuracy: epoch", float(t_acc[5]), epoch)
                writer.add_scalar("train 17  Brain Met accuracy: epoch", float(t_acc[6]), epoch)
                writer.add_scalar("train 18  LN Met accuracy: epoch", float(t_acc[7]), epoch)
                writer.add_scalar("train 19  stage accuracy: epoch", float(t_acc[8]), epoch)

        # ------------------- validation + test ------------------- #
        if _is_rank0(rank) and ((epoch + 1) % val_interval == 0):
            model.eval()
            with torch.no_grad():
                # ------ validation ------ #
                val_OS = None
                val_OStime = None
                val_OSpred = None
                val_PFS = None
                val_PFStime = None
                val_PFSpred = None
                val_label = None
                val_labelpred = None

                val_epoch_loss = 0.0
                val_step = 0

                for val_data in val_loader:
                    v_inputs = val_data["input"].to(device, non_blocking=True)
                    v_OS = val_data["OS_status"].to(device, non_blocking=True)
                    v_OStime = val_data["OS_time"].to(device, non_blocking=True)
                    v_PFS = val_data["PFS_status"].to(device, non_blocking=True)
                    v_PFStime = val_data["PFS_time"].to(device, non_blocking=True)

                    v_Age = val_data["Age"].to(device, non_blocking=True)
                    v_Gender = val_data["Gender"].to(device, non_blocking=True)
                    v_Child_Pugh = val_data["Child_Pugh"].to(device, non_blocking=True)
                    v_HBV = val_data["HBV"].to(device, non_blocking=True)
                    v_LMet = val_data["PVTT"].to(device, non_blocking=True)
                    v_AMet = val_data["LungMet"].to(device, non_blocking=True)
                    v_BoMet = val_data["BoneMet"].to(device, non_blocking=True)
                    v_BrMet = val_data["Up to seven"].to(device, non_blocking=True)
                    v_LNMet = val_data["LNMet"].to(device, non_blocking=True)
                    v_stage = val_data["Stage"].to(device, non_blocking=True)

                    v_Gender, v_Child_Pugh, v_HBV = (
                        v_Gender.unsqueeze(1),
                        v_Child_Pugh.unsqueeze(1),
                        v_HBV.unsqueeze(1),
                    )
                    v_LMet, v_AMet, v_BoMet, v_BrMet, v_LNMet = (
                        v_LMet.unsqueeze(1),
                        v_AMet.unsqueeze(1),
                        v_BoMet.unsqueeze(1),
                        v_BrMet.unsqueeze(1),
                        v_LNMet.unsqueeze(1),
                    )
                    v_stage = v_stage.unsqueeze(1)

                    v_label = torch.cat(
                        (v_Gender, v_Child_Pugh, v_HBV, v_LMet, v_AMet, v_BoMet, v_BrMet, v_LNMet, v_stage), 1
                    )

                    val_output1, val_output2, val_output3, val_output4, val_output5, _ = model(v_inputs)

                    if val_step == 0:
                        val_OS = v_OS
                        val_OStime = v_OStime
                        val_OSpred = val_output1
                        val_PFS = v_PFS
                        val_PFStime = v_PFStime
                        val_PFSpred = val_output2
                        val_label = v_label
                        val_labelpred = val_output4
                    else:
                        val_OS = torch.cat([val_OS, v_OS])
                        val_OStime = torch.cat([val_OStime, v_OStime])
                        val_OSpred = torch.cat([val_OSpred, val_output1])
                        val_PFS = torch.cat([val_PFS, v_PFS])
                        val_PFStime = torch.cat([val_PFStime, v_PFStime])
                        val_PFSpred = torch.cat([val_PFSpred, val_output2])
                        val_label = torch.cat((val_label, v_label), 0)
                        val_labelpred = torch.cat((val_labelpred, val_output4), 0)

                    v_loss_os = surv_loss(v_OS, v_OStime, val_output1)
                    v_loss_pfs = surv_loss(v_PFS, v_PFStime, val_output2)
                    v_loss_age = loss_MSE(val_output3, v_Age.unsqueeze(1).float().log_() / 4.75)
                    v_loss_label = loss_BCE(val_output4, v_label.float())
                    v_loss_img = loss_AE(val_output5, v_inputs)

                    val_step += 1
                    val_epoch_loss += float(v_loss_os.item())

                    v_loss = loss_func(v_loss_pfs, v_loss_age, v_loss_label, v_loss_img) * v_loss_os
                    val_epoch_len = len(val_ds) // val_loader.batch_size + 1
                    writer.add_scalar("valid 5 overall loss: step", v_loss.item(), val_epoch_len * epoch + val_step)
                    writer.add_scalar("valid 6 os loss: step", v_loss_os.item(), val_epoch_len * epoch + val_step)
                    writer.add_scalar("valid 7 pfs loss: step", v_loss_pfs.item(), val_epoch_len * epoch + val_step)
                    writer.add_scalar("valid 8 age loss: step", v_loss_age.item(), val_epoch_len * epoch + val_step)
                    writer.add_scalar("valid 9 label loss: step", v_loss_label.item(), val_epoch_len * epoch + val_step)
                    writer.add_scalar("valid 0 AE loss: step", v_loss_img.item(), val_epoch_len * epoch + val_step)

                val_epoch_loss /= max(val_step, 1)
                scheduler.step(val_epoch_loss)
                val_epoch_loss_values.append(val_epoch_loss)

                v_pvalue_OS = cox_log_rank(val_OSpred, val_OS, val_OStime)
                print("v_pvalue_OS", v_pvalue_OS)
                v_cindex_OS = CIndex_lifeline(val_OSpred, val_OS, val_OStime)
                print("v_cindex_OS", v_cindex_OS)
                writer.add_scalar("valid 1 overall log rank OS: epoch", float(v_pvalue_OS), epoch)
                writer.add_scalar("valid 2 overall c-index OS: epoch", float(v_cindex_OS), epoch)
                writer.add_scalar("learning rate: epoch", optimizer.param_groups[0]["lr"], epoch + 1)

                v_pvalue_PFS = cox_log_rank(val_PFSpred, val_PFS, val_PFStime)
                print("v_pvalue_PFS", v_pvalue_PFS)
                v_cindex_PFS = CIndex_lifeline(val_PFSpred, val_PFS, val_PFStime)
                print("v_cindex_PFS", v_cindex_PFS)
                writer.add_scalar("valid 3 overall log rank PFS: epoch", float(v_pvalue_PFS), epoch)
                writer.add_scalar("valid 4 overall c-index PFS: epoch", float(v_cindex_PFS), epoch)

                v_label_pred = val_labelpred >= 0.0
                v_acc = MultiLabel_Acc(v_label_pred, val_label)
                writer.add_scalar("valid 11  Gender accuracy: epoch", float(v_acc[0]), epoch)
                writer.add_scalar("valid 12  Child_Pugh accuracy: epoch", float(v_acc[1]), epoch)
                writer.add_scalar("valid 13  HBV accuracy: epoch", float(v_acc[2]), epoch)
                writer.add_scalar("valid 14  Liver Met accuracy: epoch", float(v_acc[3]), epoch)
                writer.add_scalar("valid 15  Ad Met accuracy: epoch", float(v_acc[4]), epoch)
                writer.add_scalar("valid 16  Bone Met accuracy: epoch", float(v_acc[5]), epoch)
                writer.add_scalar("valid 17  Brain Met accuracy: epoch", float(v_acc[6]), epoch)
                writer.add_scalar("valid 18  LN Met accuracy: epoch", float(v_acc[7]), epoch)
                writer.add_scalar("valid 19  stage accuracy: epoch", float(v_acc[8]), epoch)

                # ------ test ------ #
                test_OS = None
                test_OStime = None
                test_OSpred = None
                test_PFS = None
                test_PFStime = None
                test_PFSpred = None
                test_label = None
                test_labelpred = None

                test_epoch_loss = 0.0
                test_step = 0

                for test_data in test_loader:
                    s_inputs = test_data["input"].to(device, non_blocking=True)
                    s_OS = test_data["OS_status"].to(device, non_blocking=True)
                    s_OStime = test_data["OS_time"].to(device, non_blocking=True)
                    s_PFS = test_data["PFS_status"].to(device, non_blocking=True)
                    s_PFStime = test_data["PFS_time"].to(device, non_blocking=True)

                    s_Age = test_data["Age"].to(device, non_blocking=True)
                    s_Gender = test_data["Gender"].to(device, non_blocking=True)
                    s_Child_Pugh = test_data["Child_Pugh"].to(device, non_blocking=True)
                    s_HBV = test_data["HBV"].to(device, non_blocking=True)
                    s_LMet = test_data["PVTT"].to(device, non_blocking=True)
                    s_AMet = test_data["LungMet"].to(device, non_blocking=True)
                    s_BoMet = test_data["BoneMet"].to(device, non_blocking=True)
                    s_BrMet = test_data["Up to seven"].to(device, non_blocking=True)
                    s_LNMet = test_data["LNMet"].to(device, non_blocking=True)
                    s_stage = test_data["Stage"].to(device, non_blocking=True)

                    s_Gender, s_Child_Pugh, s_HBV = (
                        s_Gender.unsqueeze(1),
                        s_Child_Pugh.unsqueeze(1),
                        s_HBV.unsqueeze(1),
                    )
                    s_LMet, s_AMet, s_BoMet, s_BrMet, s_LNMet = (
                        s_LMet.unsqueeze(1),
                        s_AMet.unsqueeze(1),
                        s_BoMet.unsqueeze(1),
                        s_BrMet.unsqueeze(1),
                        s_LNMet.unsqueeze(1),
                    )
                    s_stage = s_stage.unsqueeze(1)

                    s_label = torch.cat(
                        (s_Gender, s_Child_Pugh, s_HBV, s_LMet, s_AMet, s_BoMet, s_BrMet, s_LNMet, s_stage), 1
                    )

                    s_output1, s_output2, s_output3, s_output4, s_output5, _ = model(s_inputs)

                    if test_step == 0:
                        test_OS = s_OS
                        test_OStime = s_OStime
                        test_OSpred = s_output1
                        test_PFS = s_PFS
                        test_PFStime = s_PFStime
                        test_PFSpred = s_output2
                        test_label = s_label
                        test_labelpred = s_output4
                    else:
                        test_OS = torch.cat([test_OS, s_OS])
                        test_OStime = torch.cat([test_OStime, s_OStime])
                        test_OSpred = torch.cat([test_OSpred, s_output1])
                        test_PFS = torch.cat([test_PFS, s_PFS])
                        test_PFStime = torch.cat([test_PFStime, s_PFStime])
                        test_PFSpred = torch.cat([test_PFSpred, s_output2])
                        test_label = torch.cat((test_label, s_label), 0)
                        test_labelpred = torch.cat((test_labelpred, s_output4), 0)

                    s_loss_os = surv_loss(s_OS, s_OStime, s_output1)
                    s_loss_pfs = surv_loss(s_PFS, s_PFStime, s_output2)
                    s_loss_age = loss_MSE(s_output3, s_Age.unsqueeze(1).float().log_() / 4.75)
                    s_loss_label = loss_BCE(s_output4, s_label.float())
                    s_loss_img = loss_AE(s_output5, s_inputs)

                    test_step += 1
                    test_epoch_loss += float(s_loss_os.item())

                    s_loss = loss_func(s_loss_pfs, s_loss_age, s_loss_label, s_loss_img) * s_loss_os
                    test_epoch_len = len(test_ds) // test_loader.batch_size + 1
                    writer.add_scalar("test 5 overall loss: step", s_loss.item(), test_epoch_len * epoch + test_step)
                    writer.add_scalar("test 6 os loss: step", s_loss_os.item(), test_epoch_len * epoch + test_step)
                    writer.add_scalar("test 7 pfs loss: step", s_loss_pfs.item(), test_epoch_len * epoch + test_step)
                    writer.add_scalar("test 8 age loss: step", s_loss_age.item(), test_epoch_len * epoch + test_step)
                    writer.add_scalar("test 9 label loss: step", s_loss_label.item(), test_epoch_len * epoch + test_step)
                    writer.add_scalar("test 0 AE loss: step", s_loss_img.item(), test_epoch_len * epoch + test_step)

                test_epoch_loss /= max(test_step, 1)
                test_epoch_loss_values.append(test_epoch_loss)

                s_pvalue_OS = cox_log_rank(test_OSpred, test_OS, test_OStime)
                print("s_pvalue_OS", s_pvalue_OS)
                s_cindex_OS = CIndex_lifeline(test_OSpred, test_OS, test_OStime)
                print("s_cindex_OS", s_cindex_OS)
                writer.add_scalar("test 1 overall log rank OS: epoch", float(s_pvalue_OS), epoch)
                writer.add_scalar("test 2 overall c-index OS: epoch", float(s_cindex_OS), epoch)

                s_pvalue_PFS = cox_log_rank(test_PFSpred, test_PFS, test_PFStime)
                print("s_pvalue_PFS", s_pvalue_PFS)
                s_cindex_PFS = CIndex_lifeline(test_PFSpred, test_PFS, test_PFStime)
                print("s_cindex_PFS", s_cindex_PFS)
                writer.add_scalar("test 3 overall log rank PFS: epoch", float(s_pvalue_PFS), epoch)
                writer.add_scalar("test 4 overall c-index PFS: epoch", float(s_cindex_PFS), epoch)

                s_label_pred = test_labelpred >= 0.0
                s_acc = MultiLabel_Acc(s_label_pred, test_label)
                writer.add_scalar("test 11  Gender accuracy: epoch", float(s_acc[0]), epoch)
                writer.add_scalar("test 12  Child_Pugh accuracy: epoch", float(s_acc[1]), epoch)
                writer.add_scalar("test 13  HBV accuracy: epoch", float(s_acc[2]), epoch)
                writer.add_scalar("test 14  Liver Met accuracy: epoch", float(s_acc[3]), epoch)
                writer.add_scalar("test 15  Ad Met accuracy: epoch", float(s_acc[4]), epoch)
                writer.add_scalar("test 16  Bone Met accuracy: epoch", float(s_acc[5]), epoch)
                writer.add_scalar("test 17  Brain Met accuracy: epoch", float(s_acc[6]), epoch)
                writer.add_scalar("test 18  LN Met accuracy: epoch", float(s_acc[7]), epoch)
                writer.add_scalar("test 19  stage accuracy: epoch", float(s_acc[8]), epoch)

                metric_pfs = s_cindex_PFS
                metric_os = s_cindex_OS
                print("metric1_s_cindex_PFS", metric_pfs)
                print("metric2_s_cindex_OS", metric_os)

                if epoch > args.skip_epoch_model:
                    base_model = _unwrap_model(model)

                    if metric_pfs > best_metric_pfs:
                        best_metric_pfs = metric_pfs
                        best_metric_epoch = epoch + 1
                        torch.save(base_model.state_dict(), args.best_model_name + "_CV21_PFS.pth")

                    if metric_os > best_metric_os:
                        best_metric_os = metric_os
                        best_metric_os_epoch = epoch + 1
                        torch.save(base_model.state_dict(), args.best_model_name + "_CV21_OS.pth")
                        print(f"\n epoch {epoch + 1} saved new best metric model")

                    print(
                        f"\n current epoch: {epoch + 1} current val loss: {val_epoch_loss:.4f}"
                        f"\n best PFS c-index: {best_metric_pfs:.4f} at epoch {best_metric_epoch}"
                        f"\n best OS c-index: {best_metric_os:.4f} at epoch {best_metric_os_epoch}"
                    )

    if writer is not None:
        writer.close()

    if use_ddp:
        torch.distributed.destroy_process_group()

    if _is_rank0(rank):
        print("Training finished.")


if __name__ == "__main__":
    main()
