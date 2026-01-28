# eval_utils.py
"""
Evaluation and inference utilities.

This file contains:
- model_eval_gpu: evaluate model on dataset with metrics and full outputs
- model_run_gpu: run model to obtain OS and PFS risk scores only
"""

import torch
from monai.data import DataLoader, Dataset, list_data_collate

from metrics import cox_log_rank, CIndex_lifeline


def model_eval_gpu(model, datafile, val_transforms, args):
    """
    Evaluate model on given dataset.

    This function reproduces the behavior of the original model_eval_gpu:
    - runs inference
    - aggregates OS/PFS outputs, events, durations, IDs, age outputs
    - prints OS/PFS C-index and log-rank p-value
    - returns all aggregated tensors
    """
    data_loader = DataLoader(
        Dataset(data=datafile, transform=val_transforms),
        batch_size=args.val_batch,
        shuffle=False,
        num_workers=args.num_workers_val,
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
    )

    torch.cuda.empty_cache()
    with torch.no_grad():
        model.eval()
        all_OSevents = None
        all_OSdurations = None
        all_PFSevents = None
        all_PFSdurations = None
        all_ID = None
        all_OSoutputs = None
        all_PFSoutputs = None
        all_Ageoutputs = None

        step = 0
        for batch_data in data_loader:
            inputs = batch_data["input"].cuda()
            OSevents = batch_data["OS_status"].cuda()
            OSdurations = batch_data["OS_time"].cuda()
            PFSevents = batch_data["PFS_status"].cuda()
            PFSdurations = batch_data["PFS_time"].cuda()
            ID_tensor = torch.tensor([int(id_str) for id_str in batch_data["ID"]])
            ID = ID_tensor.cuda()
            OSoutputs, PFSoutputs, Ageoutputs, _ = model(inputs)

            if step == 0:
                all_OSevents = OSevents
                all_OSdurations = OSdurations
                all_PFSevents = PFSevents
                all_PFSdurations = PFSdurations
                all_OSoutputs = OSoutputs
                all_PFSoutputs = PFSoutputs
                all_ID = ID
                all_Ageoutputs = Ageoutputs
            else:
                all_OSevents = torch.cat([all_OSevents, OSevents])
                all_OSdurations = torch.cat([all_OSdurations, OSdurations])
                all_PFSevents = torch.cat([all_PFSevents, PFSevents])
                all_PFSdurations = torch.cat([all_PFSdurations, PFSdurations])
                all_OSoutputs = torch.cat([all_OSoutputs, OSoutputs])
                all_PFSoutputs = torch.cat([all_PFSoutputs, PFSoutputs])
                all_ID = torch.cat([all_ID, ID])
                all_Ageoutputs = torch.cat([all_Ageoutputs, Ageoutputs])

            step += 1

        OS_pvalue = cox_log_rank(all_OSoutputs, all_OSevents, all_OSdurations)
        OS_cindex = CIndex_lifeline(all_OSoutputs, all_OSevents, all_OSdurations)
        PFS_pvalue = cox_log_rank(all_PFSoutputs, all_PFSevents, all_PFSdurations)
        PFS_cindex = CIndex_lifeline(all_PFSoutputs, all_PFSevents, all_PFSdurations)

    print(
        f"\n model evaluation"
        f"\n OS c-index: {OS_cindex:.4f} logrank p {OS_pvalue: .4f}"
        f"\n PFS c-index: {PFS_cindex:.4f} logrank p {PFS_pvalue: .4f}"
    )

    return (
        all_OSoutputs,
        all_PFSoutputs,
        all_OSevents,
        all_OSdurations,
        all_PFSevents,
        all_PFSdurations,
        all_ID,
        all_Ageoutputs,
    )


def model_run_gpu(model, datafile, val_transforms, args):
    """
    Run model on given dataset to get OS and PFS risk scores only.

    This reproduces the original model_run_gpu.
    """
    data_loader = DataLoader(
        Dataset(data=datafile, transform=val_transforms),
        batch_size=args.val_batch,
        shuffle=False,
        num_workers=args.num_workers_val,
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
    )

    torch.cuda.empty_cache()
    with torch.no_grad():
        model.eval()
        all_OSoutputs = None
        all_PFSoutputs = None

        step = 0
        for batch_data in data_loader:
            inputs = batch_data["input"].cuda()
            OSoutputs, PFSoutputs, _, _ = model(inputs)

            if step == 0:
                all_OSoutputs = OSoutputs
                all_PFSoutputs = PFSoutputs
            else:
                all_OSoutputs = torch.cat([all_OSoutputs, OSoutputs])
                all_PFSoutputs = torch.cat([all_PFSoutputs, PFSoutputs])

            step += 1

    return all_OSoutputs, all_PFSoutputs


def model_eval_gpu_ae(model, datafile, val_transforms, args):
    """
    Evaluate Sub-network 3 on given dataset.

    This function reproduces the behavior of the original model_eval_gpu:
    - runs inference
    - aggregates OS/PFS outputs, events, durations, IDs, age outputs, bottleneck features
    - prints OS/PFS C-index and log-rank p-value
    - returns all aggregated tensors
    """
    data_loader = DataLoader(
        Dataset(data=datafile, transform=val_transforms),
        batch_size=args.val_batch,
        shuffle=False,
        num_workers=args.num_workers_val,
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
    )

    torch.cuda.empty_cache()
    with torch.no_grad():
        model.eval()
        all_OSevents = None
        all_OSdurations = None
        all_PFSevents = None
        all_PFSdurations = None
        all_ID = None
        all_OSoutputs = None
        all_PFSoutputs = None
        all_Ageoutputs = None
        all_feature = None

        step = 0
        for batch_data in data_loader:
            inputs = batch_data["input"].cuda()
            OSevents = batch_data["OS_status"].cuda()
            OSdurations = batch_data["OS_time"].cuda()
            PFSevents = batch_data["PFS_status"].cuda()
            PFSdurations = batch_data["PFS_time"].cuda()
            ID_tensor = torch.tensor([int(id_str) for id_str in batch_data["ID"]])
            ID = ID_tensor.cuda()

            OSoutputs, PFSoutputs, Ageoutputs, _, _, feature = model(inputs)

            if step == 0:
                all_OSevents = OSevents
                all_OSdurations = OSdurations
                all_PFSevents = PFSevents
                all_PFSdurations = PFSdurations
                all_OSoutputs = OSoutputs
                all_PFSoutputs = PFSoutputs
                all_ID = ID
                all_Ageoutputs = Ageoutputs
                all_feature = feature
            else:
                all_OSevents = torch.cat([all_OSevents, OSevents])
                all_OSdurations = torch.cat([all_OSdurations, OSdurations])
                all_PFSevents = torch.cat([all_PFSevents, PFSevents])
                all_PFSdurations = torch.cat([all_PFSdurations, PFSdurations])
                all_OSoutputs = torch.cat([all_OSoutputs, OSoutputs])
                all_PFSoutputs = torch.cat([all_PFSoutputs, PFSoutputs])
                all_ID = torch.cat([all_ID, ID])
                all_Ageoutputs = torch.cat([all_Ageoutputs, Ageoutputs])
                all_feature = torch.cat([all_feature, feature], dim=0)

            step += 1

        OS_pvalue = cox_log_rank(all_OSoutputs, all_OSevents, all_OSdurations)
        OS_cindex = CIndex_lifeline(all_OSoutputs, all_OSevents, all_OSdurations)
        PFS_pvalue = cox_log_rank(all_PFSoutputs, all_PFSevents, all_PFSdurations)
        PFS_cindex = CIndex_lifeline(all_PFSoutputs, all_PFSevents, all_PFSdurations)

    print(
        f"\n model evaluation (Sub-network 3)"
        f"\n OS c-index: {OS_cindex:.4f} logrank p {OS_pvalue: .4f}"
        f"\n PFS c-index: {PFS_cindex:.4f} logrank p {PFS_pvalue: .4f}"
        f"\n dimension of bottleneck features {all_feature.size()}"
    )

    return (
        all_OSoutputs,
        all_PFSoutputs,
        all_OSevents,
        all_OSdurations,
        all_PFSevents,
        all_PFSdurations,
        all_ID,
        all_Ageoutputs,
        all_feature,
    )


def model_run_gpu_ae(model, datafile, val_transforms, args):
    """
    Run Sub-network 3 on given dataset to get OS/PFS risk scores and bottleneck features only.

    This reproduces the original model_run_gpu behavior.
    """
    data_loader = DataLoader(
        Dataset(data=datafile, transform=val_transforms),
        batch_size=args.val_batch,
        shuffle=False,
        num_workers=args.num_workers_val,
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
    )

    torch.cuda.empty_cache()
    with torch.no_grad():
        model.eval()
        all_OSoutputs = None
        all_PFSoutputs = None
        all_feature = None

        step = 0
        for batch_data in data_loader:
            inputs = batch_data["input"].cuda()
            OSoutputs, PFSoutputs, _, _, _, feature = model(inputs)

            if step == 0:
                all_OSoutputs = OSoutputs
                all_PFSoutputs = PFSoutputs
                all_feature = feature
            else:
                all_OSoutputs = torch.cat([all_OSoutputs, OSoutputs])
                all_PFSoutputs = torch.cat([all_PFSoutputs, PFSoutputs])
                all_feature = torch.cat([all_feature, feature], dim=0)

            step += 1

    return all_OSoutputs, all_PFSoutputs, all_feature


def model_eval_gpu_m3t(model, datafile, val_transforms, args):
    """
    Evaluate M3t on labelled dataset.

    Returns:
        all_OSoutputs, all_PFSoutputs,
        all_OSevents, all_OSdurations,
        all_PFSevents, all_PFSdurations,
        all_ID, all_Ageoutputs
    """
    data_loader = DataLoader(
        Dataset(data=datafile, transform=val_transforms),
        batch_size=args.val_batch,
        shuffle=False,
        num_workers=args.num_workers_val,
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
    )

    torch.cuda.empty_cache()
    with torch.no_grad():
        model.eval()
        all_OSevents = None
        all_OSdurations = None
        all_PFSevents = None
        all_PFSdurations = None
        all_ID = None
        all_OSoutputs = None
        all_PFSoutputs = None
        all_Ageoutputs = None

        step = 0
        for batch_data in data_loader:
            inputs = batch_data["input"].cuda()
            OSevents = batch_data["OS_status"].cuda()
            OSdurations = batch_data["OS_time"].cuda()
            PFSevents = batch_data["PFS_status"].cuda()
            PFSdurations = batch_data["PFS_time"].cuda()

            ID_tensor = torch.tensor([int(id_str) for id_str in batch_data["ID"]])
            ID = ID_tensor.cuda()

            OSoutputs, PFSoutputs, Ageoutputs, _ = model(inputs)

            if step == 0:
                all_OSevents = OSevents
                all_OSdurations = OSdurations
                all_PFSevents = PFSevents
                all_PFSdurations = PFSdurations
                all_OSoutputs = OSoutputs
                all_PFSoutputs = PFSoutputs
                all_ID = ID
                all_Ageoutputs = Ageoutputs
            else:
                all_OSevents = torch.cat([all_OSevents, OSevents])
                all_OSdurations = torch.cat([all_OSdurations, OSdurations])
                all_PFSevents = torch.cat([all_PFSevents, PFSevents])
                all_PFSdurations = torch.cat([all_PFSdurations, PFSdurations])
                all_OSoutputs = torch.cat([all_OSoutputs, OSoutputs])
                all_PFSoutputs = torch.cat([all_PFSoutputs, PFSoutputs])
                all_ID = torch.cat([all_ID, ID])
                all_Ageoutputs = torch.cat([all_Ageoutputs, Ageoutputs])

            step += 1

        OS_pvalue = cox_log_rank(all_OSoutputs, all_OSevents, all_OSdurations)
        OS_cindex = CIndex_lifeline(all_OSoutputs, all_OSevents, all_OSdurations)
        PFS_pvalue = cox_log_rank(all_PFSoutputs, all_PFSevents, all_PFSdurations)
        PFS_cindex = CIndex_lifeline(all_PFSoutputs, all_PFSevents, all_PFSdurations)

    print(
        f"\n M3t model evaluation"
        f"\n OS  c-index: {OS_cindex:.4f} logrank p {OS_pvalue: .4f}"
        f"\n PFS c-index: {PFS_cindex:.4f} logrank p {PFS_pvalue: .4f}"
    )

    return (
        all_OSoutputs,
        all_PFSoutputs,
        all_OSevents,
        all_OSdurations,
        all_PFSevents,
        all_PFSdurations,
        all_ID,
        all_Ageoutputs,
    )


def model_run_gpu_m3t(model, datafile, val_transforms, args):
    """
    Run M3t on image-only dataset to get OS/PFS risk scores.

    Returns:
        all_OSoutputs, all_PFSoutputs
    """
    data_loader = DataLoader(
        Dataset(data=datafile, transform=val_transforms),
        batch_size=args.val_batch,
        shuffle=False,
        num_workers=args.num_workers_val,
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
    )

    torch.cuda.empty_cache()
    with torch.no_grad():
        model.eval()
        all_OSoutputs = None
        all_PFSoutputs = None

        step = 0
        for batch_data in data_loader:
            inputs = batch_data["input"].cuda()
            OSoutputs, PFSoutputs, _, _ = model(inputs)

            if step == 0:
                all_OSoutputs = OSoutputs
                all_PFSoutputs = PFSoutputs
            else:
                all_OSoutputs = torch.cat([all_OSoutputs, OSoutputs])
                all_PFSoutputs = torch.cat([all_PFSoutputs, PFSoutputs])

            step += 1

    return all_OSoutputs, all_PFSoutputs
