# losses.py
"""
Loss functions for survival analysis and multi-task learning.

This file contains:
- surv_loss: negative partial log-likelihood for Cox-style survival
- MultiTaskLossWrapper: learned uncertainty weights for multiple tasks
"""

import numpy as np
import torch
import torch.nn as nn


def surv_loss(event, time, risk):
    """
    Cox-style survival loss (negative partial log-likelihood).

    Args:
        event (Tensor): event indicators (0/1).
        time (Tensor): survival times.
        risk (Tensor): predicted risk scores.

    Returns:
        Tensor: scalar loss.
    """
    current_batch_len = len(time)
    R_matrix_train = np.zeros([current_batch_len, current_batch_len], dtype=int)
    for i in range(current_batch_len):
        for j in range(current_batch_len):
            R_matrix_train[i, j] = time[j] >= time[i]

    train_R = torch.FloatTensor(R_matrix_train)
    train_R = train_R.cuda()
    train_ystatus = event
    theta = risk.reshape(-1)
    exp_theta = torch.exp(theta)
    loss_nn = -torch.mean((theta - torch.log(torch.sum(exp_theta * train_R, dim=1))) * train_ystatus.float())
    return loss_nn


class MultiTaskLossWrapper(nn.Module):
    """
    Multi-task loss with learned uncertainty weights.

    Given task losses L0, L1, L2, the combined loss is:
        sum( exp(-log_var_i) * Li + log_var_i )
    This allows the network to learn the relative weights of each task.
    """

    def __init__(self, task_num):
        super(MultiTaskLossWrapper, self).__init__()
        self.task_num = task_num
        self.log_vars = nn.Parameter(torch.zeros((task_num)))

    def forward(self, L0, L1, L2):
        precision0 = torch.exp(-self.log_vars[0])
        loss0 = precision0 * L0 + self.log_vars[0]

        precision1 = torch.exp(-self.log_vars[1])
        loss1 = precision1 * L1 + self.log_vars[1]

        precision2 = torch.exp(-self.log_vars[2])
        loss2 = precision2 * L2 + self.log_vars[2]

        return loss0 + loss1 + loss2


class MultiTaskLossWrapper4(nn.Module):
    """
    Multi-task loss with learned uncertainty weights (4-task version).

    This version is used in Sub-network 3, where we have:
    - PFS survival loss
    - Age regression loss
    - Clinical multi-label loss
    - Reconstruction loss (autoencoder)

    Given task losses L0, L1, L2, L3, the combined loss is:
        sum( exp(-log_var_i) * Li + log_var_i )
    """

    def __init__(self, task_num):
        super(MultiTaskLossWrapper4, self).__init__()
        assert task_num == 4, "MultiTaskLossWrapper4 is designed for 4 tasks."
        self.task_num = task_num
        self.log_vars = nn.Parameter(torch.zeros((task_num)))

    def forward(self, L0, L1, L2, L3):
        precision0 = torch.exp(-self.log_vars[0])
        loss0 = precision0 * L0 + self.log_vars[0]

        precision1 = torch.exp(-self.log_vars[1])
        loss1 = precision1 * L1 + self.log_vars[1]

        precision2 = torch.exp(-self.log_vars[2])
        loss2 = precision2 * L2 + self.log_vars[2]

        precision3 = torch.exp(-self.log_vars[3])
        loss3 = precision3 * L3 + self.log_vars[3]

        return loss0 + loss1 + loss2 + loss3