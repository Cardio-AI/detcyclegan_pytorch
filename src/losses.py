import os
import sys
import torch
import numpy as np
import random

# Imports from endo utils project
sys.path.append(os.path.abspath("./ext/endo_utils/data_utils/"))
import process_utils
import io_utils

#process_utils.seed_all(10)  # Seed models, random, and numpy

seed = 10
#print("[ Using Seed : ", seed, " ]")

np.random.seed(seed)
random.seed(seed)

# Torch seeds
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.cuda.manual_seed(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def dice_coeff(pred, target):
    smooth = 1.

    p_flat = pred.view(-1) # p is of N,C,H,W
    t_flat = target.view(-1) # t is of N, C, H, W
    intersection = (p_flat * t_flat).sum()
    return ((2. * intersection + smooth) / (p_flat.sum() + t_flat.sum() + smooth)).mean()


def dice_coeff_loss(pred, target):
    return 1 - dice_coeff(pred=pred, target=target)


def dice_loss(true, logits, eps=1e-7):
    """Computes the Sørensen–Dice loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the dice loss so we
    return the negated dice loss.
    Args:
        true: a tensor of shape [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        dice_loss: the Sørensen–Dice loss.
    """
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    dice_loss = (2. * intersection / (cardinality + eps)).mean()
    return (1 - dice_loss)


def dice_coeff_sample(pred, target, batch_size):
    smooth = 1.

    p_flat = pred.view(batch_size, -1) # p is of N, C*H*W
    t_flat = target.view(batch_size, -1) # t is of N, C*H*W
    intersection = (p_flat * t_flat).sum()
    return ((2. * intersection + smooth) / (p_flat.sum() + t_flat.sum() + smooth))


def f_beta_loss(pred, gt):
    smooth = 1.
    beta = 2.
    beta_sq = np.square(beta)

    p_flat = pred.view(-1)
    g_flat = gt.view(-1)

    intersection = (p_flat * g_flat).sum()
    g_backslash_p = ((1 - p_flat) * g_flat).sum()
    p_backslash_g = (p_flat * (1 - g_flat)).sum()

    f_beta = (((1 + beta_sq) * intersection + smooth) / (((1 + beta_sq)*intersection) + (beta_sq * g_backslash_p)
                                                         + p_backslash_g + smooth)).mean()
    return 1 - f_beta


def ss_loss(pred, gt):
    smooth = 1.
    r = 0.1
    mse = torch.nn.MSELoss()
    p_flat = pred.view(-1)
    g_flat = gt.view(-1)
    mse_loss = mse(input=p_flat, target=g_flat)

    sens_num = (mse_loss * g_flat).sum()
    sens_den = g_flat.sum()

    spec_num = (mse_loss * (1-g_flat)).sum()
    spec_den = (1 - g_flat).sum()

    ss_loss = (((r * sens_num) /sens_den + smooth) + (((1-r) * spec_num) /spec_den + smooth)).mean()
    return ss_loss

