import torch
import torch.nn.functional as F
import torch.nn as nn
#
#
def cross_entropy(input, target, weight=None, reduction='mean',ignore_index=255):
    """
    logSoftmax_with_loss
    :param input: torch.Tensor, N*C*H*W
    :param target: torch.Tensor, N*1*H*W,/ N*H*W
    :param weight: torch.Tensor, C
    :return: torch.Tensor [0]
    """
    target = target.long()
    if target.dim() == 4:
        target = torch.squeeze(target, dim=1)
    if input.shape[-1] != target.shape[-1]:
        input = F.interpolate(input, size=target.shape[1:], mode='bilinear',align_corners=True)

    return F.cross_entropy(input=input, target=target, weight=weight,
                           ignore_index=ignore_index, reduction=reduction)


import torch
import torch.nn.functional as F

def BCEDiceLoss(inputs, targets):
    """
    BCE + Dice Loss
    """
    inputs = inputs.float()
    targets = targets.float()

    if targets.dim() == 5 and targets.size(-1) == 3:
        targets = targets[..., 0]
    if targets.shape != inputs.shape:
        targets = targets.view(inputs.shape)


    with torch.no_grad():
        pos_ratio = targets.mean() + 1e-8
        dynamic_pos_weight = 1.0 / pos_ratio
        dynamic_pos_weight = torch.clamp(dynamic_pos_weight, min=5.0, max=15.0)

    bce = F.binary_cross_entropy_with_logits(
        inputs, targets, pos_weight=dynamic_pos_weight.to(inputs.device)
    )
    # ===========================================================
    smooth = 1e-5
    input_flat = inputs.view(-1)
    target_flat = targets.view(-1)
    intersection = (input_flat * target_flat).sum()
    dice_loss = 1 - (2. * intersection + smooth) / (input_flat.sum() + target_flat.sum() + smooth)

    return bce + dice_loss
