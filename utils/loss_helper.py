import torch
import torch.nn as nn
import torch.nn.functional as F


# This loss does not affect the highest performance,
# but change the optimial background score (alpha)
def adaptive_min_pooling_loss(x):
    n, c, h, w = x.size()
    k = h * w // 4
    x = torch.max(x, dim=1)[0]
    y = torch.topk(x.view(n, -1), k=k, dim=-1, largest=False)[0]
    y = F.relu(y, inplace=False)
    loss = torch.sum(y) / (k * n)
    return loss


# ER Loss from https://github.com/YudeWang/SEAM
# --------------------------------------------------------------------------------------------------------
def get_er_loss(cam1, cam2, cam_rv1, cam_rv2, label):
    ns, cs, hs, ws = cam2.size()
    cam1 = F.interpolate(max_norm(cam1), size=(hs, ws), mode='bilinear', align_corners=True) * label
    # cam1 = F.softmax(cam1, dim=1) * label
    # cam2 = F.softmax(cam2, dim=1) * label
    cam2 = max_norm(cam2) * label
    loss_er = torch.mean(torch.abs(cam1[:, :-1, :, :] - cam2[:, :-1, :, :]))

    cam1[:, -1, :, :] = 1 - torch.max(cam1[:, :-1, :, :], dim=1)[0]
    cam2[:, -1, :, :] = 1 - torch.max(cam2[:, :-1, :, :], dim=1)[0]
    cam_rv1 = F.interpolate(max_norm(cam_rv1), size=(hs, ws), mode='bilinear', align_corners=True) * label
    cam_rv2 = max_norm(cam_rv2) * label
    tensor_ecr1 = torch.abs(max_onehot(cam2.detach()) - cam_rv1)  # *eq_mask
    tensor_ecr2 = torch.abs(max_onehot(cam1.detach()) - cam_rv2)  # *eq_mask
    loss_ecr1 = torch.mean(torch.topk(tensor_ecr1.view(ns, -1), k=int(21 * hs * ws * 0.2), dim=-1)[0])
    loss_ecr2 = torch.mean(torch.topk(tensor_ecr2.view(ns, -1), k=int(21 * hs * ws * 0.2), dim=-1)[0])
    loss_ecr = loss_ecr1 + loss_ecr2

    return loss_er, loss_ecr


# EPS Loss
# --------------------------------------------------------------------------------------------------------
def get_eps_loss(cam, saliency, label, tau, alpha, intermediate=True):
    b, c, h, w = cam.size()
    saliency = F.interpolate(saliency, size=(h, w))
    label_map = label.view(b, 20, 1, 1).expand(size=(b, 20, h, w)).bool()

    # Map selection
    label_map_fg = torch.zeros(size=(b, 21, h, w)).bool().cuda()
    label_map_bg = torch.zeros(size=(b, 21, h, w)).bool().cuda()

    label_map_bg[:, 20] = True
    label_map_fg[:, :-1] = label_map.clone()

    sal_pred = F.softmax(cam, dim=1)

    iou_saliency = (torch.round(sal_pred[:, :-1].detach()) * torch.round(saliency)).view(b, 20, -1).sum(-1) / \
                   (torch.round(sal_pred[:, :-1].detach()) + 1e-04).view(b, 20, -1).sum(-1)

    valid_channel = (iou_saliency > tau).view(b, 20, 1, 1).expand(size=(b, 20, h, w))

    label_fg_valid = label_map & valid_channel

    label_map_fg[:, :-1] = label_fg_valid
    label_map_bg[:, :-1] = label_map & (~valid_channel)

    # Saliency loss
    fg_map = torch.zeros_like(sal_pred).cuda()
    bg_map = torch.zeros_like(sal_pred).cuda()

    fg_map[label_map_fg] = sal_pred[label_map_fg]
    bg_map[label_map_bg] = sal_pred[label_map_bg]

    fg_map = torch.sum(fg_map, dim=1, keepdim=True)
    bg_map = torch.sum(bg_map, dim=1, keepdim=True)

    bg_map = torch.sub(1, bg_map)
    sal_pred = fg_map * alpha + bg_map * (1 - alpha)

    loss = F.mse_loss(sal_pred, saliency)

    if intermediate:
        return loss, fg_map, bg_map, sal_pred
    else:
        return loss


# RSC Loss
class SimMinLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(SimMinLoss, self).__init__()
        self.reduction = reduction

    def cos_sim(self, embedded_fg, embedded_bg):
        embedded_fg = F.normalize(embedded_fg, dim=1)
        embedded_bg = F.normalize(embedded_bg, dim=1)
        sim = torch.matmul(embedded_fg, embedded_bg.T)
        return torch.clamp(sim, min=0.0005, max=0.9995)

    def forward(self, embedded_bg, embedded_fg):
        sim = self.cos_sim(embedded_bg, embedded_fg)
        loss = -torch.log(1 - sim)

        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)


def max_norm(p, e=1e-5):
    if p.dim() == 3:
        C, H, W = p.size()
        max_v = torch.max(p.view(C, -1), dim=-1)[0].view(C, 1, 1)
        p = p / (max_v + e)
    elif p.dim() == 4:
        N, C, H, W = p.size()
        p = F.relu(p)
        max_v = torch.max(p.view(N, C, -1), dim=-1)[0].view(N, C, 1, 1)
        min_v = torch.min(p.view(N, C, -1), dim=-1)[0].view(N, C, 1, 1)
        p = F.relu(p - min_v - e) / (max_v - min_v + e)
    return p


def max_onehot(x):
    n, c, h, w = x.size()
    x_max = torch.max(x[:, 1:, :, :], dim=1, keepdim=True)[0]
    x[:, 1:, :, :][x[:, 1:, :, :] != x_max] = 0
    return x


def get_seg_loss(prediction, label, blur_mask=None, ignore_index=255, reduction="mean"):
    if blur_mask is not None:
        mask = torch.ones(label.shape).to(label.device)
        mask[blur_mask == 1] = 0.5
        mask[blur_mask == 0] = 1.5
    ce = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction=reduction)
    all_loss = ce(prediction, label)

    if blur_mask is not None:
        loss = (all_loss * mask).mean()
    else:
        loss = all_loss
    return loss


def get_mse_loss(prediction, label, blur_mask=None, reduction="mean"):
    prediction = torch.softmax(prediction, dim=1)
    label = torch.softmax(label, dim=1)
    if blur_mask is not None:
        mask = torch.ones(blur_mask.shape).to(label.device)
        mask[blur_mask == 1] = 0.5
        mask[blur_mask == 0] = 1.5
        mask = mask.unsqueeze(dim=1)
    mse = nn.MSELoss(reduction=reduction)
    all_loss = mse(prediction, label)

    if blur_mask is not None:
        loss = (all_loss * mask).mean()
    else:
        loss = all_loss
    return loss
