import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from . import mix_transformer
from scipy.ndimage import label
import numpy as np


class GroupFusion(nn.Module):
    def __init__(self, in_d, out_d):
        super(GroupFusion, self).__init__()
        self.in_d = in_d
        self.out_d = out_d
        self.diff = nn.Sequential(
            nn.Conv2d(self.in_d, self.out_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_d),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x = torch.abs(x1 - x2)
        x = self.diff(x)
        return x


class TemporalFusionModule(nn.Module):
    def __init__(self, in_d=None, out_d=64):
        super(TemporalFusionModule, self).__init__()
        if in_d is None:
            in_d = [64, 128, 320, 512]
        self.in_d = in_d
        self.out_d = out_d
        # fusion
        self.gf_x1 = GroupFusion(self.in_d[0], self.out_d)
        self.gf_x2 = GroupFusion(self.in_d[1], self.out_d)
        self.gf_x3 = GroupFusion(self.in_d[2], self.out_d)
        self.gf_x4 = GroupFusion(self.in_d[3], self.out_d)

    def forward(self, x1_1, x1_2, x1_3, x1_4, x2_1, x2_2, x2_3, x2_4):
        # temporal fusion
        c1 = self.gf_x1(x1_1, x2_1)
        c2 = self.gf_x2(x1_2, x2_2)
        c3 = self.gf_x3(x1_3, x2_3)
        c4 = self.gf_x4(x1_4, x2_4)
        return c1, c2, c3, c4


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channel):
        super().__init__()
        self.in_d = in_channels
        self.out_d = out_channel

        self.conv = nn.Sequential(
            nn.Conv2d(self.in_d, self.out_d, 3, padding=1),
            nn.BatchNorm2d(self.out_d),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        res = self.conv(x)
        return res


class DecoderAD(nn.Module):
    def __init__(self, channels=None, mid_d=64):
        super(DecoderAD, self).__init__()
        self.channels = channels
        if self.channels is None:
            self.channels = [64, 128, 320, 512]
        self.mid_d = mid_d
        self.pro_d = mid_d // 2
        self.tfm = TemporalFusionModule(self.channels, self.mid_d)

        self.DB4 = DecoderBlock(self.mid_d, self.mid_d)
        self.DB3 = DecoderBlock(self.mid_d, self.mid_d)
        self.DB2 = DecoderBlock(self.mid_d, self.mid_d)
        self.DB1 = DecoderBlock(self.mid_d, self.mid_d)

        self.pre = nn.Sequential(
            nn.Conv2d(self.mid_d, self.pro_d, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.pro_d),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.pro_d, self.pro_d, kernel_size=1)
        )
        self.cls = nn.Sequential(
            nn.Conv2d(self.mid_d, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, _c1_1, _c2_1, _c3_1, _c4_1, _c1_2, _c2_2, _c3_2, _c4_2):
        ad1, ad2, ad3, ad4 = self.tfm(_c1_1, _c2_1, _c3_1, _c4_1, _c1_2, _c2_2, _c3_2, _c4_2)
        ad4 = self.DB4(ad4)
        ad3 = self.DB3(ad4 + ad3)
        ad2 = self.DB2(F.interpolate(ad3, scale_factor=(2, 2), mode="bilinear") + ad2)
        ad1 = self.DB1(F.interpolate(ad2, scale_factor=(2, 2), mode="bilinear") + ad1)
        ad0 = F.interpolate(ad1, scale_factor=(4, 4), mode="bilinear")
        feature = self.pre(ad0)
        out = self.cls(ad0)
        return feature, out


class TransWCD_dual(nn.Module):
    def __init__(self, backbone, num_classes=None, embedding_dim=256, stride=None, pretrained=None, pooling=None, ):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.feature_strides = [4, 8, 16, 32]
        self.stride = stride

        self.encoder = getattr(mix_transformer, backbone)(stride=self.stride)
        self.in_channels = self.encoder.embed_dims

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        if pretrained:
            state_dict = torch.load('pretrained/' + backbone + '.pth')
            state_dict.pop('head.weight')
            state_dict.pop('head.bias')
            self.encoder.load_state_dict(state_dict, )

        if pooling == "gmp":
            self.pooling4 = F.adaptive_max_pool2d
            self.pooling3 = F.adaptive_max_pool2d
            self.pooling2 = F.adaptive_max_pool2d
        elif pooling == "gap":
            self.pooling4 = F.adaptive_avg_pool2d
            self.pooling3 = F.adaptive_avg_pool2d
            self.pooling2 = F.adaptive_avg_pool2d

        # Difference Modules
        self.diff_c2 = nn.Sequential(
            nn.Conv2d(2 * c2_in_channels, c2_in_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.diff_c3 = nn.Sequential(
            nn.Conv2d(2 * c3_in_channels, c3_in_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.diff_c4 = nn.Sequential(
            nn.Conv2d(2 * c4_in_channels, c4_in_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.classifier2 = nn.Conv2d(in_channels=c2_in_channels, out_channels=self.num_classes - 1, kernel_size=1,
                                     bias=False)
        self.classifier3 = nn.Conv2d(in_channels=c3_in_channels, out_channels=self.num_classes - 1, kernel_size=1,
                                     bias=False)
        self.classifier4 = nn.Conv2d(in_channels=c4_in_channels, out_channels=self.num_classes - 1, kernel_size=1,
                                     bias=False)

    def get_param_groups(self):

        param_groups = [[], [], []]

        for name, param in list(self.encoder.named_parameters()):

            if "norm" in name:
                param_groups[1].append(param)
            else:
                param_groups[0].append(param)

        for param in list(self.diff_c4.parameters()):
            param_groups[2].append(param)
        for param in list(self.diff_c3.parameters()):
            param_groups[2].append(param)
        for param in list(self.diff_c2.parameters()):
            param_groups[2].append(param)
        param_groups[2].append(self.classifier4.weight)
        param_groups[2].append(self.classifier3.weight)
        param_groups[2].append(self.classifier2.weight)

        return param_groups

    def forward(self, x1, x2, cam_only=False, ):
        b, c, h, w = x1.shape
        _x1 = self.encoder(x1)
        _x2 = self.encoder(x2)

        _c1_1, _c2_1, _c3_1, _c4_1 = _x1
        _c1_2, _c2_2, _c3_2, _c4_2 = _x2

        # classification
        _c4 = self.diff_c4(torch.cat((_c4_1, _c4_2), dim=1))
        _c3 = self.diff_c3(torch.cat((_c3_1, _c3_2), dim=1))
        _c2 = self.diff_c2(torch.cat((_c2_1, _c2_2), dim=1))

        if cam_only:
            cam4 = F.conv2d(_c4, self.classifier4.weight).detach()
            # return cam4
            cam3 = F.conv2d(_c3, self.classifier3.weight).detach()
            cam2 = F.conv2d(_c2, self.classifier2.weight).detach()
            return cam2, cam3, cam4
        cls_x4 = self.pooling4(_c4, (1, 1))
        cls_x4 = self.classifier4(cls_x4)
        cls_x4 = cls_x4.view(-1, self.num_classes - 1)

        cls_x3 = self.pooling3(_c3, (1, 1))
        cls_x3 = self.classifier3(cls_x3)
        cls_x3 = cls_x3.view(-1, self.num_classes - 1)

        cls_x2 = self.pooling2(_c2, (1, 1))
        cls_x2 = self.classifier2(cls_x2)
        cls_x2 = cls_x2.view(-1, self.num_classes - 1)

        # return cls_x2, cls_x3, cls_x4
        with torch.no_grad():
            cam = F.conv2d(_c4, self.classifier.weight).detach()
            _cam = F.interpolate(cam, size=(h, w), mode='bilinear', align_corners=False)
            _cam = F.relu(_cam)
            _cam = _cam + F.adaptive_max_pool2d(-_cam, (1, 1))
            _cam /= F.adaptive_max_pool2d(_cam, (1, 1)) + 1e-5
            pseudo_labels = (_cam > 0.6).long()
            masks = torch.ones_like(pseudo_labels).to(pseudo_labels.device)
            for i in range(b):
                pse_lab = pseudo_labels[i][0]
                component, num = label(pse_lab.cpu().numpy())
                if num > 2:
                    if num >= 4:
                        indices = random.sample(range(1, num+1), num//2)
                    else:
                        index = random.randint(1, num)
                        indices = [index]
                    ms = np.ones_like(component)
                    for v in indices:
                        ms[component == v] = 0
                elif num >= 1:
                    index = random.randint(1, num)
                    ms = np.ones_like(component)
                    ms[component == index] = 0
                    coord = np.argwhere(ms == 0)
                    min_row, min_col = coord.min(axis=0)
                    max_row, max_col = coord.max(axis=0)
                    mid_row, mid_col = (min_row + max_row) // 2, (min_col + max_col) // 2
                    index = random.randint(1, 4)
                    ms = np.ones_like(component)
                    if index == 1:
                        ms[min_row:mid_row, min_col:max_col] = 0
                    elif index == 2:
                        ms[mid_row:max_row, min_col:max_col] = 0
                    elif index == 3:
                        ms[min_row:max_row, min_col:mid_col] = 0
                    elif index == 4:
                        ms[min_row:max_row, mid_col:max_col] = 0
                else:
                    h_s = random.randint(0, h // 2 - 1)
                    w_s = random.randint(0, w // 2 - 1)
                    ms = np.ones_like(component)
                    ms[h_s:h_s+100, w_s:w_s+100] = 0
                masks[i, 0] = torch.from_numpy(ms).to(masks.device).float()
        a = x1.clone() * masks
        b = x2.clone() * masks
        _a = self.encoder(a)
        _b = self.encoder(b)
        _c1_a, _c2_a, _c3_a, _c4_a = _a
        _c1_b, _c2_b, _c3_b, _c4_b = _b
        _c4_ = self.diff_c4(torch.cat((_c4_a, _c4_b), dim=1))
        _c3_ = self.diff_c3(torch.cat((_c3_a, _c3_b), dim=1))
        _c2_ = self.diff_c2(torch.cat((_c2_a, _c2_b), dim=1))

        cls_x4_ = self.pooling4(_c4_, (1, 1))
        cls_x4_ = self.classifier4(cls_x4_)
        cls_x4_ = cls_x4_.view(-1, self.num_classes - 1)

        cls_x3_ = self.pooling3(_c3_, (1, 1))
        cls_x3_ = self.classifier3(cls_x3_)
        cls_x3_ = cls_x3_.view(-1, self.num_classes - 1)

        cls_x2_ = self.pooling2(_c2_, (1, 1))
        cls_x2_ = self.classifier2(cls_x2_)
        cls_x2_ = cls_x2_.view(-1, self.num_classes - 1)
        return cls_x2, cls_x3, cls_x4, cls_x2_, cls_x3_, cls_x4_


if __name__ == "__main__":
    pretrained_weights = torch.load('pretrained/mit_b1.pth')
    transwcd = TransWCD_dual('mit_b1', num_classes=2, embedding_dim=256, pretrained=True)
    transwcd._param_groups()
    dummy_input = torch.rand(2, 3, 256, 256)
    transwcd(dummy_input)
