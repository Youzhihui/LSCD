# 将 val数据集换为test，取消val(如ChangeFormer)

# 将 val数据集换为test，取消val(如ChangeFormer)

import argparse
import os
from collections import OrderedDict
from PIL import Image, ImageDraw
from utils.camutils_CD import cam_to_label2, multi_scale_cam2, multi_scale_cam3
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from tqdm import tqdm
from datasets import weaklyCD
from utils import evaluate_CD
from models.Net3_MiT import TransWCD_dual

from utils import cam_helper
from scipy import ndimage as ndi
from skimage.feature import peak_local_max

parser = argparse.ArgumentParser()
# LEVIR/DSIFN/WHU.yaml
parser.add_argument("--config", default='configs/LEVIR.yaml',type=str,
                    help="config")
parser.add_argument("--save_dir", default="./results/LEVIR", type=str, help="save_dir")
parser.add_argument("--eval_set", default="train", type=str, help="eval_set")
parser.add_argument("--chckpoint_dir", default="./work_dir_LEVIR/checkpoints/Ours_a_2024-05-27-16-27", type=str, help="model_path")

parser.add_argument("--pooling", default="gmp", type=str, help="pooling method")
parser.add_argument("--bkg_score", default=0.45, type=float, help="bkg_score")
parser.add_argument("--resize_long", default=256, type=int, help="resize the long side (256 or 512)")


# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda:1")


def _test(model, dataset, test_scales=1.0):
    preds, gts, cams_mean = [], [], []
    point_dir = "{}_points".format(cfg.dataset.name_list_dir.split("/")[-1])
    os.makedirs(point_dir, exist_ok=True)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=cfg.test.batch_size, shuffle=False, num_workers=2,
                                              pin_memory=False)

    with torch.no_grad(), torch.cuda.device(0):
        model = model.to(device)
        for idx, data in tqdm(enumerate(data_loader)):
            ### 注意此处cls_label ###
            name, inputs_A, inputs_B, labels, cls_label = data

            inputs_A = inputs_A.to(device)
            inputs_B = inputs_B.to(device)

            b, c, h, w = inputs_A.shape
            labels = labels.to(device)
            cls_label = cls_label.to(device)

            _, _, h, w = inputs_A.shape
            ratio = args.resize_long / max(h, w)
            _h, _w = int(h * ratio), int(w * ratio)
            inputs_A = F.interpolate(inputs_A, size=(_h, _w), mode='bilinear', align_corners=False)

            _, _, h, w = inputs_B.shape
            ratio = args.resize_long / max(h, w)
            _h, _w = int(h * ratio), int(w * ratio)
            inputs_B = F.interpolate(inputs_B, size=(_h, _w), mode='bilinear', align_corners=False)
            if cls_label.sum() == 0:
                continue

            cam2, cam3, cam4 = multi_scale_cam3(model, inputs_A, inputs_B, cfg.cam.scales)
            resized_cam2 = F.interpolate(cam2, size=labels.shape[1:], mode='bilinear', align_corners=False)
            cam_label2 = cam_to_label2(resized_cam2, cls_label=cls_label, bkg_score=cfg.cam.bkg_score2)

            resized_cam3 = F.interpolate(cam3, size=labels.shape[1:], mode='bilinear', align_corners=False)
            cam_label3 = cam_to_label2(resized_cam3, cls_label=cls_label, bkg_score=cfg.cam.bkg_score3)

            resized_cam4 = F.interpolate(cam4, size=labels.shape[1:], mode='bilinear', align_corners=False)
            cam_label4 = cam_to_label2(resized_cam4, cls_label=cls_label, bkg_score=cfg.cam.bkg_score4)

            cam_m = (resized_cam2 + resized_cam3 + resized_cam4) / 3
            bkg_score_m = (cfg.cam.bkg_score2 + cfg.cam.bkg_score3 + cfg.cam.bkg_score4) / 3
            # stacked = torch.stack((resized_cam2, resized_cam3, resized_cam4), dim=0)
            # cam_m = torch.max(stacked, dim=0).values
            # bkg_score_m = 0.5
            cam_m_label = cam_to_label2(cam_m, cls_label=cls_label, bkg_score=bkg_score_m)

            cams_mean += list(cam_m_label.cpu().numpy().astype(np.int16))
            gts += list(labels.cpu().numpy().astype(np.int16))

            cam_target = resized_cam4[0, 0]
            cam_target_flat = cam_target.view(-1)
            argmax_indices = torch.argmax(cam_target_flat)
            coord_w = argmax_indices // _w
            coord_h = argmax_indices % _w
            peak_max = torch.cat((coord_w.view(1, 1), coord_h.view(1, 1)), dim=-1)
            peak_max = peak_max.cpu().detach().numpy()
            # Local maximums
            cam_target_np = cam_target.cpu().detach().numpy()
            cam_filtered = ndi.maximum_filter(cam_target_np, size=3, mode='constant')
            peaks_temp = peak_local_max(cam_filtered, min_distance=8)  # 5
            peaks_valid = peaks_temp[cam_target_np[peaks_temp[:, 0], peaks_temp[:, 1]] > 0.55]
            peaks = np.concatenate((peak_max, peaks_valid[1:]), axis=0)
            points = np.flip(peaks, axis=(-1))
            if labels.sum() > 0:
                binary_label = labels.squeeze().cpu().numpy()
                # mask_rgb = Image.fromarray(np.stack([binary_label * 255] * 3, axis=-1).astype(np.uint8))
                mask_rgb = Image.fromarray(np.zeros_like(binary_label).astype(np.uint8))
                draw = ImageDraw.Draw(mask_rgb)
                for point in points:
                    coord_w, coord_h = point
                    # draw.ellipse((coord_w - 1, coord_h - 1, coord_w + 1, coord_h + 1), fill=(255, 0, 0), width=1)
                    draw.ellipse((coord_w - 1, coord_h - 1, coord_w + 1, coord_h + 1), fill=(255,), width=1)
                # mask_rgb.show()
                mask_rgb.save(os.path.join(point_dir, name[0]))

        cam_mean_score = evaluate_CD.scores(gts, cams_mean)
        return cam_mean_score


def main(cfg):
    test_dataset = weaklyCD.CDDataset(
        root_dir=cfg.dataset.root_dir,
        name_list_dir=cfg.dataset.name_list_dir,
        split=args.eval_set,
        stage='test',
        aug=False,
        # ignore_index=cfg.dataset.ignore_index,
        num_classes=cfg.dataset.num_classes,
    )

    transwcd = TransWCD_dual(backbone=cfg.backbone.config,
                             stride=cfg.backbone.stride,
                             num_classes=cfg.dataset.num_classes,
                             embedding_dim=256,
                             # pretrained=args.pretrained,
                             pooling=args.pooling, )

    model_path = os.path.join(args.chckpoint_dir, "best.pth")
    trained_state_dict = torch.load(model_path, map_location="cpu")

    new_state_dict = OrderedDict()
    for k, v in trained_state_dict.items():
        if 'diff.0.bias' in k:
            k = k.replace('diff.0.bias', 'diff.bias')
        if 'diff.0.weight' in k:
            k = k.replace('diff.0.weight', 'diff.weight')
        new_state_dict[k] = v

    transwcd.load_state_dict(state_dict=new_state_dict, strict=True)  # True
    transwcd.eval()

    ###   test 输出 ###
    cam_mean_score = _test(model=transwcd, dataset=test_dataset, test_scales=[1.0])
    torch.cuda.empty_cache()
    print("cam_mean score:\n{}\n".format(cam_mean_score))

    return True


def get_bkg_scores(dataset):
    if dataset == "LEVIR":
        return [0.2, 0.3, 0.4]
    elif dataset == "EGYBCD":
        return [0.1, 0.1, 0.45]
    elif dataset == "WHU":
        return [0.1, 0.1, 0.45]
    else:
        return [0.1, 0.1, 0.45]


if __name__ == "__main__":
    dataset = "WHU"
    args = parser.parse_args()
    args.config = "configs/{}.yaml".format(dataset)
    args.save_dir = "./results/{}".format(dataset)
    s2, s3, s4 = get_bkg_scores(dataset)
    args.chckpoint_dir = "./work_dir_{}-True-s2={}_s3={}_s4={}/checkpoints/Net2_MiT_2024-12-10-06-38". \
        format(dataset, s2, s3, s4)
    args.bkg_score = 0.4

    cfg = OmegaConf.load(args.config)

    cfg.cam.bkg_score2 = s2
    cfg.cam.bkg_score3 = s3
    cfg.cam.bkg_score4 = s4
    print(cfg)
    print(args)

    args.save_dir = os.path.join(args.save_dir, args.eval_set)

    os.makedirs(args.save_dir + "/transwcd/prediction", exist_ok=True)
    os.makedirs(args.save_dir + "/transwcd/prediction_color", exist_ok=True)

    main(cfg=cfg)




