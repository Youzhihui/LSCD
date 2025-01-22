# 将 val数据集换为test，取消val(如ChangeFormer)

import argparse
import os
from collections import OrderedDict
from PIL import Image
from utils.camutils_CD import cam_to_label2, multi_scale_cam3
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from tqdm import tqdm
from datasets import weaklyCD
from utils import evaluate_CD
from models.Net3_MiT import TransWCD_dual

from utils import cam_helper

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


def _test(model, dataset, test_scales=1.0):
    preds, gts, cams2, cams3, cams4, cams_mean = [], [], [], [], [], []

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=cfg.test.batch_size, shuffle=False, num_workers=2,
                                              pin_memory=False)

    with torch.no_grad(), torch.cuda.device(0):
        model.cuda(0)
        for idx, data in tqdm(enumerate(data_loader)):
            ### 注意此处cls_label ###
            name, inputs_A, inputs_B, labels, cls_label = data

            inputs_A = inputs_A.cuda()
            inputs_B = inputs_B.cuda()

            b, c, h, w = inputs_A.shape
            labels = labels.cuda()
            cls_label = cls_label.cuda()

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
            cam_m_label = cam_to_label2(cam_m, cls_label=cls_label, bkg_score=bkg_score_m)

            cams2 += list(cam_label2.cpu().numpy().astype(np.int16))
            cams3 += list(cam_label3.cpu().numpy().astype(np.int16))
            cams4 += list(cam_label4.cpu().numpy().astype(np.int16))
            cams_mean += list(cam_m_label.cpu().numpy().astype(np.int16))
            gts += list(labels.cpu().numpy().astype(np.int16))

            # 以png格式保存cam结果
            cam_path = args.save_dir + '/transwcd/prediction/' + name[0]
            cam_img = Image.fromarray((cam_m_label.squeeze().cpu().numpy() * 255).astype(np.uint8))
            cam_img.save(cam_path)

            ### FN and FP color ###
            cam = cam_m_label.squeeze().cpu().numpy()
            labels = labels.squeeze().cpu().numpy()

            # Create RGB image from labels
            label_rgb = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)
            label_rgb[labels == 0] = [0, 0, 0]  # Background (black)
            label_rgb[labels == 1] = [255, 255, 255]  # Foreground (white)

            # Mark FN pixels as blue
            fn_pixels = np.logical_and(cam == 0, labels == 1)  # False Negatives
            label_rgb[fn_pixels] = [0, 0, 255]  # Blue

            # Mark FP pixels as red
            fp_pixels = np.logical_and(cam == 1, labels == 0)  # False Positives
            label_rgb[fp_pixels] = [255, 0, 0]  # Red

            # Save the labeled image
            label_with_fn_fp_path = args.save_dir + '/transwcd/prediction_color/' + name[0]
            label_with_fn_fp_img = Image.fromarray(label_rgb)
            label_with_fn_fp_img.save(label_with_fn_fp_path)
        cam2_score = evaluate_CD.scores(gts, cams2)
        cam3_score = evaluate_CD.scores(gts, cams3)
        cam4_score = evaluate_CD.scores(gts, cams4)
        cam_mean_score = evaluate_CD.scores(gts, cams_mean)
        return cam2_score, cam3_score, cam4_score, cam_mean_score


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
    cam2_score, cam3_score, cam4_score, cam_mean_score = _test(model=transwcd, dataset=test_dataset, test_scales=[1.0])
    torch.cuda.empty_cache()

    logFileLoc = args.chckpoint_dir + "/test.txt"
    if os.path.isfile(logFileLoc):
        logger = open(logFileLoc, 'a')
    else:
        logger = open(logFileLoc, 'w')
        logger.write(
            "%s\t%s\t%s\t%s\t%s\t%s\n" % ('Epoch', 'IoU', 'F1', 'P', 'R', 'OA'))
        logger.flush()
    logger.write("\nbkg_score2:{}\tbkg_score3:{}\tbkg_score4:{}\tbkg_score_mean:{}".format(
        cfg.cam.bkg_score2, cfg.cam.bkg_score3, cfg.cam.bkg_score4,
        (cfg.cam.bkg_score2 + cfg.cam.bkg_score3 + cfg.cam.bkg_score4) / 3))
    logger.write(
        "\ncam2_score:\n%s\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f" %
        ('Test', cam2_score['iou'][1], cam2_score['f1'][1], cam2_score['precision'][1],
         cam2_score['recall'][1], cam2_score['OA']))
    logger.write(
        "\ncam3_score:\n%s\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f" %
        ('Test', cam3_score['iou'][1], cam3_score['f1'][1], cam3_score['precision'][1],
         cam3_score['recall'][1], cam3_score['OA']))
    logger.write(
        "\ncam4_score:\n%s\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f" %
        ('Test', cam4_score['iou'][1], cam4_score['f1'][1], cam4_score['precision'][1],
         cam4_score['recall'][1], cam4_score['OA']))
    logger.write(
        "\ncam_mean_score:\n%s\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f" %
        ('Test', cam_mean_score['iou'][1], cam_mean_score['f1'][1], cam_mean_score['precision'][1],
         cam_mean_score['recall'][1], cam_mean_score['OA']))
    logger.write("\n****************************************************************\n")
    logger.flush()
    logger.close()

    print("cam2 score:\n{}\n".format(cam2_score))
    print("cam3 score:\n{}\n".format(cam3_score))
    print("cam4 score:\n{}\n".format(cam4_score))
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
    for dataset in ["EGYBCD"]:
        args = parser.parse_args()
        args.config = "configs/{}.yaml".format(dataset)
        args.save_dir = "./results/{}".format(dataset)
        s2, s3, s4 = get_bkg_scores(dataset)
        args.chckpoint_dir = "./work_dir_{}-True-s2={}_s3={}_s4={}/checkpoints/Net3_MiT_2024-12-10-06-38".\
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




