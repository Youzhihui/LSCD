import argparse
import datetime
import logging
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from datasets import weaklyCD
from utils import evaluate_CD, imutils
from utils.AverageMeter import AverageMeter
from utils.camutils_CD import cam_to_label2, multi_scale_cam2
from utils.optimizer import PolyWarmupAdamW
from models.Net2_MiT import TransWCD_dual
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
import matplotlib.pyplot as plt
from PIL import Image


import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
# LEVIR/DSIFN/WHU.yaml
parser.add_argument("--config",
                    default='CAM_configs/WHU.yaml',
                    type=str,
                    help="config")
parser.add_argument("--pooling", default="gmp", type=str, help="pooling method")
parser.add_argument("--crop_size", default=256, type=int, help="crop_size")
parser.add_argument("--scheme", default='transwcd_dual', type=str, help="transwcd_dual or transwcd_single")
parser.add_argument('--pretrained', default=True, type=bool, help="pretrained")
parser.add_argument('--checkpoint_path', default=False, type=str, help="checkpoint_path")

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda:0")


def setup_seed(seed, deterministic=True):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = deterministic


def setup_logger(filename='test.log'):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # remove existing handlers
    while logger.handlers:
        logger.handlers.pop()

    # create a file handler
    fHandler = logging.FileHandler(filename, mode='w')
    fHandler.setLevel(logging.INFO)

    logFormatter = logging.Formatter('%(asctime)s - %(filename)s - %(levelname)s: %(message)s')
    fHandler.setFormatter(logFormatter)
    logger.addHandler(fHandler)

    cHandler = logging.StreamHandler()
    cHandler.setFormatter(logFormatter)
    logger.addHandler(cHandler)
    return logger


def cal_eta(time0, cur_iter, total_iter):
    time_now = datetime.datetime.now()
    time_now = time_now.replace(microsecond=0)

    scale = (total_iter - cur_iter) / float(cur_iter)
    delta = (time_now - time0)
    eta = (delta * scale)
    time_fin = time_now + eta
    eta = time_fin.replace(microsecond=0) - time_now
    return str(delta), str(eta)


def get_down_size(ori_shape=(512, 512), stride=16):
    h, w = ori_shape
    _h = h // stride + 1 - ((h % stride) == 0)
    _w = w // stride + 1 - ((w % stride) == 0)
    return _h, _w


val_num = 1
def validate(model=None, data_loader=None, cfg=None):
    global val_num
    data = cfg.dataset.name_list_dir.split("/")[-1]
    preds, gts, cams2, cams3, cams4, cams_mean = [], [], [], [], [], []
    model.eval()
    cam_heatmap_dir = "./cam_vis/{}/baseline/heatmap/{}/".format(data, str(val_num))
    cam_label_dir = "./cam_vis/{}/baseline/label/{}/".format(data, str(val_num))
    val_num += 1
    cam_heatmap_dir_list = []
    cam_label_dir_list = []
    for s in ['scale2', 'scale3', 'scale4', 'mean']:
        path_dir = os.path.join(cam_heatmap_dir, s)
        os.makedirs(path_dir, exist_ok=True)
        cam_heatmap_dir_list.append(path_dir)
        path_dir = os.path.join(cam_label_dir, s)
        os.makedirs(path_dir, exist_ok=True)
        cam_label_dir_list.append(path_dir)

    with torch.no_grad():
        for _, data in tqdm(enumerate(data_loader),
                            total=len(data_loader), ncols=100, ascii=" >="):
            name, inputs_A, inputs_B, labels, cls_label = data

            inputs_A = inputs_A.to(device)
            inputs_B = inputs_B.to(device)
            labels = labels.to(device)
            cls_label = cls_label.to(device)
            if cls_label.sum() == 0:
                continue
            cls2, cls3, cls4 = model(inputs_A, inputs_B)
            cam2, cam3, cam4 = multi_scale_cam2(model, inputs_A, inputs_B, cfg.cam.scales)

            resized_cam2 = F.interpolate(cam2, size=labels.shape[1:], mode='bilinear', align_corners=False)
            cam_label2 = cam_to_label2(resized_cam2, cls_label=cls_label, bkg_score=cfg.cam.bkg_score2)

            resized_cam3 = F.interpolate(cam3, size=labels.shape[1:], mode='bilinear', align_corners=False)
            cam_label3 = cam_to_label2(resized_cam3, cls_label=cls_label, bkg_score=cfg.cam.bkg_score3)

            resized_cam4 = F.interpolate(cam4, size=labels.shape[1:], mode='bilinear', align_corners=False)
            cam_label4 = cam_to_label2(resized_cam4, cls_label=cls_label, bkg_score=cfg.cam.bkg_score4)

            cam_m = (resized_cam2 + resized_cam3 + resized_cam4) / 3
            bkg_score_m = (cfg.cam.bkg_score2 + cfg.cam.bkg_score3 + cfg.cam.bkg_score4) / 3
            cam_m_label = cam_to_label2(cam_m, cls_label=cls_label, bkg_score=bkg_score_m)

            if val_num % 1 == 0:
                resized_cam = resized_cam2[0, 0].cpu().numpy()
                cam_heatmap = Image.fromarray((plt.get_cmap('jet')(resized_cam)[:, :, :3] * 255).astype(np.uint8))
                cam_heatmap.save(os.path.join(os.path.join(cam_heatmap_dir_list[0], name[0])))
                resized_cam = resized_cam3[0, 0].cpu().numpy()
                cam_heatmap = Image.fromarray((plt.get_cmap('jet')(resized_cam)[:, :, :3] * 255).astype(np.uint8))
                cam_heatmap.save(os.path.join(os.path.join(cam_heatmap_dir_list[1], name[0])))
                resized_cam = resized_cam4[0, 0].cpu().numpy()
                cam_heatmap = Image.fromarray((plt.get_cmap('jet')(resized_cam)[:, :, :3] * 255).astype(np.uint8))
                cam_heatmap.save(os.path.join(os.path.join(cam_heatmap_dir_list[2], name[0])))
                resized_cam = cam_m[0, 0].cpu().numpy()
                cam_heatmap = Image.fromarray((plt.get_cmap('jet')(resized_cam)[:, :, :3] * 255).astype(np.uint8))
                cam_heatmap.save(os.path.join(os.path.join(cam_heatmap_dir_list[3], name[0])))

                pred_mask = Image.fromarray((cam_label2[0].cpu().numpy() * 255).astype(np.uint8))
                pred_mask.save(os.path.join(os.path.join(cam_label_dir_list[0], name[0])))
                pred_mask = Image.fromarray((cam_label3[0].cpu().numpy() * 255).astype(np.uint8))
                pred_mask.save(os.path.join(os.path.join(cam_label_dir_list[1], name[0])))
                pred_mask = Image.fromarray((cam_label4[0].cpu().numpy() * 255).astype(np.uint8))
                pred_mask.save(os.path.join(os.path.join(cam_label_dir_list[2], name[0])))
                pred_mask = Image.fromarray((cam_m_label[0].cpu().numpy() * 255).astype(np.uint8))
                pred_mask.save(os.path.join(os.path.join(cam_label_dir_list[3], name[0])))

            cams2 += list(cam_label2.cpu().numpy().astype(np.int16))
            cams3 += list(cam_label3.cpu().numpy().astype(np.int16))
            cams4 += list(cam_label4.cpu().numpy().astype(np.int16))
            cams_mean += list(cam_m_label.cpu().numpy().astype(np.int16))
            gts += list(labels.cpu().numpy().astype(np.int16))

    cam2_score = evaluate_CD.scores(gts, cams2)
    cam3_score = evaluate_CD.scores(gts, cams3)
    cam4_score = evaluate_CD.scores(gts, cams4)
    cam_mean_score = evaluate_CD.scores(gts, cams_mean)
    model.train()
    return cam2_score, cam3_score, cam4_score, cam_mean_score, labels


def train(cfg):
    num_workers = 10

    time0 = datetime.datetime.now()
    time0 = time0.replace(microsecond=0)

    train_dataset = weaklyCD.ClsDataset(
        root_dir=cfg.dataset.root_dir,
        name_list_dir=cfg.dataset.name_list_dir,
        split=cfg.train.split,
        stage='train',
        aug=True,
        rescale_range=cfg.dataset.rescale_range,
        crop_size=cfg.dataset.crop_size,
        img_fliplr=True,
        num_classes=cfg.dataset.num_classes,
    )

    val_dataset = weaklyCD.CDDataset(
        root_dir=cfg.dataset.root_dir,
        name_list_dir=cfg.dataset.name_list_dir,
        split=cfg.val.split,
        stage='val',
        aug=False,
        num_classes=cfg.dataset.num_classes,
    )

    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.train.batch_size,
                              # shuffle=True,
                              num_workers=num_workers,
                              pin_memory=False,
                              drop_last=True,
                              prefetch_factor=4)

    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=False,
                            drop_last=False)

    transwcd = TransWCD_dual(backbone=cfg.backbone.config,
                             stride=cfg.backbone.stride,
                             num_classes=cfg.dataset.num_classes,
                             embedding_dim=256,
                             pretrained=args.pretrained,
                             pooling=args.pooling, )

    param_groups = transwcd.get_param_groups()
    transwcd.to(device)

    writer = SummaryWriter(cfg.work_dir.logger_dir)
    print('writer:', writer)

    optimizer = PolyWarmupAdamW(
        params=[
            {
                "params": param_groups[0],
                "lr": cfg.optimizer.learning_rate,
                "weight_decay": cfg.optimizer.weight_decay,
            },
            {
                "params": param_groups[1],
                "lr": 0.0,  ## freeze norm layers
                "weight_decay": 0.0,
            },
            {
                "params": param_groups[2],
                "lr": cfg.optimizer.learning_rate * 10,
                "weight_decay": cfg.optimizer.weight_decay,
            },
        ],
        lr=cfg.optimizer.learning_rate,
        weight_decay=cfg.optimizer.weight_decay,
        betas=cfg.optimizer.betas,
        warmup_iter=cfg.scheduler.warmup_iter,
        max_iter=cfg.train.max_iters,
        warmup_ratio=cfg.scheduler.warmup_ratio,
        power=cfg.scheduler.power
    )

    train_loader_iter = iter(train_loader)
    avg_meter = AverageMeter()

    bkg_cls = torch.ones(size=(cfg.train.batch_size, 1))

    best_F1 = 0.0
    for n_iter in range(cfg.train.max_iters):

        try:
            img_name, inputs_A, inputs_B, cls_labels, img_box = next(train_loader_iter)
        except:
            train_loader_iter = iter(train_loader)
            img_name, inputs_A, inputs_B, cls_labels, img_box = next(train_loader_iter)

        inputs_A = inputs_A.to(device)
        inputs_B = inputs_B.to(device)

        cls_labels = cls_labels.to(device)

        cls_x2, cls_x3, cls_x4 = transwcd(inputs_A, inputs_B)

        cam2, cam3, cam4 = multi_scale_cam2(transwcd, inputs_A=inputs_A, inputs_B=inputs_B, scales=cfg.cam.scales)

        valid_cam, pred_cam = cam_to_label2(cam4.detach(), cls_label=cls_labels, img_box=img_box, ignore_mid=True,
                                           bkg_score=cfg.cam.bkg_score4)

        bkg_cls = bkg_cls.to(device)
        _cls_labels = torch.cat((bkg_cls, cls_labels), dim=1)

        cc_loss2 = F.binary_cross_entropy_with_logits(cls_x2, cls_labels)
        cc_loss3 = F.binary_cross_entropy_with_logits(cls_x3, cls_labels)
        cc_loss4 = F.binary_cross_entropy_with_logits(cls_x4, cls_labels)

        if n_iter <= cfg.train.cam_iters:
            loss = 1.0 * cc_loss4 + 1.0 * cc_loss3 + 1.0 * cc_loss2
        else:
            loss = 1.0 * cc_loss4 + 1.0 * cc_loss3 + 1.0 * cc_loss2

        avg_meter.add({'cc_loss2': cc_loss2.item() * 1.0})
        avg_meter.add({'cc_loss3': cc_loss3.item() * 1.0})
        avg_meter.add({'cc_loss4': cc_loss4.item() * 1.0})

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (n_iter + 1) % cfg.train.log_iters == 0:

            delta, eta = cal_eta(time0, n_iter + 1, cfg.train.max_iters)
            cur_lr = optimizer.param_groups[0]['lr']

            pred_cam = pred_cam.cpu().numpy().astype(np.int16)

            train_cc_loss2 = avg_meter.pop('cc_loss2')
            train_cc_loss3 = avg_meter.pop('cc_loss3')
            train_cc_loss4 = avg_meter.pop('cc_loss4')

            logger.info(
                "Iter: %d; Elasped: %s; ETA: %s; LR: %.3e; cc_loss4: %.4f; cc_loss3: %.4f; cc_loss2: %.4f" % (
                    n_iter + 1, delta, eta, cur_lr, train_cc_loss4, train_cc_loss3, train_cc_loss2))
            writer.add_scalar('train/cc_loss', train_cc_loss4, global_step=n_iter)

        if (n_iter + 1) % cfg.train.eval_iters == 0:
            ckpt_name = os.path.join(cfg.work_dir.ckpt_dir, "{}.pth".format(n_iter + 1))
            logging.info('Iter:{}  --> CD Validating...'.format(n_iter + 1))
            if (n_iter + 1) % 1000 == 0:
                torch.save(transwcd.state_dict(), ckpt_name)
            cam2_score, cam3_score, cam4_score, cam_mean_score, labels = validate(model=transwcd, data_loader=val_loader, cfg=cfg)
            # writer.add_scalar('val/OA', np.round(cam_score['OA'], 3), global_step=n_iter)
            # writer.add_scalar('val/F1', np.round(cam_score['f1'][1], 3), global_step=n_iter)
            # writer.add_scalar('val/precision', np.round(cam_score['precision'][1], 3), global_step=n_iter)
            # writer.add_scalar('val/iou', np.round(cam_score['iou'][1], 3), global_step=n_iter)
            # writer.add_scalar('val/recall', np.round(cam_score['recall'][1], 3), global_step=n_iter)
            # present_score = max(cam2_score['f1'][1], cam3_score['f1'][1], cam4_score['f1'][1], cam_mean_score['f1'][1])
            present_score = cam_mean_score['f1'][1]
            if present_score > best_F1:
                best_F1 = present_score
                best_iter = n_iter + 1
                best_ckpt_name = os.path.join(cfg.work_dir.ckpt_dir, "best.pth")
                torch.save(transwcd.state_dict(), best_ckpt_name)
            logger.info("cams score1: %s, \n:", cam2_score)
            logger.info("cams score2: %s, \n:", cam3_score)
            logger.info("cams score4: %s, \n", cam4_score)
            logger.info("cams mean score: %s, \n[best_iter]: %s", cam_mean_score, best_iter)

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
    for deterministic in [True]:
        for dataset in ["WHU"]:
            args = parser.parse_args()
            args.config = "configs/{}.yaml".format(dataset)
            cfg = OmegaConf.load(args.config)

            timestamp = "{0:%Y-%m-%d-%H-%M}".format(datetime.datetime.now())
            timestamp = "Net2_MiT_" + timestamp
            s2, s3, s4 = get_bkg_scores(dataset)
            cfg.cam.bkg_score2 = s2
            cfg.cam.bkg_score3 = s3
            cfg.cam.bkg_score4 = s4
            cfg.work_dir.dir = "work_dir_{}-{}-s2={}_s3={}_s4={}".format(dataset, deterministic, cfg.cam.bkg_score2,
                                                                         cfg.cam.bkg_score3, cfg.cam.bkg_score4)

            cfg.work_dir.ckpt_dir = os.path.join(cfg.work_dir.dir, cfg.work_dir.ckpt_dir, timestamp)
            cfg.work_dir.pred_dir = os.path.join(cfg.work_dir.dir, cfg.work_dir.pred_dir)
            cfg.work_dir.logger_dir = os.path.join(cfg.work_dir.dir, cfg.work_dir.logger_dir, timestamp)

            os.makedirs(cfg.work_dir.ckpt_dir, exist_ok=True)
            os.makedirs(cfg.work_dir.pred_dir, exist_ok=True)
            os.makedirs(cfg.work_dir.logger_dir, exist_ok=True)

            logger = setup_logger(filename=os.path.join(cfg.work_dir.dir, timestamp + '.log'))
            logger.info('\nargs: %s' % args)
            logger.info('\nconfigs: %s' % cfg)

            setup_seed(seed=1, deterministic=deterministic)
            train(cfg=cfg)
