import torch
import numpy as np
import random


def reliable_pseudo_label(seg1_label, seg2_label, cam1_label, cam2_label, cls_labels):
    pseudo_label1 = torch.zeros_like(seg1_label, dtype=torch.long).to(seg1_label.device)
    pseudo_label2 = torch.zeros_like(seg2_label, dtype=torch.long).to(seg1_label.device)
    cls_labels = cls_labels.squeeze()
    cam_label_use_num1 = 0
    cam_label_use_num2 = 0
    pre_label_use_num1 = 0
    pre_label_use_num2 = 0
    for i in range(len(cls_labels)):
        if cls_labels[i] == 1:
            metric1 = meanIOU(num_classes=2)
            metric2 = meanIOU(num_classes=2)
            metric3 = meanIOU(num_classes=2)
            metric1.add_batch(seg1_label[i].cpu().numpy(), cam2_label[i].cpu().numpy())
            metric2.add_batch(cam1_label[i].cpu().numpy(), cam2_label[i].cpu().numpy())
            metric3.add_batch(seg2_label[i].cpu().numpy(), cam1_label[i].cpu().numpy())
            if metric1.evaluate()[-1] > metric2.evaluate()[-1]:
                pseudo_label1[i] = seg1_label[i]
                pre_label_use_num1 += 1
            else:
                pseudo_label1[i] = cam1_label[i]
                cam_label_use_num1 += 1
            if metric2.evaluate()[-1] > metric3.evaluate()[-1]:
                pseudo_label2[i] = cam2_label[i]
                cam_label_use_num2 += 1
            else:
                pseudo_label2[i] = seg2_label[i]
                pre_label_use_num2 += 1
    return pseudo_label1, pseudo_label2, cam_label_use_num1, pre_label_use_num1, cam_label_use_num2, pre_label_use_num2


def reliable_pseudo_label_new1(seg_label1, seg_label2, cam_label1, cam_label2, cls_labels, n_iter):
    cu_label1 = torch.zeros_like(seg_label1, dtype=torch.long).to(seg_label1.device)
    cu_label2 = torch.zeros_like(seg_label2, dtype=torch.long).to(seg_label2.device)
    cls_labels = cls_labels.squeeze()
    cam_label_use_num1 = 0
    cam_label_use_num2 = 0
    pre_label_use_num1 = 0
    pre_label_use_num2 = 0
    if n_iter < 6000:
        th = 0.8
    elif n_iter < 8000:
        th = 0.75
    elif n_iter < 10000:
        th = 0.7
    else:
        th = 0.65

    for i in range(len(cls_labels)):
        if cls_labels[i] == 1:
            metric1 = meanIOU(num_classes=2)
            metric2 = meanIOU(num_classes=2)

            metric1.add_batch(seg_label1[i].cpu().numpy(), cam_label1[i].cpu().numpy())
            metric2.add_batch(seg_label2[i].cpu().numpy(), cam_label2[i].cpu().numpy())
            if metric1.evaluate()[0][-1] > th:
                cu_label1[i] = seg_label1[i]
                pre_label_use_num1 += 1
            #     cu_label1[i] = torch.logical_or(seg_label1[i], cam_label1[i]).long()
            #     pre_label_use_num1 += 1
            #     cam_label_use_num1 += 1
            # elif metric1.evaluate()[0][-1] > 0.65:
            #     cu_label1[i] = seg_label1[i]
            #     pre_label_use_num1 += 1
            else:
                cu_label1[i] = cam_label1[i]
                cam_label_use_num1 += 1

            if metric2.evaluate()[0][-1] > th:
                cu_label2[i] = seg_label2[i]
                pre_label_use_num2 += 1
            #     cu_label2[i] = torch.logical_or(seg_label2[i], cam_label2[i]).long()
            #     pre_label_use_num2 += 1
            #     cam_label_use_num2 += 1
            # elif metric2.evaluate()[0][-1] > 0.65:
            #     cu_label2[i] = seg_label2[i]
            #     pre_label_use_num2 += 1
            else:
                cu_label2[i] = cam_label2[i]
                cam_label_use_num2 += 1
    return cu_label1, cu_label2, cam_label_use_num1, pre_label_use_num1, cam_label_use_num2, pre_label_use_num2


def reliable_pseudo_label_new2(seg_label1, seg_label2, cam_label1, cam_label2, cls_labels, n_iter):
    cu_label1 = torch.zeros_like(seg_label1, dtype=torch.long).to(seg_label1.device)
    cu_label2 = torch.zeros_like(seg_label2, dtype=torch.long).to(seg_label2.device)
    cls_labels = cls_labels.squeeze()
    cam_label_use_num1 = 0
    cam_label_use_num2 = 0
    pre_label_use_num1 = 0
    pre_label_use_num2 = 0
    # th = 0.4 * ((30000 - n_iter) / 30000) + 0.45
    th = 0.35 * ((30000 - n_iter) / 30000) + 0.45   # effective
    # th = 0.35 * ((30000 - n_iter) / 30000) + 0.5

    for i in range(len(cls_labels)):
        if cls_labels[i] == 1:
            metric1 = meanIOU(num_classes=2)
            metric2 = meanIOU(num_classes=2)

            metric1.add_batch(seg_label1[i].cpu().numpy(), cam_label1[i].cpu().numpy())
            metric2.add_batch(seg_label2[i].cpu().numpy(), cam_label2[i].cpu().numpy())
            if metric1.evaluate()[0][-1] > th:
                cu_label1[i] = seg_label1[i]
                pre_label_use_num1 += 1
            #     cu_label1[i] = torch.logical_or(seg_label1[i], cam_label1[i]).long()
            #     pre_label_use_num1 += 1
            #     cam_label_use_num1 += 1
            # elif metric1.evaluate()[0][-1] > 0.65:
            #     cu_label1[i] = seg_label1[i]
            #     pre_label_use_num1 += 1
            else:
                cu_label1[i] = cam_label1[i]
                cam_label_use_num1 += 1

            if metric2.evaluate()[0][-1] > th:
                cu_label2[i] = seg_label2[i]
                pre_label_use_num2 += 1
            #     cu_label2[i] = torch.logical_or(seg_label2[i], cam_label2[i]).long()
            #     pre_label_use_num2 += 1
            #     cam_label_use_num2 += 1
            # elif metric2.evaluate()[0][-1] > 0.65:
            #     cu_label2[i] = seg_label2[i]
            #     pre_label_use_num2 += 1
            else:
                cu_label2[i] = cam_label2[i]
                cam_label_use_num2 += 1
    return cu_label1, cu_label2, cam_label_use_num1, pre_label_use_num1, cam_label_use_num2, pre_label_use_num2


def reliable_pseudo_label_new2_ablation_study(seg_label1, seg_label2, cam_label1, cam_label2, cls_labels, n_iter, mode="static", thre=0.4):
    cu_label1 = torch.zeros_like(seg_label1, dtype=torch.long).to(seg_label1.device)
    cu_label2 = torch.zeros_like(seg_label2, dtype=torch.long).to(seg_label2.device)
    cls_labels = cls_labels.squeeze()
    cam_label_use_num1 = 0
    cam_label_use_num2 = 0
    pre_label_use_num1 = 0
    pre_label_use_num2 = 0
    if mode == "static":
        th = thre
    elif mode == "random":
        th = None
    elif mode == "dynamic":
        th = 0.5 * ((30000 - n_iter) / 30000) + 0.5

    for i in range(len(cls_labels)):
        if cls_labels[i] == 1:
            metric1 = meanIOU(num_classes=2)
            metric2 = meanIOU(num_classes=2)

            metric1.add_batch(seg_label1[i].cpu().numpy(), cam_label1[i].cpu().numpy())
            metric2.add_batch(seg_label2[i].cpu().numpy(), cam_label2[i].cpu().numpy())
            if th is None:
                if random.random() > 0.5:
                    cu_label1[i] = seg_label1[i]
                    pre_label_use_num1 += 1
                else:
                    cu_label1[i] = cam_label1[i]
                    cam_label_use_num1 += 1

                if random.random() > 0.5:
                    cu_label2[i] = seg_label2[i]
                    pre_label_use_num2 += 1
                else:
                    cu_label2[i] = cam_label2[i]
                    cam_label_use_num2 += 1
            else:
                if metric1.evaluate()[0][-1] > th:
                    cu_label1[i] = seg_label1[i]
                    pre_label_use_num1 += 1
                else:
                    cu_label1[i] = cam_label1[i]
                    cam_label_use_num1 += 1

                if metric2.evaluate()[0][-1] > th:
                    cu_label2[i] = seg_label2[i]
                    pre_label_use_num2 += 1
                else:
                    cu_label2[i] = cam_label2[i]
                    cam_label_use_num2 += 1
    return cu_label1, cu_label2, cam_label_use_num1, pre_label_use_num1, cam_label_use_num2, pre_label_use_num2


def reliable_pseudo_label_new_dynamic(seg_label1, seg_label2, cam_label1, cam_label2, cls_labels, n_iter, n1, n2):
    cu_label1 = torch.zeros_like(seg_label1, dtype=torch.long).to(seg_label1.device)
    cu_label2 = torch.zeros_like(seg_label2, dtype=torch.long).to(seg_label2.device)
    cls_labels = cls_labels.squeeze()
    cam_label_use_num1 = 0
    cam_label_use_num2 = 0
    pre_label_use_num1 = 0
    pre_label_use_num2 = 0
    th = n1 * ((30000 - n_iter) / 30000) + n2   # effective              yes

    for i in range(len(cls_labels)):
        if cls_labels[i] == 1:
            metric1 = meanIOU(num_classes=2)
            metric2 = meanIOU(num_classes=2)

            metric1.add_batch(seg_label1[i].cpu().numpy(), cam_label1[i].cpu().numpy())
            metric2.add_batch(seg_label2[i].cpu().numpy(), cam_label2[i].cpu().numpy())
            if metric1.evaluate()[0][-1] > th:
                cu_label1[i] = seg_label1[i]
                pre_label_use_num1 += 1
            #     cu_label1[i] = torch.logical_or(seg_label1[i], cam_label1[i]).long()
            #     pre_label_use_num1 += 1
            #     cam_label_use_num1 += 1
            # elif metric1.evaluate()[0][-1] > 0.65:
            #     cu_label1[i] = seg_label1[i]
            #     pre_label_use_num1 += 1
            else:
                cu_label1[i] = cam_label1[i]
                cam_label_use_num1 += 1

            if metric2.evaluate()[0][-1] > th:
                cu_label2[i] = seg_label2[i]
                pre_label_use_num2 += 1
            #     cu_label2[i] = torch.logical_or(seg_label2[i], cam_label2[i]).long()
            #     pre_label_use_num2 += 1
            #     cam_label_use_num2 += 1
            # elif metric2.evaluate()[0][-1] > 0.65:
            #     cu_label2[i] = seg_label2[i]
            #     pre_label_use_num2 += 1
            else:
                cu_label2[i] = cam_label2[i]
                cam_label_use_num2 += 1
    return cu_label1, cu_label2, cam_label_use_num1, pre_label_use_num1, cam_label_use_num2, pre_label_use_num2


def reliable_pseudo_label2(seg1_label, seg2_label, cam1_label, cam2_label, cls_label):
    batch_size = cam1_label.shape[0]
    blur_mask1 = torch.zeros(seg1_label.shape).long().cuda()
    blur_mask2 = torch.zeros(seg2_label.shape).long().cuda()
    for i in range(batch_size):
        s1 = seg1_label[i]
        c1 = cam1_label[i]
        blur_mask1[i] = (s1 != c1).long()
        s2 = seg2_label[i]
        c2 = cam2_label[i]
        blur_mask2[i] = (s2 != c2).long()
    return blur_mask1, blur_mask2


def reliable_pseudo_label3(seg1_label, seg2_label, cam1_label, cam2_label, cls_label):
    batch_size = seg1_label.shape[0]
    pseudo_label1 = torch.zeros_like(seg1_label, dtype=torch.long).cuda()
    pseudo_label2 = torch.zeros_like(seg2_label, dtype=torch.long).cuda()
    cls_label = cls_label.squeeze()
    for i in range(batch_size):
        if cls_label[i] == 1:
            metric1 = meanIOU(num_classes=2)
            metric2 = meanIOU(num_classes=2)
            metric3 = meanIOU(num_classes=2)

            metric1.add_batch(seg1_label[i].cpu().numpy(), cam2_label[i].cpu().numpy())
            metric2.add_batch(cam1_label[i].cpu().numpy(), cam2_label[i].cpu().numpy())
            metric3.add_batch(seg2_label[i].cpu().numpy(), cam1_label[i].cpu().numpy())
            if metric1.evaluate()[0][-1] > metric2.evaluate()[0][-1]:
                pseudo_label1[i] = seg1_label[i]
            else:
                pseudo_label1[i] = cam1_label[i]
            if metric2.evaluate()[0][-1] > metric3.evaluate()[0][-1]:
                pseudo_label2[i] = cam2_label[i]
            else:
                pseudo_label2[i] = seg2_label[i]
    return pseudo_label1, pseudo_label2


def reliable_pseudo_label_(seg1_label, seg2_label, cam1_label, cam2_label, cls_label):
    batch_size = cam1_label.shape[0]
    pseudo_label1 = torch.zeros(seg1_label.shape).long().cuda()
    pseudo_label2 = torch.zeros(seg2_label.shape).long().cuda()
    for i in range(batch_size):
        metric1 = meanIOU(num_classes=2)
        metric2 = meanIOU(num_classes=2)
        metric3 = meanIOU(num_classes=2)
        metric1.add_batch(seg1_label[i].cpu().numpy(), cam2_label[i].cpu().numpy())
        metric2.add_batch(cam1_label[i].cpu().numpy(), cam2_label[i].cpu().numpy())
        metric3.add_batch(seg2_label[i].cpu().numpy(), cam1_label[i].cpu().numpy())
        if metric1.evaluate()[0][-1] > metric2.evaluate()[0][-1]:
            pseudo_label1[i] = seg1_label[i]
        else:
            pseudo_label1[i] = cam1_label[i]
        if metric2.evaluate()[0][-1] > metric3.evaluate()[0][-1]:
            pseudo_label2[i] = cam2_label[i]
        else:
            pseudo_label2[i] = seg2_label[i]
    return pseudo_label1, pseudo_label2


class meanIOU:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))

    def _fast_hist(self, label_pred, label_true):
        mask = (label_true >= 0) & (label_true < self.num_classes)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) +
            label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist

    def add_batch(self, predictions, gts):
        for lp, lt in zip(predictions, gts):
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())

    def evaluate(self):
        iu = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist) + 1e-5)
        return iu, np.nanmean(iu)


def batch_pix_accuracy(predict, target):

    # _, predict = torch.max(output, 1)

    predict = predict.int() + 1
    target = target.int() + 1

    pixel_labeled = (target > 0).sum()
    pixel_correct = ((predict == target)*(target > 0)).sum()
    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct.cpu().numpy(), pixel_labeled.cpu().numpy()


def regularization_label(seg_1, seg_2, cls_labels):
    seg_score1, seg_label1 = torch.max(torch.softmax(seg_1, dim=1), dim=1)
    seg_score2, seg_label2 = torch.max(torch.softmax(seg_2, dim=1), dim=1)
    pseudo_label1 = seg_label1.clone()
    pseudo_label2 = seg_label2.clone()
    mean_score1 = torch.mean(seg_score1)
    mean_score2 = torch.mean(seg_score2)
    easy_mask = seg_label1 == seg_label2
    easy_mask1 = easy_mask & (seg_score1 > mean_score1)
    easy_mask2 = easy_mask & (seg_score2 > mean_score2)

    hard_mask = seg_label1 != seg_label2
    hard_mask1 = hard_mask & (seg_score1 > mean_score1)
    hard_mask2 = hard_mask & (seg_score2 > mean_score2)

    mid_mask1 = ~(easy_mask1 | hard_mask1)
    mid_mask2 = ~(easy_mask2 | hard_mask2)

    return pseudo_label1, pseudo_label2, easy_mask1, easy_mask2, mid_mask1, mid_mask2, hard_mask1, hard_mask2


def iom(seg1_label, seg2_label, cam1_label, cam2_label, cls_label):
    batch_size = cam1_label.shape[0]
    pseudo_label1 = torch.zeros_like(seg1_label, dtype=torch.long).cuda()
    pseudo_label2 = torch.zeros_like(seg2_label, dtype=torch.long).cuda()
    cls_label = cls_label.squeeze()
    for i in range(batch_size):
        if cls_label[i] == 1:
            metric1 = meanIOU(num_classes=2)
            metric2 = meanIOU(num_classes=2)

            metric1.add_batch(seg1_label[i].cpu().numpy(), cam1_label[i].cpu().numpy())
            metric2.add_batch(seg2_label[i].cpu().numpy(), cam2_label[i].cpu().numpy())
            if metric1.evaluate()[0][-1] > 0.7:
                pseudo_label1[i] = seg1_label[i]
            else:
                pseudo_label1[i] = cam1_label[i]
            if metric2.evaluate()[0][-1] > 0.7:
                pseudo_label2[i] = seg2_label[i]
            else:
                pseudo_label2[i] = cam2_label[i]
    return pseudo_label1, pseudo_label2


def eval_acc(cam_label_1, cam_label_2, pre_label_1, pre_label_2, cls_labels, seg_labels,
             eval_cam_lab1, eval_cam_lab2, eval_pre_lab1, eval_pre_lab2):
    pre_label_1 = torch.argmax(pre_label_1, dim=1)
    pre_label_2 = torch.argmax(pre_label_2, dim=1)
    eval_cam_lab1.update_cm(pr=cam_label_1.cpu().numpy(), gt=seg_labels.cpu().numpy())
    eval_cam_lab2.update_cm(pr=cam_label_2.cpu().numpy(), gt=seg_labels.cpu().numpy())
    eval_pre_lab1.update_cm(pr=pre_label_1.cpu().numpy(), gt=seg_labels.cpu().numpy())
    eval_pre_lab2.update_cm(pr=pre_label_2.cpu().numpy(), gt=seg_labels.cpu().numpy())


def eval_acc2(cam_label_1, cam_label_2, pre_label_1, pre_label_2, cls_labels, seg_labels,
              eval_cam_lab1, eval_cam_lab2, eval_pre_lab1, eval_pre_lab2, eval_cu1, eval_cu2):
    pre_label_1 = torch.argmax(pre_label_1, dim=1)
    pre_label_2 = torch.argmax(pre_label_2, dim=1)
    eval_cam_lab1.update_cm(pr=cam_label_1.cpu().numpy(), gt=seg_labels.cpu().numpy())
    eval_cam_lab2.update_cm(pr=cam_label_2.cpu().numpy(), gt=seg_labels.cpu().numpy())
    eval_pre_lab1.update_cm(pr=pre_label_1.cpu().numpy(), gt=seg_labels.cpu().numpy())
    eval_pre_lab2.update_cm(pr=pre_label_2.cpu().numpy(), gt=seg_labels.cpu().numpy())

    batch_size = cls_labels.shape[0]
    cu_label1 = torch.zeros_like(cam_label_1, dtype=torch.long).cuda()
    cu_label2 = torch.zeros_like(cam_label_2, dtype=torch.long).cuda()
    cls_labels = cls_labels.squeeze()
    for i in range(batch_size):
        if cls_labels[i] == 1:
            metric1 = meanIOU(num_classes=2)
            metric2 = meanIOU(num_classes=2)

            metric1.add_batch(cam_label_1[i].cpu().numpy(), pre_label_1[i].cpu().numpy())
            metric2.add_batch(cam_label_2[i].cpu().numpy(), pre_label_2[i].cpu().numpy())
            if metric1.evaluate()[0][-1] > 0.8:
                cu_label1[i] = pre_label_1[i]
            elif metric1.evaluate()[0][-1] > 0.6:
                cu_label1[i] = torch.logical_or(cam_label_1[i], pre_label_1[i]).long()
            else:
                cu_label1[i] = cam_label_1[i]

            if metric2.evaluate()[0][-1] > 0.8:
                cu_label2[i] = pre_label_2[i]
            elif metric2.evaluate()[0][-1] > 0.6:
                cu_label2[i] = torch.logical_or(cam_label_2[i], pre_label_2[i]).long()
            else:
                cu_label2[i] = cam_label_2[i]

    eval_cu1.update_cm(pr=cu_label1.cpu().numpy(), gt=seg_labels.cpu().numpy())
    eval_cu2.update_cm(pr=cu_label2.cpu().numpy(), gt=seg_labels.cpu().numpy())
