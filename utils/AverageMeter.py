import numpy as np


class AverageMeter:
    def __init__(self, *keys):
        self.__data = dict()
        for k in keys:
            self.__data[k] = [0.0, 0]

    def add(self, dict):
        for k, v in dict.items():
            if k not in self.__data:
                self.__data[k] = [0.0, 0]
            self.__data[k][0] += v
            self.__data[k][1] += 1

    def get(self, *keys):
        if len(keys) == 1:
            return self.__data[keys[0]][0] / self.__data[keys[0]][1]
        else:
            v_list = [self.__data[k][0] / self.__data[k][1] for k in keys]
            return tuple(v_list)

    def pop(self, key=None):
        if key is None:
            for k in self.__data.keys():
                self.__data[k] = [0.0, 0]
        else:
            v = self.get(key)
            self.__data[key] = [0.0, 0]
            return v


###################       metrics      ###################
class MatrixMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg

    def get_scores(self):
        scores_dict = cm2score(self.sum)
        return scores_dict

    def clear(self):
        self.initialized = False


###################      cm metrics      ###################
class ConfuseMatrixMeter(MatrixMeter):
    """Computes and stores the average and current value"""

    def __init__(self, n_class):
        super(ConfuseMatrixMeter, self).__init__()
        self.n_class = n_class

    def update_cm(self, pr, gt, weight=1):
        """获得当前混淆矩阵，并计算当前F1得分，并更新混淆矩阵"""
        val = get_confuse_matrix(num_classes=self.n_class, label_gts=gt, label_preds=pr)
        self.update(val, weight)
        current_score = cm2F1(val)
        return current_score

    def get_scores(self):
        scores_dict = cm2score(self.sum)
        return scores_dict


def harmonic_mean(xs):
    harmonic_mean = len(xs) / sum((x + 1e-6) ** -1 for x in xs)
    return harmonic_mean


def cm2F1(confusion_matrix):
    hist = confusion_matrix
    tp = hist[1, 1]
    fn = hist[1, 0]
    fp = hist[0, 1]
    tn = hist[0, 0]
    # recall
    recall = tp / (tp + fn + np.finfo(np.float32).eps)
    # precision
    precision = tp / (tp + fp + np.finfo(np.float32).eps)
    # F1 score
    f1 = 2 * recall * precision / (recall + precision + np.finfo(np.float32).eps)
    return f1


def cm2score(confusion_matrix):
    hist = confusion_matrix
    # OA
    acc = np.diag(hist).sum() / (hist.sum() + np.finfo(np.float32).eps)  # acc = np.diag(hist).sum() / hist.sum()

    # recall
    recall = np.diag(hist) / (hist.sum(axis=1) + np.finfo(np.float32).eps)  # np.finfo(np.float32).eps
    # acc_cls = np.nanmean(recall)

    # precision
    precision = np.diag(hist) / (hist.sum(axis=0) + np.finfo(np.float32).eps)

    # F1 score
    F1 = 2 * recall * precision / (recall + precision + np.finfo(np.float32).eps)
    mean_F1 = np.nanmean(F1)
    # return mean_F1

    # IoU
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist) + np.finfo(np.float32).eps)
    mean_IoU = np.nanmean(iu)
    valid = hist.sum(axis=1) > 0  # added
    # mean_iu = np.nanmean(iu[valid])
    # freq = hist.sum(axis=1) / (hist.sum())
    # cls_iu = dict(zip(range(num_classes), iu))

    ###
    cls_iu = dict(zip(range(2), iu))
    cls_precision = dict(zip(range(2), precision))
    cls_recall = dict(zip(range(2), recall))
    cls_F1 = dict(zip(range(2), F1))

    score_dict = {'mF1': mean_F1, 'mIoU': mean_IoU, 'OA': acc, 'f1': cls_F1, 'iou': cls_iu}

    return score_dict



    # tp = hist[1, 1]
    # fn = hist[1, 0]
    # fp = hist[0, 1]
    # tn = hist[0, 0]
    # # acc
    # oa = (tp + tn) / (tp + fn + fp + tn + np.finfo(np.float32).eps)
    # # recall
    # recall = tp / (tp + fn + np.finfo(np.float32).eps)
    # # precision
    # precision = tp / (tp + fp + np.finfo(np.float32).eps)
    # # F1 score
    # f1 = 2 * recall * precision / (recall + precision + np.finfo(np.float32).eps)
    # # IoU
    # iou = tp / (tp + fp + fn + np.finfo(np.float32).eps)
    # # pre
    # pre = ((tp + fn) * (tp + fp) + (tn + fp) * (tn + fn)) / (tp + fp + tn + fn) ** 2
    # # kappa
    # kappa = (oa - pre) / (1 - pre)
    # score_dict = {'Kappa': kappa, 'IoU': iou, 'F1': f1, 'OA': oa, 'recall': recall, 'precision': precision, 'Pre': pre}
    # score_dict = {'OA': oa, 'IoU': iou, 'F1': f1, 'OA': oa, 'recall': recall, 'precision': precision, 'Pre': pre}
    # return score_dict


def get_confuse_matrix(num_classes, label_gts, label_preds):
    """计算一组预测的混淆矩阵"""

    def __fast_hist(label_gt, label_pred):
        """
        Collect values for Confusion Matrix
        For reference, please see: https://en.wikipedia.org/wiki/Confusion_matrix
        :param label_gt: <np.array> ground-truth
        :param label_pred: <np.array> prediction
        :return: <np.ndarray> values for confusion matrix
        """
        mask = (label_gt >= 0) & (label_gt < num_classes)
        hist = np.bincount(num_classes * label_gt[mask].astype(int) + label_pred[mask],
                           minlength=num_classes ** 2).reshape(num_classes, num_classes)
        return hist

    confusion_matrix = np.zeros((num_classes, num_classes))
    for lt, lp in zip(label_gts, label_preds):
        confusion_matrix += __fast_hist(lt.flatten(), lp.flatten())
    return confusion_matrix