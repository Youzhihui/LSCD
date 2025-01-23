import os
import random
import shutil

import matplotlib.pyplot as plt
from skimage.io import imread
from PIL import Image
import numpy as np
from scipy.ndimage import label, center_of_mass
from shutil import copyfile
from torchange.models.segment_any_change import AnyChange, our_change_masks2, refine_pseudo_label

from skimage.segmentation import find_boundaries


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == "__main__":
    # loading dataset
    A_dir = ""
    B_dir = ""
    label_dir = ""
    point_dir = ""
    pred_dir = ""
    original_pred_dir = ""
    refined_pred_dir = ""

    os.makedirs(original_pred_dir, exist_ok=True)
    os.makedirs(refined_pred_dir, exist_ok=True)


    image_name_list = os.listdir(point_dir)

    # initialize AnyChange
    m = AnyChange('vit_h', sam_checkpoint='./pretrained/sam_vit_h_4b8939.pth')
    # customize the hyperparameters of SAM's mask generator
    m.make_mask_generator(
        points_per_side=32,
        stability_score_thresh=0.95,
    )
    # customize your AnyChange's hyperparameters
    m.set_hyperparameters(
        change_confidence_threshold=145,
        use_normalized_feature=True,
        bitemporal_match=True,
        object_sim_thresh=60,  # for point query
    )
    filter_area = 0.005
    gt_dir = ""
    save_dir = "".format(filter_area)
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    for name in image_name_list:
        A_path = os.path.join(A_dir, name)
        B_path = os.path.join(B_dir, name)
        label_path = os.path.join(label_dir, name)
        pred_path = os.path.join(pred_dir, name)

        img1 = np.array(Image.open(A_path).convert("RGB").resize(size=(256, 256)))
        img2 = np.array(Image.open(B_path).convert("RGB").resize(size=(256, 256)))
        gt = np.stack([np.array(Image.open(label_path).convert("L").resize(size=(256, 256)))] * 3, axis=-1)
        pred = np.array(Image.open(pred_path).convert("L").resize(size=(256, 256))) // 255

        point_path = os.path.join(point_dir, name)
        point = np.array(Image.open(point_path).convert("L").resize(size=(256, 256))) // 255
        labeled_array, num_features = label(point)
        centroids = center_of_mass(point, labeled_array, range(1, num_features + 1))
        points = np.array(centroids).astype(np.int16)[:, [1, 0]]

        t1 = np.ones(shape=(len(points), 1)).astype(np.int16) * 2
        points = np.concatenate((points, t1), axis=-1)
        changemasks = m.multi_points_match(xyts=points, img1=img1, img2=img2)
        if changemasks is None:
            continue

        # filter to
        w, h, c = img1.shape
        keep = (changemasks["areas"] / (h * w)) > filter_area
        changemasks.filter(keep)

        refine_pred = refine_pseudo_label(changemasks, pred)
        r = refine_pred.copy()

        refine_pred = Image.fromarray((refine_pred * 255).astype(np.uint8))
        refine_pred.save(os.path.join(refined_pred_dir, name))
        shutil.copy(os.path.join(pred_dir, name), os.path.join(original_pred_dir, name))

        p = np.stack([np.array(Image.open(pred_path).convert("L").resize(size=(256, 256)))] * 3, axis=-1)
        r = np.stack([np.array(r * 255)] * 3, axis=-1)
        fig, axes = our_change_masks2(img1, img2, changemasks, p, r, gt)
        m.clear_cached_embedding()
        del changemasks

        fig.savefig(os.path.join(save_dir, name))
        copyfile(os.path.join(label_dir, name), os.path.join(gt_dir, name))
        plt.close()
