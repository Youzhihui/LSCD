import PIL.Image
import torch
from skimage.segmentation import find_boundaries
from .segment_anything.utils.amg import (
    area_from_rle,
    box_xyxy_to_xywh,
    rle_to_mask,
    MaskData
)
import matplotlib.pyplot as plt
import numpy as np


def show_mask_data(mask_data, ax=None):
    assert isinstance(mask_data, MaskData)
    anns = []
    for idx in range(len(mask_data["rles"])):
        ann_i = {
            "segmentation": rle_to_mask(mask_data["rles"][idx]),
            "area": area_from_rle(mask_data["rles"][idx]),
        }
        if 'boxes' in mask_data._stats:
            ann_i['bbox'] = box_xyxy_to_xywh(mask_data["boxes"][idx]).tolist()
        anns.append(ann_i)

    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    if ax is None:
        ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        boundary = find_boundaries(m)
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        color_boundary = np.array([0., 1., 1., 0.8])
        img[m] = color_mask
        img[boundary] = color_boundary

        if 'label' in ann:
            x, y, w, h = ann['bbox']
            ax.text(
                x + w / 2,
                y + h / 2,
                ann['label'],
                bbox={
                    'facecolor': 'black',
                    'alpha': 0.8,
                    'pad': 0.7,
                    'edgecolor': 'none'
                },
                color='red',
                fontsize=4,
                verticalalignment='top',
                horizontalalignment='left'
            )
    ax.imshow(img)


def show_change_masks(img1, img2, change_masks):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
    axes[0].imshow(img1)
    show_mask_data(change_masks, axes[0])

    axes[1].imshow(img2)
    show_mask_data(change_masks, axes[1])

    axes[2].imshow(255 * np.ones_like(img1))
    show_mask_data(change_masks, axes[2])
    for ax in axes:
        ax.axis('off')

    return fig, axes


def our_change_masks(img1, img2, change_masks, gt):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharex=True, sharey=True)
    axes[0].imshow(img1)
    show_mask_data(change_masks, axes[0])

    axes[1].imshow(img2)
    show_mask_data(change_masks, axes[1])

    axes[2].imshow(255 * np.ones_like(img1))
    show_mask_data(change_masks, axes[2])

    axes[3].imshow(gt)
    show_mask_data(change_masks, axes[3])

    for ax in axes:
        ax.axis('off')

    return fig, axes


def our_change_masks2(img1, img2, change_masks, p, r, gt):
    fig, axes = plt.subplots(1, 6, figsize=(24, 4), sharex=True, sharey=True)
    axes[0].imshow(img1)
    # show_mask_data(change_masks, axes[0])

    axes[1].imshow(img2)
    # show_mask_data(change_masks, axes[1])

    axes[2].imshow(0 * np.ones_like(img1))
    show_mask_data(change_masks, axes[2])

    axes[3].imshow(p)
    show_mask_data(change_masks, axes[3])

    axes[4].imshow(r)
    show_mask_data(change_masks, axes[4])

    axes[5].imshow(gt)
    show_mask_data(change_masks, axes[5])

    for ax in axes:
        ax.axis('off')

    return fig, axes


def get_mask_data(mask_data, ax=None):
    assert isinstance(mask_data, MaskData)
    anns = []
    for idx in range(len(mask_data["rles"])):
        ann_i = {
            "segmentation": rle_to_mask(mask_data["rles"][idx]),
            "area": area_from_rle(mask_data["rles"][idx]),
        }
        if 'boxes' in mask_data._stats:
            ann_i['bbox'] = box_xyxy_to_xywh(mask_data["boxes"][idx]).tolist()
        anns.append(ann_i)

    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)

    img = np.zeros((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 3))
    for ann in sorted_anns:
        m = ann['segmentation']
        boundary = find_boundaries(m)
        color_mask = np.concatenate([np.random.random(3)])
        color_boundary = np.array([0., 1., 1.])
        img[m] = color_mask
        img[boundary] = color_boundary
    return img


def our_change_masks3(change_masks):
    img = get_mask_data(change_masks)
    img = (img * 255).astype(np.uint8)
    # alpha_channel = img[..., 3]
    # img[..., :3] = img[..., :3] * alpha_channel[..., np.newaxis]
    # img[alpha_channel == 0, :3] = [0, 0, 0]
    img = PIL.Image.fromarray(img.astype(np.uint8))
    return img


def refine_pseudo_label0(mask_data, pseudo_label=None, threshold=0.5):
    from scipy.ndimage import label
    assert isinstance(mask_data, MaskData)
    anns = []
    for idx in range(len(mask_data["rles"])):
        ann_i = {
            "segmentation": rle_to_mask(mask_data["rles"][idx]),
            "area": area_from_rle(mask_data["rles"][idx]),
        }
        if 'boxes' in mask_data._stats:
            ann_i['bbox'] = box_xyxy_to_xywh(mask_data["boxes"][idx]).tolist()
        anns.append(ann_i)

    if len(anns) == 0:
        return pseudo_label
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    updated_pred = pseudo_label.copy()
    labels, num_labels = label(pseudo_label)
    components = []
    for component_id in range(1, num_labels + 1):
        component_array = np.zeros_like(labels, dtype=np.uint8)
        component_array[labels == component_id] = 1
        components.append(component_array)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    masks_list = []
    for ann in sorted_anns:
        masks = []
        for component in components:
            mask = ann['segmentation'].astype(np.uint8)
            y_indices, x_indices = np.where(mask > 0)
            if len(y_indices) == 0 or len(x_indices) == 0:
                continue
            y_min, y_max = np.min(y_indices), np.max(y_indices)
            x_min, x_max = np.min(x_indices), np.max(x_indices)

            pred_region = pseudo_label[y_min:y_max + 1, x_min:x_max + 1]
            mask_region = mask[y_min:y_max + 1, x_min:x_max + 1]

            intersection = np.logical_and(pred_region > 0, mask_region > 0).sum()
            mask_area = mask_region.sum()
            overlap_ratio = intersection / mask_area if mask_area > 0 else 0

            if overlap_ratio > threshold:
                updated_pred[mask > 0] = 1
    return updated_pred


def refine_pseudo_label1(mask_data, pseudo_label=None, threshold=0.5):
    from scipy.ndimage import label
    from scipy import ndimage
    from PIL import Image
    assert isinstance(mask_data, MaskData)
    anns = []
    for idx in range(len(mask_data["rles"])):
        ann_i = {
            "segmentation": rle_to_mask(mask_data["rles"][idx]),
            "area": area_from_rle(mask_data["rles"][idx]),
        }
        if 'boxes' in mask_data._stats:
            ann_i['bbox'] = box_xyxy_to_xywh(mask_data["boxes"][idx]).tolist()
        anns.append(ann_i)

    if len(anns) == 0:
        return pseudo_label
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    updated_pred = pseudo_label.copy()
    labels, num_labels = label(pseudo_label)
    components = []
    for component_id in range(1, num_labels + 1):
        component_array = np.zeros_like(labels, dtype=np.uint8)
        component_array[labels == component_id] = 1
        components.append(component_array)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    masks_list = [list() for i in range(num_labels)]
    for ann in sorted_anns:
        scores = []
        mask = ann['segmentation'].astype(np.uint8)
        for component in components:
            intersection = np.logical_and(mask > 0, component > 0).sum()
            scores.append(intersection)
        index = np.argmax(np.array(scores))
        # if scores[index] > 0:
        if scores[index] / mask.sum() > 0.1 or scores[index] / components[index].sum() > 0.1:
            masks_list[index].append(mask)

    refined_components_list = []
    for i, component in enumerate(components):
        if len(masks_list[i]) == 0:
            if component.sum() / component.size > 0.001:
                refined_components_list.append(component.copy())
            continue
        refined_component = np.zeros_like(component).astype(np.bool)
        region_c = ndimage.find_objects(component)[0]
        c_y_min, c_y_max = region_c[0].start, region_c[0].stop
        c_x_min, c_x_max = region_c[1].start, region_c[1].stop
        for masks in masks_list:
            for m in masks:
                o_m = np.logical_and(m > 0, component > 0).sum() / m.sum()
                o_c = np.logical_and(m > 0, component > 0).sum() / component.sum()
                region_m = ndimage.find_objects(m)[0]
                m_y_min, m_y_max = region_m[0].start, region_m[0].stop
                m_x_min, m_x_max = region_m[1].start, region_m[1].stop
                if c_y_min >= m_y_min and c_x_min >= m_x_min and c_y_max <= m_y_max and c_x_max <= m_x_max:
                    np.logical_or(refined_component, m)
                    continue
                if o_m > 0.75:
                    np.logical_or(refined_component, m)
                if o_c > 0.45:
                    np.logical_or(refined_component, m)
        refined_components_list.append(np.logical_or(refined_component, component).astype(np.uint8))
    updated_pred = np.array(refined_components_list).sum(axis=0)
    updated_pred[updated_pred > 0] = 1
    return updated_pred


def refine_pseudo_label_EGYBCD(mask_data, pseudo_label=None, threshold=0.5):
    from scipy.ndimage import label
    from scipy import ndimage
    from PIL import Image
    assert isinstance(mask_data, MaskData)
    anns = []
    for idx in range(len(mask_data["rles"])):
        ann_i = {
            "segmentation": rle_to_mask(mask_data["rles"][idx]),
            "area": area_from_rle(mask_data["rles"][idx]),
        }
        if 'boxes' in mask_data._stats:
            ann_i['bbox'] = box_xyxy_to_xywh(mask_data["boxes"][idx]).tolist()
        anns.append(ann_i)

    if len(anns) == 0:
        return pseudo_label
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    labels, num_labels = label(pseudo_label)
    components = []
    for component_id in range(1, num_labels + 1):
        component_array = np.zeros_like(labels, dtype=np.uint8)
        component_array[labels == component_id] = 1
        components.append(component_array)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    masks_list = [list() for i in range(num_labels)]
    for ann in sorted_anns:
        scores = []
        mask = ann['segmentation'].astype(np.uint8)
        for component in components:
            intersection = np.logical_and(mask > 0, component > 0).sum()
            scores.append(intersection)
        index = np.argmax(np.array(scores))
        if scores[index] / mask.sum() > 0.25 or scores[index] / components[index].sum() > 0.25:
            masks_list[index].append(mask)

    refined_components_list = []
    for i, component in enumerate(components):
        if len(masks_list[i]) == 0:
            if component.sum() / component.size > 0.005:
                refined_components_list.append(component.copy())
            continue
        mask_con = np.array(masks_list[i]).sum(axis=0)
        mask_con[mask_con > 0] = 1
        intersection = np.logical_and(mask_con > 0, component > 0).sum()
        if intersection / component.sum() > 0.8:
            refined_component = mask_con.astype(np.bool)
        elif intersection / mask_con.sum() > 0.5 and intersection / component.sum() > 0.5:
            refined_component = mask_con.astype(np.bool)
        else:
            refined_component = component.copy().astype(np.bool)
            for mask in masks_list[i]:
                intersection = np.logical_and(mask > 0, component > 0).sum()
                if intersection / mask.sum() > 0.5 or intersection / component.sum() > 0.5:
                    refined_component = np.logical_or(refined_component, mask.astype(np.bool))
        refined_components_list.append(refined_component.astype(np.uint8))
    updated_pred = np.array(refined_components_list).sum(axis=0)
    updated_pred[updated_pred > 0] = 1
    return updated_pred


def refine_pseudo_label_LEVIR(mask_data, pseudo_label=None, threshold=0.5):
    from scipy.ndimage import label
    from scipy import ndimage
    from PIL import Image
    assert isinstance(mask_data, MaskData)
    anns = []
    for idx in range(len(mask_data["rles"])):
        ann_i = {
            "segmentation": rle_to_mask(mask_data["rles"][idx]),
            "area": area_from_rle(mask_data["rles"][idx]),
        }
        if 'boxes' in mask_data._stats:
            ann_i['bbox'] = box_xyxy_to_xywh(mask_data["boxes"][idx]).tolist()
        anns.append(ann_i)

    if len(anns) == 0:
        return pseudo_label
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    labels, num_labels = label(pseudo_label)
    components = []
    for component_id in range(1, num_labels + 1):
        component_array = np.zeros_like(labels, dtype=np.uint8)
        component_array[labels == component_id] = 1
        components.append(component_array)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    masks_list = [list() for i in range(num_labels)]
    for ann in sorted_anns:
        scores = []
        mask = ann['segmentation'].astype(np.uint8)
        for component in components:
            intersection = np.logical_and(mask > 0, component > 0).sum()
            scores.append(intersection)
        index = np.argmax(np.array(scores))
        if scores[index] / mask.sum() > 0.5 or scores[index] / components[index].sum() > 0.5:
            masks_list[index].append(mask)

    refined_components_list = []
    for i, component in enumerate(components):
        if len(masks_list[i]) == 0:
            if component.sum() / component.size > 0.003:
                refined_components_list.append(component.copy())
            continue
        mask_con = np.array(masks_list[i]).sum(axis=0)
        mask_con[mask_con > 0] = 1
        intersection = np.logical_and(mask_con > 0, component > 0).sum()
        if intersection / component.sum() > 0.85:
            refined_component = mask_con.astype(np.bool)
        elif intersection / mask_con.sum() > 0.65 and intersection / component.sum() > 0.65:
            refined_component = mask_con.astype(np.bool)
        else:
            refined_component = component.copy().astype(np.bool)
            for mask in masks_list[i]:
                intersection = np.logical_and(mask > 0, component > 0).sum()
                if intersection / mask.sum() > 0.5 or intersection / component.sum() > 0.5:
                    refined_component = np.logical_or(refined_component, mask.astype(np.bool))
        refined_components_list.append(refined_component.astype(np.uint8))
    if len(refined_components_list) == 0:
        updated_pred = pseudo_label
    else:
        updated_pred = np.array(refined_components_list).sum(axis=0)
        updated_pred[updated_pred > 0] = 1
    return updated_pred


def refine_pseudo_label_NewCD(mask_data, pseudo_label=None, threshold=0.5):
    from scipy.ndimage import label
    assert isinstance(mask_data, MaskData)
    anns = []
    for idx in range(len(mask_data["rles"])):
        ann_i = {
            "segmentation": rle_to_mask(mask_data["rles"][idx]),
            "area": area_from_rle(mask_data["rles"][idx]),
        }
        if 'boxes' in mask_data._stats:
            ann_i['bbox'] = box_xyxy_to_xywh(mask_data["boxes"][idx]).tolist()
        anns.append(ann_i)

    if len(anns) == 0:
        return pseudo_label
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)

    labels, num_labels = label(pseudo_label)
    components = []
    for component_id in range(1, num_labels + 1):
        component_array = np.zeros_like(labels)
        component_array[labels == component_id] = 1
        components.append(component_array.astype(np.uint8))
    if len(components) == 0:
        return pseudo_label

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    masks_list = [list() for i in range(num_labels)]

    for ann in sorted_anns:
        scores = []
        mask = ann['segmentation'].astype(np.uint8)
        for component in components:
            intersection = np.logical_and(mask > 0, component > 0).sum()
            scores.append(intersection)
        index = np.argmax(np.array(scores))
        if scores[index] / mask.sum() > 0.5 or scores[index] / components[index].sum() > 0.75:
        # if scores[index] / mask.sum() > 0.5:
            masks_list[index].append(mask)

    refined_components_list = []
    for i, component in enumerate(components):
        if len(masks_list[i]) == 0:
            if component.sum() / component.size > 0.003:
                refined_components_list.append(component.copy())
            continue
        mask_con = np.array(masks_list[i]).sum(axis=0)
        mask_con[mask_con > 0] = 1
        intersection = np.logical_and(mask_con > 0, component > 0).sum()
        if intersection / component.sum() > 0.75:
            refined_component = mask_con.astype(np.bool)
        elif intersection / mask_con.sum() > 0.5 and intersection / component.sum() > 0.5:
            refined_component = mask_con.astype(np.bool)
        else:
            refined_component = component.copy().astype(np.bool)
            for mask in masks_list[i]:
                intersection = np.logical_and(mask > 0, component > 0).sum()
                if intersection / mask.sum() > 0.5 or intersection / component.sum() > 0.5:
                    refined_component = np.logical_or(refined_component, mask.astype(np.bool))
        refined_components_list.append(refined_component.astype(np.uint8))
    if len(refined_components_list) == 0:
        updated_pred = pseudo_label
    else:
        updated_pred = np.array(refined_components_list).sum(axis=0)
        updated_pred[updated_pred > 0] = 1
    return updated_pred


def refine_pseudo_label(mask_data, pseudo_label=None, threshold=0.5):
    from scipy.ndimage import label
    assert isinstance(mask_data, MaskData)
    anns = []
    for idx in range(len(mask_data["rles"])):
        ann_i = {
            "segmentation": rle_to_mask(mask_data["rles"][idx]),
            "area": area_from_rle(mask_data["rles"][idx]),
        }
        if 'boxes' in mask_data._stats:
            ann_i['bbox'] = box_xyxy_to_xywh(mask_data["boxes"][idx]).tolist()
        anns.append(ann_i)

    if len(anns) == 0:
        return pseudo_label
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)

    labels, num_labels = label(pseudo_label)
    components = []
    for component_id in range(1, num_labels + 1):
        component_array = np.zeros_like(labels)
        component_array[labels == component_id] = 1
        components.append(component_array.astype(np.uint8))
    if len(components) == 0:
        return pseudo_label

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    masks_list = [list() for i in range(num_labels)]

    for ann in sorted_anns:
        scores = []
        mask = ann['segmentation'].astype(np.uint8)
        for component in components:
            intersection = np.logical_and(mask > 0, component > 0).sum()
            scores.append(intersection)
        index = np.argmax(np.array(scores))
        if scores[index] / mask.sum() > 0.5 or scores[index] / components[index].sum() > 0.75:
        # if scores[index] / mask.sum() > 0.5:
            masks_list[index].append(mask)

    refined_components_list = []
    for i, component in enumerate(components):
        if len(masks_list[i]) == 0:
            if component.sum() / component.size > 0.008:
                refined_components_list.append(component.copy())
            continue
        mask_con = np.array(masks_list[i]).sum(axis=0)
        mask_con[mask_con > 0] = 1
        intersection = np.logical_and(mask_con > 0, component > 0).sum()
        if intersection / component.sum() > 0.85:
            refined_component = mask_con.astype(np.bool)
        elif intersection / mask_con.sum() > 0.65 and intersection / component.sum() > 0.65:
            refined_component = mask_con.astype(np.bool)
        else:
            refined_component = component.copy().astype(np.bool)
            for mask in masks_list[i]:
                intersection = np.logical_and(mask > 0, component > 0).sum()
                if intersection / mask.sum() > 0.5 or intersection / component.sum() > 0.5:
                    refined_component = np.logical_or(refined_component, mask.astype(np.bool))
        refined_components_list.append(refined_component.astype(np.uint8))
    if len(refined_components_list) == 0:
        updated_pred = pseudo_label
    else:
        updated_pred = np.array(refined_components_list).sum(axis=0)
        updated_pred[updated_pred > 0] = 1
    return updated_pred


def vis_mask_pseudo(mask_data, pseudo_label=None, threshold=0.5):
    from scipy.ndimage import label
    assert isinstance(mask_data, MaskData)
    anns = []
    for idx in range(len(mask_data["rles"])):
        ann_i = {
            "segmentation": rle_to_mask(mask_data["rles"][idx]),
            "area": area_from_rle(mask_data["rles"][idx]),
        }
        if 'boxes' in mask_data._stats:
            ann_i['bbox'] = box_xyxy_to_xywh(mask_data["boxes"][idx]).tolist()
        anns.append(ann_i)

    if len(anns) == 0:
        return [], [], []

    labels, num_labels = label(pseudo_label)
    components = []
    for component_id in range(1, num_labels + 1):
        component_array = np.zeros_like(labels)
        component_array[labels == component_id] = 1
        components.append(component_array.astype(np.uint8))
    if len(components) == 0:
        return pseudo_label

    img = np.ones((anns[0]['segmentation'].shape[0], anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    masks_list = [list() for i in range(num_labels)]
    keep_list = [torch.zeros(size=(len(anns),)).bool() for i in range(num_labels)]

    for i, ann in enumerate(anns):
        scores = []
        mask = ann['segmentation'].astype(np.uint8)
        for component in components:
            intersection = np.logical_and(mask > 0, component > 0).sum()
            scores.append(intersection)
        index = np.argmax(np.array(scores))
        if scores[index] / mask.sum() > 0.5 or scores[index] / components[index].sum() > 0.75:
            masks_list[index].append(mask)
            keep_list[index][i] = True

    refined_components_list = []
    flag = torch.zeros(size=(len(components),)).bool().cpu().numpy().tolist()
    for i, component in enumerate(components):
        if len(masks_list[i]) == 0:
            if component.sum() / component.size > 0.003:
                flag[i] = True
                refined_components_list.append(component.copy())
            continue
        flag[i] = True
        mask_con = np.array(masks_list[i]).sum(axis=0)
        mask_con[mask_con > 0] = 1
        intersection = np.logical_and(mask_con > 0, component > 0).sum()
        if intersection / component.sum() > 0.8:
            refined_component = mask_con.astype(np.bool)
        elif intersection / mask_con.sum() > 0.5 and intersection / component.sum() > 0.65:
            refined_component = mask_con.astype(np.bool)
        else:
            refined_component = component.copy().astype(np.bool)
            for mask in masks_list[i]:
                intersection = np.logical_and(mask > 0, component > 0).sum()
                if intersection / mask.sum() > 0.5 or intersection / component.sum() > 0.5:
                    refined_component = np.logical_or(refined_component, mask.astype(np.bool))
        refined_components_list.append(refined_component.astype(np.uint8))
    keep_list_ = []
    components_ = []
    for i, f in enumerate(flag):
        if f:
            keep_list_.append(keep_list[i])
            components_.append(components[i])
    return keep_list_, components_, refined_components_list
