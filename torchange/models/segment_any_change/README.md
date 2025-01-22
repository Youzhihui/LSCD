# Segment Any Change (NeurIPS 2024)

This is the official repository for the NeurIPS 2024 paper 
"_Segment Any Change_".  

Authors: 
[Zhuo Zheng](https://zhuozheng.top/)
[Yanfei Zhong](http://rsidea.whu.edu.cn/)
[Liangpei Zhang](http://www.lmars.whu.edu.cn/prof_web/zhangliangpei/rs/index.html)
[Stefano Ermon](https://cs.stanford.edu/~ermon/).

Abstract: Visual foundation models have achieved remarkable results in zero-shot image classification and segmentation, but zero-shot change detection remains an open problem.
In this paper, we propose the segment any change models (AnyChange), a new type of change detection model that supports zero-shot prediction and generalization on unseen change types and data distributions.
AnyChange is built on the segment anything model (SAM) via our training-free adaptation method, bitemporal latent matching.
By revealing and exploiting intra-image and inter-image semantic similarities in SAM's latent space, bitemporal latent matching endows SAM with zero-shot change detection capabilities in a training-free way. 
We also propose a point query mechanism to enable AnyChange's zero-shot object-centric change detection capability.

## Get Started
### Case 1: automatic mode (segment any change)
```python
import matplotlib.pyplot as plt
from skimage.io import imread
from torchange.models.segment_any_change import AnyChange, show_change_masks

# initialize AnyChange  
m = AnyChange('vit_h', sam_checkpoint='./sam_vit_h_4b8939.pth')
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
)

img1 = imread('https://github.com/Z-Zheng/pytorch-change-models/blob/main/demo_images/t1_img.png')
img2 = imread('https://github.com/Z-Zheng/pytorch-change-models/blob/main/demo_images/t2_img.png')

changemasks, _, _ = m.forward(img1, img2) # automatic mode
fig, axes = show_change_masks(img1, img2, changemasks)

plt.show()
```

### Case 2: point query mode (segment change of interest)
```python
import matplotlib.pyplot as plt
from skimage.io import imread
from torchange.models.segment_any_change import AnyChange, show_change_masks

# initialize AnyChange  
m = AnyChange('vit_h', sam_checkpoint='./sam_vit_h_4b8939.pth')
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
    object_sim_thresh=60, # for point query
)

img1 = imread('https://github.com/Z-Zheng/pytorch-change-models/blob/main/demo_images/t1_img.png')
img2 = imread('https://github.com/Z-Zheng/pytorch-change-models/blob/main/demo_images/t2_img.png')

# parameter description:
# xy: an absolute image coordinate.
# temporal: indicate which time the point belongs to
changemasks = m.single_point_match(xy=[926, 44], temporal=2, img1=img1, img2=img2)
fig, axes = show_change_masks(img1, img2, changemasks)

plt.show()
```



## Citation
If you find our project helpful, please cite our paper:
```
@inproceedings{
zheng2024anychange,
title={Segment Any Change},
author={Zhuo Zheng and Yanfei Zhong and Liangpei Zhang and Stefano Ermon},
booktitle={Advances in Neural Information Processing Systems},
year={2024},
}
```