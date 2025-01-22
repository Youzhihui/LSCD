# Changen2 (TPAMI 2024)

This is the official repository for IEEE TPAMI 2024 paper 
"_Changen2: Multi-Temporal Remote Sensing Generative Change Foundation Model_".  

Authors: 
[Zhuo Zheng](https://zhuozheng.top/)
[Stefano Ermon](https://cs.stanford.edu/~ermon/)
[Dongjun Kim](https://sites.google.com/view/dongjun-kim)
[Liangpei Zhang](http://www.lmars.whu.edu.cn/prof_web/zhangliangpei/rs/index.html)
[Yanfei Zhong](http://rsidea.whu.edu.cn/)

Abstract: Our understanding of the temporal dynamics of the Earth's surface has been significantly advanced by deep vision models, which often require a massive amount of labeled multi-temporal images for training.
However, collecting, preprocessing, and annotating multi-temporal remote sensing images at scale is non-trivial since it is expensive and knowledge-intensive.
In this paper, we present scalable multi-temporal change data generators based on generative models, which are cheap and automatic, alleviating these data problems. 
Our main idea is to simulate a stochastic change process over time.
We describe the stochastic change process as a probabilistic graphical model, namely the generative probabilistic change model (GPCM), which factorizes the complex simulation problem into two more tractable sub-problems, i.e., condition-level change event simulation and image-level semantic change synthesis. 
To solve these two problems, we present Changen2, a GPCM implemented with a resolution-scalable diffusion transformer which can generate time series of remote sensing images and corresponding semantic and change labels from labeled and even unlabeled single-temporal images.
Changen2 is a generative change foundation model that can be trained at scale via self-supervision, and is capable of producing change supervisory signals from unlabeled single-temporal images.
Unlike existing foundation models, our generative change foundation model synthesizes change data to train task-specific foundation models for change detection.
The resulting model possesses inherent zero-shot change detection capabilities and excellent transferability. 
Comprehensive experiments suggest Changen2 has superior spatiotemporal scalability in data generation, e.g., Changen2 model trained on 256$^2$ pixel single-temporal images can yield time series of any length and resolutions of 1,024^2 pixels.
Changen2 pre-trained models exhibit superior zero-shot performance (narrowing the performance gap to 3% on LEVIR-CD and approximately 10% on both S2Looking and SECOND, compared to fully supervised counterpart) and transferability across multiple types of change tasks, including ordinary and off-nadir building change, land-use/land-cover change, and disaster assessment.

## Get Started (TBD)

### Resolution-Scalable DiT models

```python
from torchange.models.changen2 import RSDiT_models
```

### Changen2 pre-trained ChangeStar (1x256) models

```python
from torchange.models.changen2 import changestar_1x256
```

### Synthetic Change Datasets
[Changen2-S1-15k](https://huggingface.co/datasets/EVER-Z/Changen2-S1-15k), a building change dataset with 15k pairs and 2 change types), 0.3-1m spatial resolution, RGB bands

[Changen2-S9-27k](https://huggingface.co/datasets/EVER-Z/Changen2-S9-27k), an urban land-use/landcover change dataset with 27k pairs and 38 change types), 0.25-0.5m spatial resolution, RGB bands


## Citation
If you find our project helpful, please cite our paper:
```
@article{zheng2024changen2,
  author={Zheng, Zhuo and Ermon, Stefano and Kim, Dongjun and Zhang, Liangpei and Zhong, Yanfei},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Changen2: Multi-Temporal Remote Sensing Generative Change Foundation Model}, 
  year={2024},
  volume={},
  number={},
  pages={1-17},
  doi={10.1109/TPAMI.2024.3475824}
}
```