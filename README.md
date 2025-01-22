# <p align=center>`Weakly Supervised Change Detection with Layer-wise CAM Fusion and SACM-refined Pseudo Labels`</p>


## News
Repo is created in 2025-01-22. Code will come soon.

### 1. Overview

<p align="center">
    <img src="assest/Overview.png"/> <br />
</p>

`Weakly supervised change detection (WSCD) seeks to segment changed objects in remote sensing (RS) bi-temporal images using image-level annotations. Recent WSCD works commonly utilize class activation map (CAM) to generate pixel-level seeds for training change detection (CD) models. However, CAM-derived seeds often lack object-awareness, leading to inaccurate boundary delineation. In contrast, segment any change model (SACM), a SAM-based zero-shot CD model, can yield class-agnostic masks with clear contours for changed instances. This paper proposes a novel WSCD framework that CAM-derived seeds with SACM to improve pixel-level pseudo labels. The framework consists of layer-wise CAM fusion (LCF), local points prompting for SACM (LPPS), and SACM-refined pseudo labels (SPRL). LCF aggregates low-level spatial information and high-level semantics to activate comprehensive change maps. Meanwhile, LPPS converts LCF-derived CAM into relevant local points as the query for SACM, ensuring accurate mask generation. Further, SPRL refines pseudo labels by leveraging seeds from LCF and detailed boundary masks from LPPS, mitigating partial and false activation. Finally, refined pseudo labels are used to train the CD model. Our approach achieves state-of-the-art performance on both LEVIR-CD and WHU-CD datasets.` <br>

### 2. Usage
#### 2.1 Dataset
+ Prepare the data:
    Download the change detection datasets from the following links. Place them inside your `datasets` folder.

    - [`LEVIR-CD`](https://www.dropbox.com/s/18fb5jo0npu5evm/LEVIR-CD256.zip?dl=0)
    - [`WHU-CD`](https://www.dropbox.com/s/r76a00jcxp5d3hl/WHU-CD-256.zip?dl=0)
- Crop all datasets into 256x256 patches.
- Generate list file as `ls -R ./label/* > test.txt`
- Prepare datasets into following structure and set their path in `train.py` and `test.py`
  ```
  ├─A
      ├─A1.jpg/png
      ├─A2.jpg/png
      ├─...jpg/png
      └─...jpg/png
  ├─B
      ├─B1.jpg/png
      ├─B2.jpg/png
      ├─...jpg/png
      └─...jpg/png
  ├─label
      ├─label1.jpg/png
      ├─label2.jpg/png
      ├─...jpg/png
      └─...jpg/png
  ├─list
      ├─train.txt
      ├─val.txt
      └─test.txt
  ```
#### 2.2 Setting up conda environment
+ Prerequisites for Python:
    - Creating a virtual environment in terminal: `conda create -n RFANet python=3.8`
    - Installing necessary packages: `pip install -r requirements.txt `

#### 2.3 Installation
+ Clone this repo:
    ```shell
    git clone https://github.com/Youzhihui/RFANet.git
    cd RFANet
    ```
