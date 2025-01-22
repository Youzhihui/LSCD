# <p align=center>`Weakly Supervised Change Detection with Layer-wise CAM Fusion and SACM-refined Pseudo Labels`</p>


## News
Repo is created in 2025-01-22. Code will come soon.

### 1. Overview

<p align="center">
    <img src="assest/Overview.png"/> <br />
</p>

`  A lightweight change detection network, called as robust feature aggregation network (RFANet). To improve representative capability of weaker features extracted from lightweight backbone, a feature reinforcement module (FRM) is proposed. FRM allows current level feature to densely interact and fuse with other level features, thus accomplishing the complementarity of fine-grained details and semantic information. Considering massive objects with rich correlations in RS images, we design semantic split-aggregation module (SSAM) to better capture global semantic information of changed objects. Besides, we present a lightweight decoder containing channel interaction module (CIM), which allows multi-level refined difference features to emphasize changed areas and suppress background and pseudo-changes.` <br>

### 2. Usage
#### 2.1 Dataset
+ Prepare the data:
    Download the change detection datasets from the following links. Place them inside your `datasets` folder.

    - [`LEVIR-CD`](https://www.dropbox.com/s/18fb5jo0npu5evm/LEVIR-CD256.zip?dl=0)
    - [`WHU-CD`](https://www.dropbox.com/s/r76a00jcxp5d3hl/WHU-CD-256.zip?dl=0)
    - [`CDD-CD`](https://www.dropbox.com/s/ls9fq5u61k8wxwk/CDD.zip?dl=0)
    - [`SYSU-CD`](https://github.com/liumency/SYSU-CD)
    - [`./samples/test`](https://github.com/Youzhihui/RFANet/tree/main/samples) is a sample to start quickly.
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
