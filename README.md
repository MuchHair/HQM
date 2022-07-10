# Towards Hard-Positive Query Mining for DETR-based Human-Object Interaction Detection
by [Xubin Zhong](https://scholar.google.com/citations?user=Y_ZvaccAAAAJ&hl=zh-CN&oi=sra), [Changxing Ding](https://scholar.google.com/citations?user=8Z8jplgAAAAJ&hl=zh-CN), Zijian Li and [Shaoli Huang](https://scholar.google.com/citations?user=o31BPFsAAAAJ&hl=zh-CN).

This repository contains the official implementation of the paper "[Towards Hard-Positive Query Mining for DETR-based Human-Object Interaction Detection](https://arxiv.org)", which is accepted to ECCV2022.

<div align="center">
  <img src=".github/overview.jpg" width="900px" />
</div>

To the best of our knowledge, HQM is the first approach that promotes the robustness of DETR-based models from the perspective of hard example mining. Moreover, HQM is plug-and-play and can be readily applied to many DETR-based HOI detection methods.

## Preparation

### Dependencies
Our implementation uses external libraries such as NumPy and PyTorch. You can resolve the dependencies with the following command.
```
pip install numpy
pip install -r requirements.txt
```
Note that this command may dump errors during installing pycocotools, but the errors can be ignored.

### Dataset

#### HICO-DET
HICO-DET dataset can be downloaded [here](https://drive.google.com/open?id=1QZcJmGVlF9f4h-XLWe9Gkmnmj2z1gSnk). After finishing downloading, unpack the tarball (`hico_20160224_det.tar.gz`) to the `data` directory.

Instead of using the original annotations files, we use the annotation files provided by the PPDM authors. The annotation files can be downloaded from [here](https://drive.google.com/open?id=1WI-gsNLS-t0Kh8TVki1wXqc3y2Ow1f2R). The downloaded annotation files have to be placed as follows.
```
HQM
 |─ data
 │   └─ hico_20160224_det
 |       |─ annotations
 |       |   |─ trainval_hico.json
 |       |   |─ test_hico.json
 |       |   └─ corre_hico.npy
 :       :
```


### Pre-trained parameters
The annotations file,
pre-trained weights and 
trained parameters can be downloaded [here]() (pwd:1111)

### Trained parameters
The trained parameters are available [here](https://pan.baidu.com/s/13HUv_dsQncZIvQLAEuLavg) (pwd:1111).

## Training
After the preparation, you can start the training with the following command.
Note that the number of object classes is 81 because one class is added for missing object.

If you have multiple GPUs on your machine, you can utilize them to speed up the training. The number of GPUs is specified with the `--nproc_per_node` option. The following command starts the training with 8 GPUs for the HICO-DET training.




## Evaluation
The evaluation is conducted at the end of each epoch during the training. The results are written in `logs/log.txt` like below:
```
"test_mAP": 0.313470564574163, "test_mAP rare": 0.26546478777620686, "test_mAP non-rare": 0.32780995244887723
```
`test_mAP`, `test_mAP rare`, and `test_mAP non-rare` are the results of the default full, rare, and non-rare setting, respectively.

You can also conduct the evaluation with trained parameters as follows.
```
python main.py \
        --pretrained checkpoint.pth \
        --hoi \
        --dataset_file hico \
        --hoi_path data/hico_20160224_det \
        --num_obj_classes 80 \
        --num_verb_classes 117 \
        --backbone resnet50 \
        --eval
```


## Results
HICO-DET.
|| Full (D) | Rare (D) | Non-rare (D) | Full(KO) | Rare (KO) | Non-rare (KO) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
|HOTR + HQM (ResNet50)| 25.69 | 24.70 | 25.98 |28.24| 27.35 | 28.51 |
|QPIC + HQM (ResNet50)| 31.34 | 26.54 | 32.78 | 34.09 | 29.63 | 35.42 |
|CDN-S + HQM (ResNet101)| 32.47 |28.15 | 33.76 | 35.17 |30.73 |36.50|

D: Default, KO: Known object

V-COCO.
|| Scenario 1 | 
| :--- | :---: | :---: |
|QPIC + HQM (ResNet50)| 63.6 |

## Citation
Please consider citing our papers if it helps your research.
```
@inproceedings{zhong_eccv2022,
author = {Zhong, Xubin and Ding, Changxing and Li, Zijian and Huang, Shaoli},
title = {Towards Hard-Positive Query Mining for DETR-based Human-Object Interaction Detection},
booktitle={ECCV},
year = {2022},
}

@InProceedings{Qu_2022_CVPR,
    author    = {Qu, Xian and Ding, Changxing and Li, Xingao and Zhong, Xubin and Tao, Dacheng},
    title     = {Distillation Using Oracle Queries for Transformer-Based Human-Object Interaction Detection},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {19558-19567}
}
```
## Acknowledgement
[DOQ](https://github.com/SherlockHolmes221/DOQ) [QPIC](https://github.com/hitachi-rd-cv/qpic) 