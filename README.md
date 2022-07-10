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
Our QPIC + HQM have to be pre-trained with the COCO object detection dataset. For the HICO-DET training, this pre-training can be omitted by using the parameters of DETR. The parameters can be downloaded from [here](https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth) for the ResNet50 backbone. 


### Trained parameters
The trained parameters are available [here](https://pan.baidu.com/s/13HUv_dsQncZIvQLAEuLavg) (pwd:1111).

## Training
After the preparation, you can start the training with the following command.
Note that the number of object classes is 81 because one class is added for missing object.

If you have multiple GPUs on your machine, you can utilize them to speed up the training. The number of GPUs is specified with the `--nproc_per_node` option. The following command starts the training with 8 GPUs for the HICO-DET training.


For the GBS training on HICO-DET.
```
python -m torch.distributed.launch \
        --nproc_per_node=8 \
        --use_env \
        main.py \
        --pretrained params/detr-r50-pre-hico.pth \
        --output_dir logs \
        --hoi \
        --dataset_file hico \
        --hoi_path data/hico_20160224_det \
        --num_obj_classes 80 \
        --num_verb_classes 117 \
        --backbone resnet50 \
        --set_cost_bbox 2.5 \
        --set_cost_giou 1 \
        --bbox_loss_coef 2.5 \
        --giou_loss_coef 1
```

For the AMM training on HICO-DET.
```
python -m torch.distributed.launch \
        --nproc_per_node=8 \
        --use_env \
        main.py \
        --pretrained params/detr-r50-pre-hico.pth \
        --output_dir logs \
        --hoi \
        --dataset_file hico \
        --hoi_path data/hico_20160224_det \
        --num_obj_classes 80 \
        --num_verb_classes 117 \
        --backbone resnet50 \
        --set_cost_bbox 2.5 \
        --set_cost_giou 1 \
        --bbox_loss_coef 2.5 \
        --giou_loss_coef 1
```

For HQM  training on HICO-DET.
```
python -m torch.distributed.launch \
        --nproc_per_node=8 \
        --use_env \
        main.py \
        --pretrained params/detr-r50-pre-hico.pth \
        --output_dir logs \
        --hoi \
        --dataset_file hico \
        --hoi_path data/hico_20160224_det \
        --num_obj_classes 80 \
        --num_verb_classes 117 \
        --backbone resnet50 \
        --set_cost_bbox 2.5 \
        --set_cost_giou 1 \
        --bbox_loss_coef 2.5 \
        --giou_loss_coef 1
```


## Evaluation
The evaluation is conducted at the end of each epoch during the training. The results are written in `logs/log.txt` like below:
```
"test_mAP": 0.29061250833779456, "test_mAP rare": 0.21910348492395765, "test_mAP non-rare": 0.31197234650036926
```
`test_mAP`, `test_mAP rare`, and `test_mAP non-rare` are the results of the default full, rare, and non-rare setting, respectively.

You can also conduct the evaluation with trained parameters as follows.
```
python main.py \
        --pretrained qpic_resnet50_hico.pth \
        --hoi \
        --dataset_file hico \
        --hoi_path data/hico_20160224_det \
        --num_obj_classes 80 \
        --num_verb_classes 117 \
        --backbone resnet50 \
        --eval
```

For the official evaluation of V-COCO, a pickle file of detection results have to be generated. You can generate the file as follows.
```
python generate_vcoco_official.py \
        --param_path logs/checkpoint.pth \
        --save_path vcoco.pickle \
        --hoi_path data/v-coco
```

## Results
HICO-DET.
|| Full (D) | Rare (D) | Non-rare (D) | Full(KO) | Rare (KO) | Non-rare (KO) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
|QPIC (ResNet50)| 29.07 | 21.85 | 31.23 | 31.68 | 24.14 | 33.93 |
|QPIC (ResNet101)| 29.90 | 23.92 | 31.69 | 32.38 | 26.06 | 34.27 |

D: Default, KO: Known object

V-COCO.
|| Scenario 1 | Scenario 2 |
| :--- | :---: | :---: |
|QPIC (ResNet50)| 58.8 | 61.0
|QPIC (ResNet101)| 58.3 | 60.7

## Citation
Please consider citing our paper if it helps your research.
```
@inproceedings{tamura_cvpr2021,
author = {Tamura, Masato and Ohashi, Hiroki and Yoshinaga, Tomoaki},
title = {{QPIC}: Query-Based Pairwise Human-Object Interaction Detection with Image-Wide Contextual Information},
booktitle={CVPR},
year = {2021},
}
```
