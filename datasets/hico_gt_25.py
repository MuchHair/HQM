# ------------------------------------------------------------------------
# Copyright (c) Hitachi, Ltd. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""
HICO detection dataset.
"""
from pathlib import Path
from PIL import Image
import json
from collections import defaultdict
import numpy as np

import torch
import torch.utils.data

import datasets.transforms as T


class HICODetection(torch.utils.data.Dataset):

    def __init__(self, img_set, img_folder, anno_file, transforms, num_queries, hard_negative=False):
        self.img_set = img_set
        self.img_folder = img_folder
        with open(anno_file, 'r') as f:
            self.annotations = json.load(f)
        self._transforms = transforms

        self.num_queries = num_queries

        self._valid_obj_ids = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13,
                               14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                               24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
                               37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
                               48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
                               58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
                               72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
                               82, 84, 85, 86, 87, 88, 89, 90)
        self._valid_verb_ids = list(range(1, 118))

        if img_set == 'train':
            self._valid_hoi_ids = list(range(1, 601))
            self.ids = []
            for idx, img_anno in enumerate(self.annotations):
                for hoi in img_anno['hoi_annotation']:
                    if hoi['subject_id'] >= len(img_anno['annotations']) or hoi['object_id'] >= len(
                            img_anno['annotations']):
                        break
                else:
                    self.ids.append(idx)
        else:
            self._valid_hoi_ids = list(range(0, 600))
            self.ids = list(range(len(self.annotations)))

        self.hoi_word = np.load('data/hico_20160224_det/annotations/hoi_600.npy')
        self.hard_negative = hard_negative
        self.corre_hico_hg = np.load('data/hico_20160224_det/annotations/corre_hico_hg.npy')
        self.map = json.load(open("data/hico_20160224_det/annotations/verb_index_obj_index_hoi_index.json", "r"))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_anno = self.annotations[self.ids[idx]]

        img = Image.open(self.img_folder / img_anno['file_name']).convert('RGB')
        w, h = img.size

        if self.img_set == 'train' and len(img_anno['annotations']) > self.num_queries:
            img_anno['annotations'] = img_anno['annotations'][:self.num_queries]

        boxes = [obj['bbox'] for obj in img_anno['annotations']]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)

        # Add index for confirming which boxes are kept after image transformation
        classes = [(i, self._valid_obj_ids.index(obj['category_id'])) for i, obj in
                   enumerate(img_anno['annotations'])]
        classes = torch.tensor(classes, dtype=torch.int64)

        target = {}
        target['orig_size'] = torch.as_tensor([int(h), int(w)])
        target['size'] = torch.as_tensor([int(h), int(w)])
        ####################################################################################
        if self.img_set == 'train':
            boxes[:, 0::2].clamp_(min=0, max=w)
            boxes[:, 1::2].clamp_(min=0, max=h)
            keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
            boxes = boxes[keep]
            classes = classes[keep]

            target['boxes'] = boxes
            target['labels'] = classes
            target['iscrowd'] = torch.tensor([0 for _ in range(boxes.shape[0])])
            target['area'] = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

            if self._transforms is not None:
                img, target = self._transforms(img, target)

            kept_box_indices = [label[0] for label in target['labels']]

            target['labels'] = target['labels'][:, 1]

            obj_labels, verb_labels, sub_boxes, obj_boxes = [], [], [], []
            obj_masks = []
            sub_obj_pairs = []

            gt_items = np.empty([0, 300 + 12])
            gt_inter_point = []

            action_image_labels = torch.zeros([1, 117], dtype=target["labels"].dtype)
            hoi_image_labels = [[0 for _ in range(len(self._valid_hoi_ids))]]
            obj_image_labels = torch.zeros([1, 80], dtype=target["labels"].dtype)

            for i in range(min(len(img_anno['hoi_annotation']), 25)):
                hoi = img_anno['hoi_annotation'][i]
                if hoi['subject_id'] not in kept_box_indices or hoi['object_id'] not in kept_box_indices:
                    continue
                sub_obj_pair = (hoi['subject_id'], hoi['object_id'])
                if sub_obj_pair in sub_obj_pairs:
                    verb_labels[sub_obj_pairs.index(sub_obj_pair)][self._valid_verb_ids.index(hoi['category_id'])] = 1
                    hoi_image_labels[0][self._valid_hoi_ids.index(hoi['hoi_category_id'])] = 1
                    action_image_labels[0][self._valid_verb_ids.index(hoi['category_id'])] = 1
                else:
                    sub_obj_pairs.append(sub_obj_pair)
                    obj_labels.append(target['labels'][kept_box_indices.index(hoi['object_id'])])
                    verb_label = [0 for _ in range(len(self._valid_verb_ids))]
                    obj_mask = [0 for _ in range(len(self._valid_verb_ids))]
                    obj_label = target['labels'][kept_box_indices.index(hoi['object_id'])]
                    assert 0 <= obj_label < 80
                    coor_index = np.where(self.corre_hico_hg[obj_label])[0]
                    for coor_item in coor_index:
                        obj_mask[coor_item] = 1
                    obj_masks.append(obj_mask)
                    ##############################################################################
                    if self.hard_negative:
                        obj_label = target['labels'][kept_box_indices.index(hoi['object_id'])]
                        assert 0 <= obj_label < 80
                        coor_index = np.where(self.corre_hico_hg[obj_label])[0]
                        for coor_item in coor_index:
                            verb_label[coor_item] = -1
                    ##############################################################################
                    verb_label[self._valid_verb_ids.index(hoi['category_id'])] = 1
                    hoi_image_labels[0][self._valid_hoi_ids.index(hoi['hoi_category_id'])] = 1
                    action_image_labels[0][self._valid_verb_ids.index(hoi['category_id'])] = 1
                    obj_image_labels[0][target['labels'][kept_box_indices.index(hoi['object_id'])]] = 1

                    sub_box = target['boxes'][kept_box_indices.index(hoi['subject_id'])]
                    obj_box = target['boxes'][kept_box_indices.index(hoi['object_id'])]
                    verb_labels.append(verb_label)

                    sub_boxes.append(sub_box)
                    obj_boxes.append(obj_box)

                    c_dis = sub_box[0:2] - obj_box[0:2]
                    wh_size = torch.stack([sub_box[2] * sub_box[3], obj_box[2] * obj_box[3]])
                    gt_items_ = np.concatenate(
                        [self.hoi_word[self._valid_hoi_ids.index(hoi['hoi_category_id'])][300:],
                         sub_box, obj_box, c_dis, wh_size])
                    gt_items = np.concatenate([gt_items, gt_items_.reshape(1, 312)], axis=0)

                    h_, w_ = target['size']
                    subx, suby = sub_box[0] * w_, sub_box[1] * h_
                    objx, objy = obj_box[0] * w_, obj_box[1] * h_
                    gt_inter_point.append([(subx + objx) / 2 / 32, (suby + objy) / 2 / 32])

            if len(sub_obj_pairs) == 0:
                target['obj_labels'] = torch.zeros((0,), dtype=torch.int64)
                target['verb_labels'] = torch.zeros((0, len(self._valid_verb_ids)), dtype=torch.float32)
                target['obj_masks'] = torch.zeros((0, len(self._valid_verb_ids)), dtype=torch.float32)
                target['hoi_image_labels'] = torch.zeros((1, len(self._valid_hoi_ids)), dtype=torch.float32)
                target['action_image_labels'] = torch.zeros((1, len(self._valid_verb_ids)), dtype=torch.float32)
                target['obj_image_labels'] = torch.zeros((1, 80), dtype=torch.float32)
                target['sub_boxes'] = torch.zeros((0, 4), dtype=torch.float32)
                target['obj_boxes'] = torch.zeros((0, 4), dtype=torch.float32)
                target['gt_items'] = torch.zeros((0, 312 + 117), dtype=torch.float32)
                target['gt_inter_point'] = torch.zeros((0, 2), dtype=torch.float32)
            else:
                target['obj_labels'] = torch.stack(obj_labels)
                target['verb_labels'] = torch.as_tensor(verb_labels, dtype=torch.float32)
                target['obj_masks'] = torch.as_tensor(obj_masks, dtype=torch.float32)
                target['hoi_image_labels'] = torch.as_tensor(hoi_image_labels, dtype=torch.float32)
                target['obj_image_labels'] = torch.as_tensor(obj_image_labels, dtype=torch.float32)
                target['action_image_labels'] = torch.as_tensor(action_image_labels, dtype=torch.float32)
                target['sub_boxes'] = torch.stack(sub_boxes)
                target['obj_boxes'] = torch.stack(obj_boxes)
                gt_items = np.concatenate([gt_items, np.asarray(verb_labels)], axis=1)
                target['gt_items'] = torch.as_tensor(gt_items, dtype=torch.float32)
                target['gt_inter_point'] = torch.as_tensor(gt_inter_point, dtype=torch.float32)
        else:
            target['boxes'] = boxes
            target['labels'] = classes
            target['id'] = idx

            if self._transforms is not None:
                img, target = self._transforms(img, target)

            kept_box_indices = [label[0] for label in target['labels']]
            target['labels'] = target['labels'][:, 1]

            sub_obj_pairs = []
            gt_items = np.empty([0, 300 + 12])
            obj_labels, verb_labels, sub_boxes, obj_boxes = [], [], [], []

            for hoi in img_anno['hoi_annotation']:
                sub_obj_pair = (hoi['subject_id'], hoi['object_id'])
                if sub_obj_pair in sub_obj_pairs:
                    verb_labels[sub_obj_pairs.index(sub_obj_pair)][self._valid_verb_ids.index(hoi['category_id'])] = 1
                    continue
                sub_obj_pairs.append(sub_obj_pair)

                obj_index = target['labels'][kept_box_indices.index(hoi['object_id'])].cpu().numpy()
                hoi['hoi_category_id'] = self.map[str(self._valid_verb_ids.index(hoi['category_id']))][
                    str(obj_index)]

                sub_box = target['boxes'][kept_box_indices.index(hoi['subject_id'])]
                obj_box = target['boxes'][kept_box_indices.index(hoi['object_id'])]
                c_dis = sub_box[0:2] - obj_box[0:2]
                wh_size = torch.stack([sub_box[2] * sub_box[3], obj_box[2] * obj_box[3]])
                gt_items_ = np.concatenate(
                    [self.hoi_word[self._valid_hoi_ids.index(hoi['hoi_category_id'])][300:],
                     target['boxes'][kept_box_indices.index(hoi['subject_id'])],
                     target['boxes'][kept_box_indices.index(hoi['object_id'])],
                     c_dis, wh_size])
                gt_items = np.concatenate([gt_items, gt_items_.reshape(1, 312)], axis=0)

                verb_label = [0 for _ in range(len(self._valid_verb_ids))]
                verb_label[self._valid_verb_ids.index(hoi['category_id'])] = 1
                verb_labels.append(verb_label)

                obj_labels.append(target['labels'][kept_box_indices.index(hoi['object_id'])])

                sub_box = target['boxes'][kept_box_indices.index(hoi['subject_id'])]
                obj_box = target['boxes'][kept_box_indices.index(hoi['object_id'])]
                sub_boxes.append(sub_box)
                obj_boxes.append(obj_box)

            if len(sub_obj_pairs) == 0:
                target['obj_labels'] = torch.zeros((0,), dtype=torch.int64)
                target['verb_labels'] = torch.zeros((0, len(self._valid_verb_ids)), dtype=torch.float32)
                target['gt_items'] = torch.zeros((0, 312 + 117), dtype=torch.float32)
                target['sub_boxes'] = torch.zeros((0, 4), dtype=torch.float32)
                target['obj_boxes'] = torch.zeros((0, 4), dtype=torch.float32)

            else:
                target['obj_labels'] = torch.stack(obj_labels)
                target['verb_labels'] = torch.as_tensor(verb_labels, dtype=torch.float32)
                gt_items = np.concatenate([gt_items, np.asarray(verb_labels)], axis=1)
                target['gt_items'] = torch.as_tensor(gt_items, dtype=torch.float32)
                target['sub_boxes'] = torch.stack(sub_boxes)
                target['obj_boxes'] = torch.stack(obj_boxes)

            target['boxes'] = boxes
            target['labels'] = classes[:, 1]
            target['id'] = idx

            hois = []
            for hoi in img_anno['hoi_annotation']:
                hois.append((hoi['subject_id'], hoi['object_id'], self._valid_verb_ids.index(hoi['category_id'])))
            target['hois'] = torch.as_tensor(hois, dtype=torch.int64)

        return img, target

    def set_rare_hois(self, anno_file):
        with open(anno_file, 'r') as f:
            annotations = json.load(f)

        counts = defaultdict(lambda: 0)
        for img_anno in annotations:
            hois = img_anno['hoi_annotation']
            bboxes = img_anno['annotations']
            for hoi in hois:
                triplet = (self._valid_obj_ids.index(bboxes[hoi['subject_id']]['category_id']),
                           self._valid_obj_ids.index(bboxes[hoi['object_id']]['category_id']),
                           self._valid_verb_ids.index(hoi['category_id']))
                counts[triplet] += 1
        self.rare_triplets = []
        self.non_rare_triplets = []
        for triplet, count in counts.items():
            if count < 10:
                self.rare_triplets.append(triplet)
            else:
                self.non_rare_triplets.append(triplet)

    def load_correct_mat(self, path):
        self.correct_mat = np.load(path)


# Add color jitter to coco transforms
def make_hico_transforms(image_set):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.ColorJitter(.4, .4, .4),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.hoi_path)
    assert root.exists(), f'provided HOI path {root} does not exist'
    PATHS = {
        'train': (root / 'images' / 'train2015', root / 'annotations' / 'trainval_hico.json'),
        'val': (root / 'images' / 'test2015', root / 'annotations' / 'test_hico.json')
    }
    CORRECT_MAT_PATH = root / 'annotations' / 'corre_hico.npy'

    img_folder, anno_file = PATHS[image_set]
    dataset = HICODetection(image_set, img_folder, anno_file, transforms=make_hico_transforms(image_set),
                            num_queries=args.num_queries, hard_negative=args.hard_negative)
    if image_set == 'val':
        dataset.set_rare_hois(PATHS['train'][1])
        dataset.load_correct_mat(CORRECT_MAT_PATH)
    return dataset
