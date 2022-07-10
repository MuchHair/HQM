# ------------------------------------------------------------------------
# Copyright (c) Hitachi, Ltd. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""
HICO detection dataset.
"""
import pickle
from pathlib import Path
from PIL import Image
import json
from collections import defaultdict
import numpy as np

import torch
import torch.utils.data

import datasets.transforms as T
from util.box_ops import box_cxcywh_to_xyxy
from datasets.stitch_images import get_replace_image, get_sim_index


class HICODetection(torch.utils.data.Dataset):

    def __init__(self, img_set, img_folder, anno_file, transforms, num_queries, eval_gt=False):
        self.img_set = img_set
        self.img_folder = img_folder
        with open(anno_file, 'r') as f:
            self.annotations = json.load(f)
        self._transforms = transforms
        self.eval_gt = eval_gt
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

        # self.word2vec_verb = np.load('data/hico_20160224_det/annotations/verb_117.npy')
        self.word2vec_obj = np.load("data/hico_20160224_det/annotations/coco_clipvec.npy")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # if np.random.random() < 0.25 and self.img_set == 'train' and self.ids[idx] not in self.nohoi_index:
        #     sim_in_num = 3
        #     random_index = [self.ids[idx]] + get_sim_index(sim_in_num, self.nohoi_index, self.sim_index[self.ids[idx]])
        #     img_anno, img = get_replace_image(random_index, self.annotations, self.img_folder, "hoia")
        # else:
        #     img_anno = self.annotations[self.ids[idx]]
        #     img = Image.open(self.img_folder / img_anno['file_name']).convert('RGB')
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
                h, w = img.shape[-2:]
                target['orig_size'] = torch.as_tensor([int(h), int(w)])
            kept_box_indices = [label[0] for label in target['labels']]

            target['labels'] = target['labels'][:, 1]

            obj_labels, verb_labels, sub_boxes, obj_boxes = [], [], [], []
            sub_obj_pairs = []
            gt_items = np.empty([0, 512 + 12])
            # verb_word_wmb = np.empty([0, 300])
            sub_ids, obj_ids = [], []

            for hoi in img_anno['hoi_annotation']:
                if hoi['subject_id'] not in kept_box_indices or hoi['object_id'] not in kept_box_indices:
                    continue
                sub_obj_pair = (hoi['subject_id'], hoi['object_id'])
                if sub_obj_pair in sub_obj_pairs:
                    verb_index = self._valid_verb_ids.index(hoi['category_id'])
                    verb_labels[sub_obj_pairs.index(sub_obj_pair)][verb_index] = 1
                    # verb_word_wmb[sub_obj_pairs.index(sub_obj_pair)] += self.word2vec_verb[verb_index]
                else:
                    sub_obj_pairs.append(sub_obj_pair)
                    obj_labels.append(target['labels'][kept_box_indices.index(hoi['object_id'])])
                    verb_label = [0 for _ in range(len(self._valid_verb_ids))]
                    ##############################################################################
                    verb_label[self._valid_verb_ids.index(hoi['category_id'])] = 1
                    sub_box = target['boxes'][kept_box_indices.index(hoi['subject_id'])]
                    obj_box = target['boxes'][kept_box_indices.index(hoi['object_id'])]
                    verb_labels.append(verb_label)

                    sub_boxes.append(sub_box)
                    obj_boxes.append(obj_box)
                    c_dis = sub_box[0:2] - obj_box[0:2]
                    wh_size = torch.stack([sub_box[2] * sub_box[3], obj_box[2] * obj_box[3]])

                    obj_label = target['labels'][kept_box_indices.index(hoi['object_id'])]
                    assert 0 <= obj_label < 80
                    gt_items_ = np.concatenate([self.word2vec_obj[obj_label],
                                                sub_box, obj_box, c_dis, wh_size])
                    gt_items = np.concatenate([gt_items, gt_items_.reshape(1, 524)], axis=0)

                    # verb_index = self._valid_verb_ids.index(hoi['category_id'])
                    # assert 0 <= verb_index < 117
                    # verb_word_wmb = np.concatenate([verb_word_wmb,
                    #                                 self.word2vec_verb[verb_index]], axis=0)
                    sub_ids.append(kept_box_indices.index(hoi['subject_id']))
                    obj_ids.append(kept_box_indices.index(hoi['object_id']))

            if len(sub_obj_pairs) == 0:
                target['obj_labels'] = torch.zeros((0,), dtype=torch.int64)
                target['verb_labels'] = torch.zeros((0, len(self._valid_verb_ids)), dtype=torch.float32)
                target['sub_boxes'] = torch.zeros((0, 4), dtype=torch.float32)
                target['obj_boxes'] = torch.zeros((0, 4), dtype=torch.float32)
                target['gt_items'] = torch.zeros((0, 524), dtype=torch.float32)
                target['sub_ids'] = torch.zeros((0), dtype=torch.int64)
                target['obj_ids'] = torch.zeros((0), dtype=torch.int64)
            else:
                target['obj_labels'] = torch.stack(obj_labels)
                target['verb_labels'] = torch.as_tensor(verb_labels, dtype=torch.float32)
                target['sub_boxes'] = torch.stack(sub_boxes)
                target['obj_boxes'] = torch.stack(obj_boxes)
                # gt_items = np.concatenate([gt_items, np.asarray(verb_labels)], axis=1)
                # num = np.sum(np.asarray(verb_labels), axis=-1).reshape(-1, 1)
                # verb_word_wmb = verb_word_wmb / num
                # gt_items = np.concatenate([gt_items, verb_word_wmb], axis=1)
                target['gt_items'] = torch.as_tensor(gt_items, dtype=torch.float32)
                target['sub_ids'] = torch.as_tensor(sub_ids, dtype=torch.int64)
                target['obj_ids'] = torch.as_tensor(obj_ids, dtype=torch.int64)
        else:
            #############################################################
            if self.eval_gt:
                target['boxes'] = boxes
                target['labels'] = classes
                target['id'] = idx

                if self._transforms is not None:
                    img, target = self._transforms(img, target)

                kept_box_indices = [label[0] for label in target['labels']]
                target['labels'] = target['labels'][:, 1]

                sub_obj_pairs = []
                gt_items = np.empty([0, 512 + 12])
                sub_ids, obj_ids = [], []
                hois = []
                for hoi in img_anno['hoi_annotation']:
                    hois.append((hoi['subject_id'], hoi['object_id'], self._valid_verb_ids.index(hoi['category_id'])))
                    sub_obj_pair = (hoi['subject_id'], hoi['object_id'])
                    if sub_obj_pair in sub_obj_pairs:
                        continue
                    sub_obj_pairs.append(sub_obj_pair)

                    sub_box = target['boxes'][kept_box_indices.index(hoi['subject_id'])]
                    obj_box = target['boxes'][kept_box_indices.index(hoi['object_id'])]
                    c_dis = sub_box[0:2] - obj_box[0:2]
                    wh_size = torch.stack([sub_box[2] * sub_box[3], obj_box[2] * obj_box[3]])

                    obj_label = target['labels'][kept_box_indices.index(hoi['object_id'])]
                    assert 0 <= obj_label < 80
                    gt_items_ = np.concatenate([self.word2vec_obj[obj_label],
                                                sub_box, obj_box, c_dis, wh_size])
                    gt_items = np.concatenate([gt_items, gt_items_.reshape(1, 524)], axis=0)
                    sub_ids.append(kept_box_indices.index(hoi['subject_id']))
                    obj_ids.append(kept_box_indices.index(hoi['object_id']))

                if len(sub_obj_pairs) == 0:
                    target['gt_items'] = torch.zeros((0, 524), dtype=torch.float32)
                    target['sub_ids'] = torch.zeros((0), dtype=torch.int64)
                    target['obj_ids'] = torch.zeros((0), dtype=torch.int64)
                else:
                    target['gt_items'] = torch.as_tensor(gt_items, dtype=torch.float32)
                    target['sub_ids'] = torch.as_tensor(sub_ids, dtype=torch.int64)
                    target['obj_ids'] = torch.as_tensor(obj_ids, dtype=torch.int64)

                target['hois'] = torch.as_tensor(hois, dtype=torch.int64)
                target['boxes'] = boxes
                #############################################################
            else:
                target['boxes'] = boxes
                target['labels'] = classes[:, 1]
                target['id'] = idx

                if self._transforms is not None:
                    img, _ = self._transforms(img, None)

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

    def get_nohoid_index(self):
        nohoi_index = []
        for idx, img_anno in enumerate(self.annotations):
            if len(img_anno['hoi_annotation']) == 0 or len(img_anno['annotations']) > 100:
                nohoi_index.append(idx)
                continue
            for hoi in img_anno['hoi_annotation']:
                if hoi['subject_id'] >= len(img_anno['annotations']) or \
                        hoi['object_id'] >= len(img_anno['annotations']):
                    nohoi_index.append(idx)
                    break
        self.nohoi_index = nohoi_index

    def get_sim_index(self):
        self.sim_index = pickle.load(open('data/hico_20160224_det/annotations/sim_index_hico.pickle', 'rb'))


def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_gaussian(heatmap, center, radius, k=1, factor=6):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / factor)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)
    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


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


def generate_pose_configmap(sub_boxes1, obj_boxes1, HEATMAP_SIZE=64):
    num_triplets = len(sub_boxes1)
    ret = np.zeros((num_triplets, 2, HEATMAP_SIZE, HEATMAP_SIZE))
    sub_boxes = box_cxcywh_to_xyxy(torch.stack(sub_boxes1))
    obj_boxes = box_cxcywh_to_xyxy(torch.stack(obj_boxes1))

    for i in range(num_triplets):
        x1, y1, x2, y2 = sub_boxes[i].numpy()

        ox0 = np.clip(np.round(x1 * HEATMAP_SIZE).astype(np.int), 0, HEATMAP_SIZE - 1)
        oy0 = np.clip(np.round(y1 * HEATMAP_SIZE).astype(np.int), 0, HEATMAP_SIZE - 1)
        ox1 = np.clip(np.round(x2 * HEATMAP_SIZE).astype(np.int), 0, HEATMAP_SIZE - 1)
        oy1 = np.clip(np.round(y2 * HEATMAP_SIZE).astype(np.int), 0, HEATMAP_SIZE - 1)
        ret[i, 0, oy0:oy1, ox0:ox1] = 1.0

        x_c, y_c, w, h = obj_boxes[i].numpy()
        x1, y1, x2, y2 = x_c - 0.5 * w, y_c - 0.5 * h, x_c + 0.5 * w, y_c + 0.5 * h
        ox0 = np.clip(np.round(x1 * HEATMAP_SIZE).astype(np.int), 0, HEATMAP_SIZE - 1)
        oy0 = np.clip(np.round(y1 * HEATMAP_SIZE).astype(np.int), 0, HEATMAP_SIZE - 1)
        ox1 = np.clip(np.round(x2 * HEATMAP_SIZE).astype(np.int), 0, HEATMAP_SIZE - 1)
        oy1 = np.clip(np.round(y2 * HEATMAP_SIZE).astype(np.int), 0, HEATMAP_SIZE - 1)
        ret[i, 1, ox0:ox1, oy0:oy1] = 1.0

    return ret.astype(np.float32)


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
                            num_queries=args.num_queries, eval_gt=args.eval_gt)
    if image_set == 'train':
        dataset.get_nohoid_index()
        dataset.get_sim_index()
    elif image_set == 'val':
        dataset.set_rare_hois(PATHS['train'][1])
        dataset.load_correct_mat(CORRECT_MAT_PATH)
    return dataset
