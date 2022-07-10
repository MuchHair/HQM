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


class HOIADetection(torch.utils.data.Dataset):

    def __init__(self, img_set, img_folder, anno_file, transforms, num_queries, hard_negative=False):
        self.img_set = img_set
        self.img_folder = img_folder
        with open(anno_file, 'r') as f:
            self.annotations = json.load(f)
        self._transforms = transforms

        self.num_queries = num_queries

        self._valid_obj_ids = list(range(1, 12))
        self._valid_verb_ids = list(range(1, 11))

        self.verb_name_dict = {1: 'smoke', 2: 'call', 3: 'play(cellphone)', 4: 'eat', 5: 'drink',
                               6: 'ride', 7: 'hold', 8: 'kick', 9: 'read', 10: 'play (computer)'}
        self.obj_name_dict = {1: 'person', 2: 'cellphone', 3: 'cigarette', 4: 'drink', 5: 'food',
                              6: 'bike', 7: 'motorbike', 8: 'horse', 9: 'ball', 10: 'computer', 11: 'document'}

        if img_set == 'train':
            self.ids = []
            for idx, img_anno in enumerate(self.annotations):
                for hoi in img_anno['hoi_annotation']:
                    if hoi['subject_id'] >= len(img_anno['annotations']) or hoi['object_id'] >= len(
                            img_anno['annotations']) or int(hoi['category_id']) == 0:
                        break
                else:
                    self.ids.append(idx)
        else:
            self.ids = []
            for idx, img_anno in enumerate(self.annotations):
                if img_anno["file_name"] == "test_006960.jpg":
                    continue
                self.ids.append(idx)

        self.hoi_list = json.load(open("data/HOIA-2021/annotations/hoi_list.json", "r"))
        self._valid_hoi_ids = list(range(0, 18))
        self.verb_index_obj_index_hoi_index = {}
        for hoi in self.hoi_list:
            verb_index = hoi["verb_index"]
            object_index = hoi["object_index"]
            if verb_index not in self.verb_index_obj_index_hoi_index:
                self.verb_index_obj_index_hoi_index[verb_index] = {}
            self.verb_index_obj_index_hoi_index[verb_index][object_index] = int(hoi["id"]) - 1

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

        if self.img_set == 'train':
            # Add index for confirming which boxes are kept after image transformation
            classes = [(i, self._valid_obj_ids.index(int(obj['category_id']))) for i, obj in
                       enumerate(img_anno['annotations'])]
        else:
            classes = [self._valid_obj_ids.index(int(obj['category_id'])) for obj in img_anno['annotations']]
        classes = torch.tensor(classes, dtype=torch.int64)

        target = {}
        target['orig_size'] = torch.as_tensor([int(h), int(w)])
        target['size'] = torch.as_tensor([int(h), int(w)])
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
            sub_obj_pairs = []
            hoi_labels = []
            verb_labels_one = []

            action_image_labels = torch.zeros([1, 10], dtype=target["labels"].dtype)
            obj_image_labels = torch.zeros([1, 11], dtype=target["labels"].dtype)

            for hoi in img_anno['hoi_annotation']:
                if hoi['subject_id'] not in kept_box_indices or hoi['object_id'] not in kept_box_indices:
                    continue
                #################################################################################################
                obj_index = target['labels'][kept_box_indices.index(hoi['object_id'])].cpu().numpy()
                if int(obj_index) not in self.verb_index_obj_index_hoi_index[
                    self._valid_verb_ids.index(hoi['category_id'])]:
                    continue
                hoi_category_id = self.verb_index_obj_index_hoi_index[self._valid_verb_ids.index(hoi['category_id'])][int(obj_index)]
                assert 0 <= hoi_category_id < 18
                hoi_labels.append(torch.as_tensor(hoi_category_id))
                #################################################################################################
                sub_obj_pair = (hoi['subject_id'], hoi['object_id'])
                sub_obj_pairs.append(sub_obj_pair)
                obj_labels.append(target['labels'][kept_box_indices.index(hoi['object_id'])])
                verb_label = [0 for _ in range(len(self._valid_verb_ids))]
                obj_label = target['labels'][kept_box_indices.index(hoi['object_id'])]
                assert 0 <= obj_label < 11
                ##############################################################################
                verb_label[self._valid_verb_ids.index(hoi['category_id'])] = 1
                action_image_labels[0][self._valid_verb_ids.index(hoi['category_id'])] = 1
                obj_image_labels[0][target['labels'][kept_box_indices.index(hoi['object_id'])]] = 1

                sub_box = target['boxes'][kept_box_indices.index(hoi['subject_id'])]
                obj_box = target['boxes'][kept_box_indices.index(hoi['object_id'])]
                verb_labels.append(verb_label)
                verb_labels_one.append(torch.as_tensor(self._valid_verb_ids.index(hoi['category_id'])))

                sub_boxes.append(sub_box)
                obj_boxes.append(obj_box)

            if len(sub_obj_pairs) == 0:
                target['obj_labels'] = torch.zeros((0,), dtype=torch.int64)
                target['hoi_labels'] = torch.zeros((0,), dtype=torch.int64)
                target['verb_labels'] = torch.zeros((0, len(self._valid_verb_ids)), dtype=torch.float32)
                target['verb_labels_one'] = torch.zeros((0,), dtype=torch.int64)
                target['action_image_labels'] = torch.zeros((1, len(self._valid_verb_ids)), dtype=torch.float32)
                target['obj_image_labels'] = torch.zeros((1, 80), dtype=torch.float32)
                target['sub_boxes'] = torch.zeros((0, 4), dtype=torch.float32)
                target['obj_boxes'] = torch.zeros((0, 4), dtype=torch.float32)
            else:
                target['obj_labels'] = torch.stack(obj_labels)
                target['verb_labels_one'] = torch.stack(verb_labels_one)
                target['hoi_labels'] = torch.stack(hoi_labels)
                target['verb_labels'] = torch.as_tensor(verb_labels, dtype=torch.float32)
                target['obj_image_labels'] = torch.as_tensor(obj_image_labels, dtype=torch.float32)
                target['action_image_labels'] = torch.as_tensor(action_image_labels, dtype=torch.float32)
                target['sub_boxes'] = torch.stack(sub_boxes)
                target['obj_boxes'] = torch.stack(obj_boxes)
        else:
            target['boxes'] = boxes
            target['labels'] = classes
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
                if hoi['subject_id'] >= len(img_anno['annotations']) or hoi['object_id'] >= len(
                        img_anno['annotations']) or int(hoi['category_id']) == 0:
                    continue
                triplet = (self._valid_obj_ids.index(int(bboxes[hoi['subject_id']]['category_id'])),
                           self._valid_obj_ids.index(int(bboxes[hoi['object_id']]['category_id'])),
                           self._valid_verb_ids.index(int(hoi['category_id'])))
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
        'train': (root / 'images' / 'trainval2019', root / 'annotations' / 'train_2019.json'),
        'val': (root / 'images' / 'test2019', root / 'annotations' / 'test_2019.json')
    }
    CORRECT_MAT_PATH = root / 'annotations' / 'corre_hoia.npy'

    img_folder, anno_file = PATHS[image_set]
    dataset = HOIADetection(image_set, img_folder, anno_file, transforms=make_hico_transforms(image_set),
                            num_queries=args.num_queries, hard_negative=args.hard_negative)
    if image_set == 'val':
        dataset.set_rare_hois(PATHS['train'][1])
        dataset.load_correct_mat(CORRECT_MAT_PATH)
    return dataset
