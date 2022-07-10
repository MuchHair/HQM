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

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_anno = self.annotations[self.ids[idx]]
        img = Image.open(self.img_folder / img_anno['file_name']).convert('RGB')
        w, h = img.size

        target = {}
        target['orig_size'] = torch.as_tensor([int(h), int(w)])
        target['size'] = torch.as_tensor([int(h), int(w)])
        target['id'] = idx

        if self._transforms is not None:
            img, _ = self._transforms(img, None)

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
        'train': (root / 'images' , root / 'annotations' / '2019_all.json'),
        'val': (root / 'images' / 'Test_2021', root / 'annotations' / 'test_2021.json')
    }
    CORRECT_MAT_PATH = root / 'annotations' / 'corre_hoia.npy'

    img_folder, anno_file = PATHS[image_set]
    dataset = HOIADetection(image_set, img_folder, anno_file, transforms=make_hico_transforms(image_set),
                            num_queries=args.num_queries, hard_negative=args.hard_negative)
    if image_set == 'val':
        dataset.set_rare_hois(PATHS['train'][1])
        dataset.load_correct_mat(CORRECT_MAT_PATH)
    return dataset
