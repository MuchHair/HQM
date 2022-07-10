import numpy as np
import util.io as io


class HOIAEvaluator():
    def __init__(self, preds, gts, correct_mat,write_file_name=None):
        self.overlap_iou = 0.5
        self.max_hois = 100
        # self.verb_name_dict = {1: 'smoke', 2: 'call', 3: 'play(cellphone)', 4: 'eat', 5: 'drink',
        #                        6: 'ride', 7: 'hold', 8: 'kick', 9: 'read', 10: 'play (computer)'}
        self.verb_name_dict = {0: 'smoke', 1: 'call', 2: 'play(cellphone)', 3: 'eat', 4: 'drink',
                               5: 'ride', 6: 'hold', 7: 'kick', 8: 'read', 9: 'play (computer)'}
        self.fp = {}
        self.tp = {}
        self.score = {}
        self.sum_gt = {}
        for i in list(self.verb_name_dict.keys()):
            self.fp[i] = []
            self.tp[i] = []
            self.score[i] = []
            self.sum_gt[i] = 0
        self.file_name = []
        ####################################################
        self.gts = []
        for img_gts in gts:
            img_gts = {k: v.to('cpu').numpy() for k, v in img_gts.items() if k != 'id'}
            self.gts.append({
                'annotations': [{'bbox': bbox, 'category_id': label} for bbox, label in
                                zip(img_gts['boxes'], img_gts['labels'])],
                'hoi_annotation': [{'subject_id': hoi[0], 'object_id': hoi[1], 'category_id': hoi[2]} for hoi in
                                   img_gts['hois']]
            })

        for gt_i in self.gts:
            gt_hoi = gt_i['hoi_annotation']
            for gt_hoi_i in gt_hoi:
                if isinstance(gt_hoi_i['category_id'], str):
                    gt_hoi_i['category_id'] = int(gt_hoi_i['category_id'].replace('\n', ''))
                assert gt_hoi_i['category_id'] in list(self.verb_name_dict.keys())
                self.sum_gt[gt_hoi_i['category_id']] += 1
        self.num_class = len(list(self.verb_name_dict.keys()))

        ####################################################
        self.preds = []
        for img_preds in preds:
            # img_preds = {k: v.to('cpu').numpy() for k, v in img_preds.items()}
            bboxes = [{'bbox': list(bbox), 'category_id': label} for bbox, label in
                      zip(img_preds['boxes'], img_preds['labels'])]
            hoi_scores = img_preds['verb_scores']
            verb_labels = np.tile(np.arange(hoi_scores.shape[1]), (hoi_scores.shape[0], 1))
            subject_ids = np.tile(img_preds['sub_ids'], (hoi_scores.shape[1], 1)).T
            object_ids = np.tile(img_preds['obj_ids'], (hoi_scores.shape[1], 1)).T

            hoi_scores = hoi_scores.ravel()
            verb_labels = verb_labels.ravel()
            subject_ids = subject_ids.ravel()
            object_ids = object_ids.ravel()

            if len(subject_ids) > 0:
                object_labels = np.array([bboxes[object_id]['category_id'] for object_id in object_ids])
                masks = correct_mat[verb_labels, object_labels]
                hoi_scores *= masks

                hois = [{'subject_id': subject_id, 'object_id': object_id, 'category_id': category_id, 'score': score}
                        for
                        subject_id, object_id, category_id, score in
                        zip(subject_ids, object_ids, verb_labels, hoi_scores)]
                hois.sort(key=lambda k: (k.get('score', 0)), reverse=True)
                hois = hois[:self.max_hois]
            else:
                hois = []

            self.preds.append({
                'predictions': bboxes,
                'hoi_prediction': hois
            })

        if write_file_name:
            io.dump_json_object(self.preds, write_file_name)
            print(len(self.preds), "finish write json file")

    def evaluate(self):
        for pred_i, gt_i in zip(self.preds, self.gts):
            gt_bbox = gt_i['annotations']
            pred_bbox = pred_i['predictions']
            pred_hoi = pred_i['hoi_prediction']
            gt_hoi = gt_i['hoi_annotation']
            bbox_pairs = self.compute_iou_mat(gt_bbox, pred_bbox)
            self.compute_fptp(pred_hoi, gt_hoi, bbox_pairs)
        map = self.compute_map()
        print(map)
        return map

    def compute_map(self):
        ap = np.zeros(self.num_class)
        max_recall = np.zeros(self.num_class)
        for i in list(self.verb_name_dict.keys()):
            sum_gt = self.sum_gt[i]

            if sum_gt == 0:
                continue
            tp = np.asarray((self.tp[i]).copy())
            fp = np.asarray((self.fp[i]).copy())
            res_num = len(tp)
            if res_num == 0:
                continue
            score = np.asarray(self.score[i].copy())
            sort_inds = np.argsort(-score)
            fp = fp[sort_inds]
            tp = tp[sort_inds]
            fp = np.cumsum(fp)
            tp = np.cumsum(tp)
            rec = tp / sum_gt
            prec = tp / (fp + tp)
            ap[i] = self.voc_ap(rec, prec)
            max_recall[i] = np.max(rec)
            # print('class {} --- ap: {}   max recall: {}'.format(
            #     i, ap[i-1], max_recall[i-1]))
        mAP = np.mean(ap[:])
        m_rec = np.mean(max_recall[:])
        print('--------------------')
        print('mAP: {}   max recall: {}'.format(mAP, m_rec))
        print('--------------------')
        cate_map = {}
        for i in range(len(ap)):
            cate_map[self.verb_name_dict[i]] = ap[i]
        cate_map.update({'mAP': mAP,
                               'mean max recall': m_rec})
        return cate_map

    def voc_ap(self, rec, prec):
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap

    def compute_fptp(self, pred_hoi, gt_hoi, match_pairs):
        pos_pred_ids = match_pairs.keys()
        vis_tag = np.zeros(len(gt_hoi))
        pred_hoi.sort(key=lambda k: (k.get('score', 0)), reverse=True)
        if len(pred_hoi) != 0:
            for i, pred_hoi_i in enumerate(pred_hoi):
                is_match = 0
                if isinstance(pred_hoi_i['category_id'], str):
                    pred_hoi_i['category_id'] = int(pred_hoi_i['category_id'].replace('\n', ''))
                if len(match_pairs) != 0 and pred_hoi_i['subject_id'] in pos_pred_ids and pred_hoi_i[
                    'object_id'] in pos_pred_ids:
                    pred_sub_ids = match_pairs[pred_hoi_i['subject_id']]
                    pred_obj_ids = match_pairs[pred_hoi_i['object_id']]
                    pred_category_id = pred_hoi_i['category_id']
                    for gt_id in np.nonzero(1 - vis_tag)[0]:
                        gt_hoi_i = gt_hoi[gt_id]
                        if (gt_hoi_i['subject_id'] in pred_sub_ids) and (gt_hoi_i['object_id'] in pred_obj_ids) and (
                                pred_category_id == gt_hoi_i['category_id']):
                            is_match = 1
                            vis_tag[gt_id] = 1
                            continue
                if pred_hoi_i['category_id'] not in list(self.fp.keys()):
                    continue
                if is_match == 1:
                    self.fp[pred_hoi_i['category_id']].append(0)
                    self.tp[pred_hoi_i['category_id']].append(1)

                else:
                    self.fp[pred_hoi_i['category_id']].append(1)
                    self.tp[pred_hoi_i['category_id']].append(0)
                self.score[pred_hoi_i['category_id']].append(pred_hoi_i['score'])

    def compute_iou_mat(self, bbox_list1, bbox_list2):
        iou_mat = np.zeros((len(bbox_list1), len(bbox_list2)))
        if len(bbox_list1) == 0 or len(bbox_list2) == 0:
            return {}
        for i, bbox1 in enumerate(bbox_list1):
            for j, bbox2 in enumerate(bbox_list2):
                iou_i = self.compute_IOU(bbox1, bbox2)
                iou_mat[i, j] = iou_i
        iou_mat[iou_mat >= self.overlap_iou] = 1
        iou_mat[iou_mat < self.overlap_iou] = 0

        match_pairs = np.nonzero(iou_mat)
        match_pairs_dict = {}
        if iou_mat.max() > 0:
            for i, pred_id in enumerate(match_pairs[1]):
                if pred_id not in match_pairs_dict.keys():
                    match_pairs_dict[pred_id] = []
                match_pairs_dict[pred_id].append(match_pairs[0][i])
        return match_pairs_dict

    def compute_IOU(self, bbox1, bbox2):
        if isinstance(bbox1['category_id'], str):
            bbox1['category_id'] = int(bbox1['category_id'].replace('\n', ''))
        if isinstance(bbox2['category_id'], str):
            bbox2['category_id'] = int(bbox2['category_id'].replace('\n', ''))
        if bbox1['category_id'] == bbox2['category_id']:
            rec1 = bbox1['bbox']
            rec2 = bbox2['bbox']
            # computing area of each rectangles
            S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
            S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])

            # computing the sum_area
            sum_area = S_rec1 + S_rec2

            # find the each edge of intersect rectangle
            left_line = max(rec1[1], rec2[1])
            right_line = min(rec1[3], rec2[3])
            top_line = max(rec1[0], rec2[0])
            bottom_line = min(rec1[2], rec2[2])
            # judge if there is an intersect
            if left_line >= right_line or top_line >= bottom_line:
                return 0
            else:
                intersect = (right_line - left_line) * (bottom_line - top_line)
                return intersect / (sum_area - intersect)
        else:
            return 0
