import copy

import torch
from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, is_dist_avail_and_initialized)
import torch.nn.functional as F
from torchvision.ops.boxes import box_area

def box_iou(boxes1, boxes2):

    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]

    iou = inter / (area1 + area2 - inter)
    return iou


def encoder_gt_postion(targets, batch_size):
    enc_targets = copy.deepcopy(targets)
    gt_positions = []
    for i in range(batch_size):
        sub_boxes = enc_targets[i]['sub_boxes']  # 4
        obj_boxes = enc_targets[i]['obj_boxes']  # 4

        sub_diff = torch.zeros_like(sub_boxes)
        sub_diff[:, :2] = sub_boxes[:, 2:] / 2
        sub_diff[:, 2:] = sub_boxes[:, 2:]
        sub_boxes += torch.mul((torch.rand_like(sub_boxes) * 2 - 1.0), sub_diff) * 0.2
        sub_boxes = sub_boxes.clamp(min=0.0, max=1.0)

        obj_diff = torch.zeros_like(obj_boxes)
        obj_diff[:, :2] = obj_boxes[:, 2:] / 2
        obj_diff[:, 2:] = obj_boxes[:, 2:]
        obj_boxes += torch.mul((torch.rand_like(obj_boxes) * 2 - 1.0), obj_diff) * 0.2
        obj_boxes = obj_boxes.clamp(min=0.0, max=1.0)

        # note that the position and area is normalized by img's orig size(w, h)
        # because the boxes is normalized after data transformation
        sub_obj_position = sub_boxes[:, 0:2] - obj_boxes[:, 0:2]  # 2
        sub_area = (sub_boxes[:, 2] * sub_boxes[:, 3]).unsqueeze(1)  # 1
        obj_area = (obj_boxes[:, 2] * obj_boxes[:, 3]).unsqueeze(1)  # 1

        gt_position = torch.cat([sub_boxes, obj_boxes, sub_obj_position, sub_area, obj_area], dim=1)

        if len(gt_position) == 0:
            gt_position = torch.zeros((0, 12), dtype=torch.float32)
        else:
            gt_position = torch.as_tensor(gt_position, dtype=torch.float32)

        gt_positions.append(gt_position)
    return gt_positions


def encoder_gt_postion_iou(targets, batch_size):
    enc_targets = copy.deepcopy(targets)
    gt_positions = []
    for i in range(batch_size):
        sub_boxes = enc_targets[i]['sub_boxes']  # 4
        obj_boxes = enc_targets[i]['obj_boxes']  # 4

        thres_low = 0.6
        thres_high = 1
        shift_r = 0.7

        count = 0
        count_max = 40
        augmented1 = True

        while augmented1 and count < count_max:

            sub_diff = torch.zeros_like(sub_boxes)
            sub_diff[:, :2] = sub_boxes[:, 2:] / 2
            sub_diff[:, 2:] = sub_boxes[:, 2:]
            sub_boxes_o = sub_boxes + torch.mul((torch.rand_like(sub_boxes) * 2 - 1.0), sub_diff) * 0.2
            sub_boxes_o = sub_boxes_o.clamp(min=0.0, max=1.0)

            iou = box_iou(box_cxcywh_to_xyxy(sub_boxes_o), box_cxcywh_to_xyxy(sub_boxes))
            iou_rate = (iou > thres_low) & (iou < thres_high)

            # print(f"iou_rate: {iou_rate}, count: {count}")
            if (iou_rate.float().sum() / len(sub_boxes)) >= shift_r or count == count_max:
                augmented1 = False
                sub_boxes[iou_rate] = sub_boxes_o[iou_rate]
            count += 1

        count = 0
        count_max = 40
        augmented1 = True

        while augmented1 and count < count_max:
            obj_diff = torch.zeros_like(obj_boxes)
            obj_diff[:, :2] = obj_boxes[:, 2:] / 2
            obj_diff[:, 2:] = obj_boxes[:, 2:]
            obj_boxes_o = obj_boxes + torch.mul((torch.rand_like(obj_boxes) * 2 - 1.0), obj_diff) * 0.2
            obj_boxes_o = obj_boxes_o.clamp(min=0.0, max=1.0)

            iou = box_iou(box_cxcywh_to_xyxy(obj_boxes_o), box_cxcywh_to_xyxy(obj_boxes))
            iou_rate = (iou > thres_low) & (iou < thres_high)

            if (iou_rate.float().sum() / len(obj_boxes)) >= shift_r or count == count_max:
                augmented1 = False
                obj_boxes[iou_rate] = obj_boxes_o[iou_rate]
            count += 1

        # note that the position and area is normalized by img's orig size(w, h)
        # because the boxes is normalized after data transformation
        sub_obj_position = sub_boxes[:, 0:2] - obj_boxes[:, 0:2]  # 2
        sub_area = (sub_boxes[:, 2] * sub_boxes[:, 3]).unsqueeze(1)  # 1
        obj_area = (obj_boxes[:, 2] * obj_boxes[:, 3]).unsqueeze(1)  # 1

        gt_position = torch.cat([sub_boxes, obj_boxes, sub_obj_position, sub_area, obj_area], dim=1)

        if len(gt_position) == 0:
            gt_position = torch.zeros((0, 12), dtype=torch.float32)
        else:
            gt_position = torch.as_tensor(gt_position, dtype=torch.float32)

        gt_positions.append(gt_position)
    return gt_positions


def prepare_for_doq(doq_args, batch_size, bbox_enc, query_embedding_weight):

    targets = doq_args

    gt_positions = encoder_gt_postion(targets, batch_size)
    gt_positions_padding_size = int(max([len(gt_positions[v]) for v in range(batch_size)]))
    gt_positions_arrays = torch.zeros((gt_positions_padding_size, batch_size, 12), dtype=torch.float32)
    gt_positions_mask = torch.zeros((batch_size, gt_positions_padding_size), dtype=torch.bool)

    for i in range(batch_size):
        gt_position = gt_positions[i]
        if len(gt_position) > 0:
            gt_positions_arrays[0:len(gt_position), i, :] = gt_position
            gt_positions_mask[i, len(gt_position):] = True

    query_embedding_len = query_embedding_weight.shape[0]
    query_embedding_mask = torch.zeros(batch_size, query_embedding_len, dtype=torch.bool)
    gt_positions_mask = torch.cat([gt_positions_mask, query_embedding_mask], dim=1).to('cuda')

    gt_positions_arrays = gt_positions_arrays.to('cuda')
    input_gt_position_embedding = torch.tanh(bbox_enc(gt_positions_arrays)).permute(1, 0, 2)
    query_embedding_weight = query_embedding_weight.repeat(batch_size, 1, 1)
    input_query_with_gt_position_query = torch.cat([input_gt_position_embedding, query_embedding_weight], dim=1)

    query_size = input_query_with_gt_position_query.shape[1]
    attn_mask = (torch.ones(query_size, query_size, dtype=torch.bool)*float('-inf')).to('cuda')
    attn_mask[0:gt_positions_padding_size, 0:gt_positions_padding_size] = 0
    attn_mask[gt_positions_padding_size:, gt_positions_padding_size:] = 0

    mask_dict = {
        'padding_size': gt_positions_padding_size,
        'targets': targets
    }

    return input_query_with_gt_position_query, attn_mask, gt_positions_mask, mask_dict


def prepare_for_hqm(doq_args, batch_size, bbox_enc, query_embedding_weight, hard=True, gt_lr_cross=False):
    targets = doq_args

    if hard:
        gt_positions = encoder_gt_postion_iou(targets, batch_size)
    else:
        gt_positions = encoder_gt_postion(targets, batch_size)
    gt_positions_padding_size = int(max([len(gt_positions[v]) for v in range(batch_size)]))
    gt_positions_arrays = torch.zeros((gt_positions_padding_size, batch_size, 12), dtype=torch.float32)
    gt_positions_mask = torch.zeros((batch_size, gt_positions_padding_size), dtype=torch.bool)

    for i in range(batch_size):
        gt_position = gt_positions[i]
        if len(gt_position) > 0:
            gt_positions_arrays[0:len(gt_position), i, :] = gt_position
            gt_positions_mask[i, len(gt_position):] = True

    query_embedding_len = query_embedding_weight.shape[0]
    query_embedding_mask = torch.zeros(batch_size, query_embedding_len, dtype=torch.bool)
    gt_positions_mask = torch.cat([gt_positions_mask, query_embedding_mask], dim=1).to('cuda')

    gt_positions_arrays = gt_positions_arrays.to('cuda')
    input_gt_position_embedding = torch.tanh(bbox_enc(gt_positions_arrays)).permute(1, 0, 2)
    query_embedding_weight = query_embedding_weight.repeat(batch_size, 1, 1)
    input_query_with_gt_position_query = torch.cat([input_gt_position_embedding, query_embedding_weight], dim=1)

    query_size = input_query_with_gt_position_query.shape[1]
    attn_mask = (torch.ones(query_size, query_size, dtype=torch.bool)*float('-inf')).to('cuda')
    attn_mask[0:gt_positions_padding_size, 0:gt_positions_padding_size] = 0
    attn_mask[gt_positions_padding_size:, gt_positions_padding_size:] = 0
    if gt_lr_cross:

        attn_mask[0:gt_positions_padding_size, gt_positions_padding_size:] = 0
    mask_dict = {
        'padding_size': gt_positions_padding_size,
        'targets': targets
    }

    return input_query_with_gt_position_query, attn_mask, gt_positions_mask, mask_dict


def doq_post_process(outputs_obj_class, outputs_verb_class, outputs_sub_coord, outputs_obj_coord, mask_dict):
    """
    post process of doq after output from the transformer
    put the doq part in the mask_dict
    """
    if mask_dict and mask_dict['padding_size'] > -1:
        outputs_known_obj_class = outputs_obj_class[:, :, :mask_dict['padding_size'], :]
        outputs_known_verb_class = outputs_verb_class[:, :, :mask_dict['padding_size'], :]
        outputs_known_sub_coord = outputs_sub_coord[:, :, :mask_dict['padding_size'], :]
        outputs_known_obj_coord = outputs_obj_coord[:, :, :mask_dict['padding_size'], :]

        outputs_obj_class = outputs_obj_class[:, :, mask_dict['padding_size']:, :]
        outputs_verb_class = outputs_verb_class[:, :, mask_dict['padding_size']:, :]
        outputs_sub_coord = outputs_sub_coord[:, :, mask_dict['padding_size']:, :]
        outputs_obj_coord = outputs_obj_coord[:, :, mask_dict['padding_size']:, :]

        mask_dict['output_known_gt'] = {'gt_obj_logits': outputs_known_obj_class[-1], 'gt_verb_logits': outputs_known_verb_class[-1], 'gt_sub_boxes': outputs_known_sub_coord[-1], 'gt_obj_boxes': outputs_known_obj_coord[-1]}
        aux_output = [{'gt_obj_logits': a, 'gt_verb_logits': b, 'gt_sub_boxes': c, 'gt_obj_boxes': d} for a, b, c, d in zip(outputs_known_obj_class[:-1], outputs_known_verb_class[:-1], outputs_known_sub_coord[:-1], outputs_known_obj_coord[:-1])]

        mask_dict['output_known_gt'].update({'aux_output': aux_output})
    return outputs_obj_class, outputs_verb_class, outputs_sub_coord, outputs_obj_coord


def loss_gt_obj_labels(outputs, targets, gt_idx, num_interactions, log=True):
    src_logits = outputs['gt_obj_logits']
    target_classes_o = torch.cat([t['obj_labels'] for t in targets])
    src_logits = src_logits[gt_idx]
    if len(target_classes_o) == 0:
        loss_obj_ce = torch.as_tensor(0.).to('cuda')
    else:
        loss_obj_ce = F.cross_entropy(src_logits, target_classes_o)

    losses = {'loss_gt_obj_ce': loss_obj_ce}

    if log:
        if len(target_classes_o) != 0:
            losses['obj_gt_class_error'] = 100 - accuracy(src_logits, target_classes_o)[0]
        else:
            losses['obj_gt_class_error'] = torch.as_tensor(0.).to('cuda')

    return losses


def _neg_loss(pred, gt):
    ''' Modified focal loss. Exactly the same as CornerNet.
      Runs faster and costs a little bit more memory
    '''
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos

    return loss


def loss_gt_verb_labels(outputs, targets, gt_idx, num_interactions):
    verb_loss_type = 'focal'
    src_logits = outputs['gt_verb_logits']
    target_classes_o = torch.cat([t['verb_labels'] for t in targets])
    src_logits = src_logits[gt_idx]

    if verb_loss_type == 'bce':
        loss_verb_ce = F.binary_cross_entropy_with_logits(src_logits, target_classes_o)
    elif verb_loss_type == 'focal':
        src_logits = src_logits.sigmoid()
        loss_verb_ce = _neg_loss(src_logits, target_classes_o)

    losses = {'loss_gt_verb_ce': loss_verb_ce}
    return losses


def loss_gt_sub_obj_boxes(outputs, targets, gt_idx, num_interactions):
    src_sub_boxes = outputs['gt_sub_boxes'][gt_idx]
    src_obj_boxes = outputs['gt_obj_boxes'][gt_idx]
    target_sub_boxes = torch.cat([t['sub_boxes'] for t in targets], dim=0)
    target_obj_boxes = torch.cat([t['obj_boxes'] for t in targets], dim=0)

    exist_obj_boxes = (target_obj_boxes != 0).any(dim=1)

    losses = {}
    if src_sub_boxes.shape[0] == 0:
        losses['loss_gt_sub_bbox'] = src_sub_boxes.sum()
        losses['loss_gt_obj_bbox'] = src_obj_boxes.sum()
        losses['loss_gt_sub_giou'] = src_sub_boxes.sum()
        losses['loss_gt_obj_giou'] = src_obj_boxes.sum()
    else:
        loss_sub_bbox = F.l1_loss(src_sub_boxes, target_sub_boxes, reduction='none')
        loss_obj_bbox = F.l1_loss(src_obj_boxes, target_obj_boxes, reduction='none')
        losses['loss_gt_sub_bbox'] = loss_sub_bbox.sum() / num_interactions
        losses['loss_gt_obj_bbox'] = (loss_obj_bbox * exist_obj_boxes.unsqueeze(1)).sum() / (
                exist_obj_boxes.sum() + 1e-4)
        loss_sub_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_sub_boxes),
                                                           box_cxcywh_to_xyxy(target_sub_boxes)))
        loss_obj_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_obj_boxes),
                                                           box_cxcywh_to_xyxy(target_obj_boxes)))
        losses['loss_gt_sub_giou'] = loss_sub_giou.sum() / num_interactions
        losses['loss_gt_obj_giou'] = (loss_obj_giou * exist_obj_boxes).sum() / (exist_obj_boxes.sum() + 1e-4)
    return losses


def get_loss(loss, outputs, targets, gt_idx, num, **kwargs):
    loss_map = {
        'gt_obj_labels': loss_gt_obj_labels,
        'gt_verb_labels': loss_gt_verb_labels,
        'gt_sub_obj_boxes': loss_gt_sub_obj_boxes,
    }

    assert loss in loss_map, f'do you really want to compute {loss} loss?'
    return loss_map[loss](outputs, targets, gt_idx, num, **kwargs)


def compute_doq_loss(mask_dict, aux_num):
    losses = ['gt_obj_labels', 'gt_verb_labels', 'gt_sub_obj_boxes']
    losses_dict = {}
    if 'output_known_gt' in mask_dict:
        targets = mask_dict['targets']
        outputs = mask_dict['output_known_gt']

        gt_batch_idx = []
        gt_indices = []
        gt_idx = []
        for i, t in enumerate(targets):
            gt_batch_idx.extend([i] * len(t['sub_boxes']))
            gt_indices.extend(torch.arange(0, len(t['sub_boxes'])))

        gt_idx.append(gt_batch_idx)
        gt_idx.append(gt_indices)

        num_interactions = sum(len(t['obj_labels']) for t in targets)
        num_interactions = torch.as_tensor([num_interactions], dtype=torch.float32).to('cuda')
        num_interactions = torch.clamp(num_interactions / 1, min=1).item()

        for loss in losses:
            losses_dict.update(get_loss(loss, outputs, targets, gt_idx, num_interactions))

        if 'aux_output' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_output']):
                for loss in losses:
                    kwargs = {}
                    if loss == 'gt_obj_labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = get_loss(loss, aux_outputs, targets, gt_idx, num_interactions, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses_dict.update(l_dict)
    else:
        l_dict = dict()
        l_dict['loss_gt_obj_ce'] = torch.as_tensor(0.).to('cuda')
        l_dict['obj_gt_class_error'] = torch.as_tensor(0.).to('cuda')
        l_dict['loss_gt_verb_ce'] = torch.as_tensor(0.).to('cuda')
        l_dict['loss_gt_sub_bbox'] = torch.as_tensor(0.).to('cuda')
        l_dict['loss_gt_obj_bbox'] = torch.as_tensor(0.).to('cuda')
        l_dict['loss_gt_sub_giou'] = torch.as_tensor(0.).to('cuda')
        l_dict['loss_gt_obj_giou'] = torch.as_tensor(0.).to('cuda')

        if aux_num:
            l_dict_aux = {}
            for i in range(aux_num):
                for k, v in l_dict.items():
                    if k != 'obj_gt_class_error':
                        l_dict_aux.update({k + f'_{i}': v})

            l_dict.update(l_dict_aux)

        losses_dict.update(l_dict)

    return losses_dict
