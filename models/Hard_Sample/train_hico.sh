python -m torch.distributed.launch \
--nproc_per_node=8  \
--use_env \
main.py \
--pretrained params/detr-r50-pre_inside.pth \
--output_dir outputs/12_20_hoi_hardm_query_att_each \
--hoi \
--dataset_file hico_gt \
--model_name hoi_hardm_query_att_each \
--hoi_path data/hico_20160224_det/ \
--num_obj_classes 80 \
--num_verb_classes 117 \
--backbone resnet50 \
--set_cost_bbox 2.5 \
--set_cost_giou 1 \
--bbox_loss_coef 2.5 \
--giou_loss_coef 1 \
--ts_begin 0 \
--eval_gt \
--lr 1e-5 \
--lr_backbone 1e-6 \
--resume outputs/12_6_hoi_share_qpos_ezero_shiftbbox_04_06_noamb/checkpoint.pth

python -m torch.distributed.launch \
--nproc_per_node=8  \
--use_env \
main.py \
--pretrained outputs/12_17_hoi_share_qpos_ezero_shiftbbox_04_06_noamb/checkpoint0057.pth \
--output_dir outputs/12_17_hoi_share_qpos_ezero_shiftbbox_04_06_noamb/d57_iou_03_05 \
--hoi \
--dataset_file hico_gt \
--model_name hoi_share_qpos_ezero_shiftbbox_03_05_noamb \
--hoi_path data/hico_20160224_det/ \
--num_obj_classes 80 \
--num_verb_classes 117 \
--backbone resnet50 \
--set_cost_bbox 2.5 \
--set_cost_giou 1 \
--bbox_loss_coef 2.5 \
--giou_loss_coef 1 \
--ts_begin 0 \
--eval_gt \
--lr 1e-5 \
--lr_backbone 1e-6 \
--start_epoch 58
--resume outputs/hico_abla/12_6_hoi_share_qpos_ezero_shiftbbox_04_06_noamb/d52/checkpoint.pth

