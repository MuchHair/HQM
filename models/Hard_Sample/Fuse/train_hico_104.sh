## 1.6 104
python -m torch.distributed.launch \
--nproc_per_node=8  \
--use_env \
main.py \
--pretrained outputs/1_26_hoi_share_qpos_occ_att_each_shiftbbox_random_occ_pass/checkpoint0070.pth \
--output_dir outputs/1_26_hoi_share_qpos_occ_att_each_shiftbbox_random_occ_pass/d70 \
--hoi \
--dataset_file hico_gt \
--model_name hoi_share_qpos_occ_att_each_shiftbbox_random_occ_pass \
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
--start_epoch 71
--resume outputs/12_29_hoi_share_qpos_occ_att_each_shiftbbox_random/checkpoint0057.pth