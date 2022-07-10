python -m torch.distributed.launch \
--nproc_per_node=8 \
--use_env \
main.py \
--pretrained params/detr-r50-pre_inside.pth \
--output_dir outputs/1_30_hoi_share_qpos_ezero_shiftbbox_04_06 \
--hoi \
--dataset_file hico_gt \
--model_name hoi_share_qpos_ezero_shiftbbox_04_06 \
--hoi_path data/hico_20160224_det/ \
--num_obj_classes 80 \
--num_verb_classes 117 \
--backbone resnet50 \
--set_cost_bbox 2.5 \
--set_cost_giou 1 \
--bbox_loss_coef 2.5 \
--giou_loss_coef 1 \
--ts_begin 0 \
--eval_gt

# decay
python -m torch.distributed.launch \
--nproc_per_node=8 \
--use_env \
main.py \
--pretrained outputs/01_20_hoi_share_qpos_gtbbox_noise/checkpoint0051.pth \
--output_dir outputs/01_20_hoi_share_qpos_gtbbox_noise/d51 \
--hoi \
--dataset_file hico_gt \
--model_name hoi_share_qpos_gtbbox_noise \
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
--start_epoch 52 \
--resume outputs/01_15_hoi_hardm_query_att_each_pos_top_src_len_rand04/checkpoint0051.pth


# nipairs  --pretrained params/detr-r50-pre_inside.pth \
python -m torch.distributed.launch \
--nproc_per_node=8 \
--use_env \
main.py \
--pretrained outputs/01_17_hoi_share_qpos_ezero_shiftbbox_nipairs_04_06/checkpoint0035.pth \
--output_dir outputs/01_17_hoi_share_qpos_ezero_shiftbbox_nipairs_04_06/d35 \
--hoi \
--dataset_file hico_gt_nipairs \
--model_name hoi_share_qpos_ezero_shiftbbox_nipairs_04_06 \
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
--start_epoch 36


python -m torch.distributed.launch \
--nproc_per_node=8 \
--use_env \
main.py \
--pretrained outputs/12_28_hoi_share_qpos_occ_att_each_shiftbbox/checkpoint0051.pth \
--output_dir outputs/12_28_hoi_share_qpos_occ_att_each_shiftbbox/d51 \
--hoi \
--dataset_file hico_gt \
--model_name hoi_share_qpos_occ_att_each_shiftbbox \
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
--start_epoch 52 \
--resume outputs/12_28_hoi_share_qpos_occ_att_each_shiftbbox_check/checkpoint.pth



python -m torch.distributed.launch \
--nproc_per_node=1  \
--use_env \
main_store.py \
--pretrained params/detr-r50-pre_inside.pth \
--output_dir outputs/1_30_hoi_share_qpos_ezero_shiftbbox_04_06 \
--hoi \
--dataset_file hico_gt \
--model_name hoi_share_qpos_ezero_shiftbbox_04_06 \
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
--eval \
--resume /home/xubin2/Documents/code/qpic/outputs/hico_abla/12_2_hoi_qpos_ezero_cos_kl5_shiftbbox_04_06/checkpoint0066.pth


# store
python -m torch.distributed.launch \
--nproc_per_node=1  \
--use_env \
main_store.py \
--pretrained params/detr-r50-pre_inside.pth \
--output_dir outputs/12_26_hoi_qpos_cos_kl_shiftbbox_04_06 \
--hoi \
--dataset_file hico_gt_store \
--model_name hoi_qpos_cos_kl_shiftbbox_04_06 \
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
--eval



python -m torch.distributed.launch \
--nproc_per_node=8  \
--use_env \
main.py \
--pretrained params/detr-r50-pre_inside.pth \
--output_dir outputs/12_15_hoi_share_qpos_gtbbox_nipairs/d46 \
--hoi \
--dataset_file hico_gt \
--model_name hoi_share_qpos_gtbbox_nipairs \
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
--start_epoch 47
