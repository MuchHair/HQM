# qpic (convert from detr-r50-pre.pth)
python -m torch.distributed.launch \
--nproc_per_node=8 \
--use_env \
main.py \
--pretrained params/detr-r50-pre-vcoco.pth \
--output_dir outputs/vcoco/01_14_vcoco_baseline_convert \
--hoi \
--dataset_file vcoco \
--hoi_path data/v-coco \
--num_obj_classes 81 \
--num_verb_classes 29 \
--backbone resnet50 \
--set_cost_bbox 2.5 \
--set_cost_giou 1 \
--bbox_loss_coef 2.5 \
--giou_loss_coef 1 \
--lr_drop 60 \
--epoch 90 \

# qpic_hard_memory
python -m torch.distributed.launch \
--nproc_per_node=8  \
--use_env \
main.py \
--pretrained params/detr-r50-pre-vcoco-inside.pth \
--output_dir outputs/vcoco/01_14_hoi_learnable_shiftbbox_occ_random_cxcy_04_06_top100_rand05 \
--hoi \
--dataset_file vcoco_gt \
--model_name hoi_learnable_shiftbbox_occ_random_cxcy \
--hoi_path data/v-coco/ \
--num_obj_classes 81 \
--num_verb_classes 29 \
--backbone resnet50 \
--set_cost_bbox 2.5 \
--set_cost_giou 1 \
--bbox_loss_coef 2.5 \
--giou_loss_coef 1 \
--ts_begin 0 \
--eval_gt \
--lr_drop 60 \
--epoch 90 \
--eval \


# eval
python generate_vcoco_officialv2.py \
--param_path outputs/vcoco/01_14_hoi_learnable_shiftbbox_occ_random_cxcy_04_06_top100_rand05/checkpoint0082.pth \
--save_path outputs/vcoco/01_14_hoi_learnable_shiftbbox_occ_random_cxcy_04_06_top100_rand05/vcoco_82.pickle \
--hoi_path data/v-coco \
--model_name hoi_learnable_shiftbbox_occ_random_cxcy \

cd data/v-coco
python vsrl_eval.py /home/xubin/code/qpic/outputs/vcoco/01_14_hoi_learnable_shiftbbox_occ_random_cxcy_04_06_top100_rand05/vcoco_82.pickle
