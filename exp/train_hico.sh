python -m torch.distributed.launch \
--nproc_per_node=8  \
--use_env \
main.py \
--pretrained params/detr-r50-pre_inside.pth  \
--output_dir logs/hico/HQM_7_10 \
--hoi \
--dataset_file hico_gt \
--model_name HQM \
--hoi_path data/hico_20160224_det/ \
--num_obj_classes 80 \
--num_verb_classes 117 \
--backbone resnet50 \
--set_cost_bbox 2.5 \
--set_cost_giou 1 \
--bbox_loss_coef 2.5 \
--giou_loss_coef 1 \
--find_unused_parameters \
--AJL


python -m torch.distributed.launch \
--nproc_per_node=8  \
--use_env \
main.py \
--pretrained params/detr-r50-pre_inside.pth  \
--output_dir logs/hico/HQM_7_10 \
--hoi \
--dataset_file hico_gt \
--model_name GBS \
--hoi_path data/hico_20160224_det/ \
--num_obj_classes 80 \
--num_verb_classes 117 \
--backbone resnet50 \
--set_cost_bbox 2.5 \
--set_cost_giou 1 \
--bbox_loss_coef 2.5 \
--giou_loss_coef 1 \
--find_unused_parameters \
--AJL


python -m torch.distributed.launch \
--nproc_per_node=8  \
--use_env \
main.py \
--pretrained params/detr-r50-pre_inside.pth  \
--output_dir logs/hico/HQM_7_10 \
--hoi \
--dataset_file hico_gt \
--model_name AMM \
--hoi_path data/hico_20160224_det/ \
--num_obj_classes 80 \
--num_verb_classes 117 \
--backbone resnet50 \
--set_cost_bbox 2.5 \
--set_cost_giou 1 \
--bbox_loss_coef 2.5 \
--giou_loss_coef 1 \
