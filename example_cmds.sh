############################################################# 
###### yolor voc
############################################################# 

### generate and save/dump raw detection results so that we can reuse them for future experiments (evaluating clean performance and robustness)
python main_yolor.py --device 0 --data yolor/data/voc.yaml --names yolor/data/voc.names --weights checkpoints/yolor_p6_voc.pt --num-line 30 --save-raw 

### evaluate clean AP for yolor on voc 
# (matching class labels)
python main_yolor.py --device 0 --data yolor/data/voc.yaml --names yolor/data/voc.names --weights checkpoints/yolor_p6_voc.pt --load-raw --match-class --num-line 30  --alpha 0.8 --beta 0.5 --ioa-prune-thres 0.6 --map 
# (not matching class labels)
python main_yolor.py --device 0 --data yolor/data/voc.yaml --names yolor/data/voc.names --weights checkpoints/yolor_p6_voc.pt --load-raw --num-line 30  --alpha 0.8 --beta 0.5 --ioa-prune-thres 0.6 --map 

### the following two lines are optional since clean_eval.py will be called in main_yolor.py
#python clean_eval.py  --dataset voc --model yolor_p6_voc  --match-class --num-line 30  --alpha 0.8 --beta 0.5 --ioa-prune-thres 0.6 --preprocess --save 
#python clean_eval.py  --dataset voc --model yolor_p6_voc  --num-line 30  --alpha 0.8 --beta 0.5 --ioa-prune-thres 0.6 --preprocess --save 

### evaluate certified robustness with certain confidence thresholds (CertR@0.8)
# note that the confidence thresholds depend on the clean evaluation results: we need to know which confidence threshold gives the specified clean recall for robustness evaluation
# (matching class labels)
python main_yolor.py --device 0 --data yolor/data/voc.yaml --names yolor/data/voc.names --weights checkpoints/yolor_p6_voc.pt --load-raw --match-class --num-line 30  --base-conf-thres 0.66 --masked-conf-thres 0.83 --ioa-prune-thres 0.6 --certify --patch-size 0.01 --certify-ioa-thres 0.0 
# (not matching class labels)
python main_yolor.py --device 0 --data yolor/data/voc.yaml --names yolor/data/voc.names --weights checkpoints/yolor_p6_voc.pt --load-raw --num-line 30  --base-conf-thres 0.66 --masked-conf-thres 0.83 --ioa-prune-thres 0.6 --certify --patch-size 0.01 --certify-ioa-thres 0.0 

### (optional) evaluate the clean performance of the vanilla object detector
#python main_yolor.py --device 0 --data yolor/data/voc.yaml --names yolor/data/voc.names --weights checkpoints/yolor_p6_voc.pt --load-raw --num-line 0 --map  
#python clean_eval.py  --dataset voc --model yolor_p6_voc --num-line 0  --preprocess --save 


############################################################# 
###### yolor coco
############################################################# 
### dump raw detections
python main_yolor.py --device 0 --data yolor/data/coco.yaml --names yolor/data/coco.names --weights checkpoints/yolor_p6_coco.pt --num-line 30 --save-raw 
### clean evaluation
python main_yolor.py --device 0 --data yolor/data/coco.yaml --names yolor/data/coco.names --weights checkpoints/yolor_p6_coco.pt --load-raw --num-line 0 --map 
python main_yolor.py --device 0 --data yolor/data/coco.yaml --names yolor/data/coco.names --weights checkpoints/yolor_p6_coco.pt --load-raw --num-line 30  --alpha 0.7 --beta 0.5 --ioa-prune-thres 0.6 --map 
#python clean_eval.py  --dataset coco --model yolor_p6_coco  --match-class --num-line 30  --alpha 0.7 --beta 0.5 --ioa-prune-thres 0.6 --save 
#python clean_eval.py  --dataset coco --model yolor_p6_coco --num-line 30  --alpha 0.7 --beta 0.5 --ioa-prune-thres 0.6 --save 
### robustness evaluation (CertR@0.6)
python main_yolor.py --device 0 --data yolor/data/coco.yaml --names yolor/data/coco.names --weights checkpoints/yolor_p6_coco.pt --load-raw --match-class --num-line 30  --base-conf-thres 0.44 --masked-conf-thres 0.72 --ioa-prune-thres 0.6 --certify --patch-size 0.01 --certify-ioa-thres 0.0 
python main_yolor.py --device 0 --data yolor/data/coco.yaml --names yolor/data/coco.names --weights checkpoints/yolor_p6_coco.pt --load-raw --num-line 30  --base-conf-thres 0.44 --masked-conf-thres 0.72 --ioa-prune-thres 0.6 --certify --patch-size 0.01 --certify-ioa-thres 0.0 
### vanilla
#python clean_eval.py  --dataset coco --model yolor_p6_coco --num-line 0  --preprocess --save 
#python main_yolor.py --device 0 --data yolor/data/coco.yaml --names yolor/data/coco.names --weights checkpoints/yolor_p6_coco.pt --load-raw --match-class --num-line 30  --alpha 0.7 --beta 0.5 --ioa-prune-thres 0.6 --map 


############################################################# 
###### swin voc
############################################################# 
### dump raw detections
python main_mmdet.py mmdet/configs/objseeker/faster_rcnn_swin_s_voc.py  checkpoints/faster_rcnn_swin_s_voc.pth --eval mAP --num-line 30 --save-raw 
### clean evaluation
python main_mmdet.py mmdet/configs/objseeker/faster_rcnn_swin_s_voc.py  checkpoints/faster_rcnn_swin_s_voc.pth --eval mAP --load-raw --match-class --num-line 30  --alpha 0.8 --beta 0.5 --ioa-prune-thres 0.6 --map 
python main_mmdet.py mmdet/configs/objseeker/faster_rcnn_swin_s_voc.py  checkpoints/faster_rcnn_swin_s_voc.pth --eval mAP --load-raw --num-line 30  --alpha 0.8 --beta 0.5 --ioa-prune-thres 0.6 --map 
#python clean_eval.py  --dataset voc --model swin_voc --match-class --num-line 30  --alpha 0.8 --beta 0.5 --ioa-prune-thres 0.6 --preprocess --save 
#python clean_eval.py  --dataset voc --model swin_voc --num-line 30  --alpha 0.8 --beta 0.5 --ioa-prune-thres 0.6 --preprocess --save 
### robustness evaluation (CertR@0.8)
python main_mmdet.py mmdet/configs/objseeker/faster_rcnn_swin_s_voc.py  checkpoints/faster_rcnn_swin_s_voc.pth --eval mAP --load-raw --match-class --num-line 30  --base-conf-thres 0.62 --masked-conf-thres 0.81 --ioa-prune-thres 0.6 --certify --patch-size 0.01 --certify-ioa-thres 0.0 
python main_mmdet.py mmdet/configs/objseeker/faster_rcnn_swin_s_voc.py  checkpoints/faster_rcnn_swin_s_voc.pth --eval mAP --load-raw --num-line 30  --base-conf-thres 0.62 --masked-conf-thres 0.81 --ioa-prune-thres 0.6 --certify --patch-size 0.01 --certify-ioa-thres 0.0 
### vanilla
#python main_mmdet.py mmdet/configs/objseeker/faster_rcnn_swin_s_voc.py  checkpoints/faster_rcnn_swin_s_voc.pth --eval mAP --load-raw --num-line 0  --map 
#python clean_eval.py  --dataset voc --model swin_voc --num-line 0 --preprocess --save 



############################################################# 
###### swin coco
############################################################# 
# save
### dump raw detections
python main_mmdet.py mmdet/configs/objseeker/faster_rcnn_swin_s_coco.py checkpoints/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco_20210903_104808-b92c91f1.pth --eval bbox --num-line 30 --save-raw 
### clean evaluation
python main_mmdet.py mmdet/configs/objseeker/faster_rcnn_swin_s_coco.py checkpoints/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco_20210903_104808-b92c91f1.pth --eval bbox --load-raw --match-class --num-line 30  --alpha 0.9 --beta 0.8 --ioa-prune-thres 0.6 --map 
python main_mmdet.py mmdet/configs/objseeker/faster_rcnn_swin_s_coco.py checkpoints/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco_20210903_104808-b92c91f1.pth --eval bbox --load-raw --num-line 30  --alpha 0.9 --beta 0.8 --ioa-prune-thres 0.6 --map 
#python clean_eval.py  --dataset coco --model swin_coco  --match-class --num-line 30  --alpha 0.9 --beta 0.8 --ioa-prune-thres 0.6 --save 
#python clean_eval.py  --dataset coco --model swin_coco --num-line 30  --alpha 0.9 --beta 0.8 --ioa-prune-thres 0.6 --save 
### robustness evaluation (CertR@0.6)
python main_mmdet.py mmdet/configs/objseeker/faster_rcnn_swin_s_coco.py checkpoints/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco_20210903_104808-b92c91f1.pth --eval bbox --load-raw --match-class --num-line 30  --base-conf-thres 0.36 --masked-conf-thres 0.9 --ioa-prune-thres 0.6 --certify --patch-size 0.01 --certify-ioa-thres 0.0 
python main_mmdet.py mmdet/configs/objseeker/faster_rcnn_swin_s_coco.py checkpoints/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco_20210903_104808-b92c91f1.pth --eval bbox --load-raw --num-line 30  --base-conf-thres 0.36 --masked-conf-thres 0.9 --ioa-prune-thres 0.6 --certify --patch-size 0.01 --certify-ioa-thres 0.0 
### vanilla
#python main_mmdet.py mmdet/configs/objseeker/faster_rcnn_swin_s_coco.py  checkpoints/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco_20210903_104808-b92c91f1.pth --eval bbox --load-raw --num-line 0  --map 
#python clean_eval.py  --dataset coco --model swin_coco --num-line 0 --preprocess --save 
