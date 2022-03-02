Download checkpoints from this Google Drive [link](https://drive.google.com/drive/folders/1kf4O42nohcQYfU9brDK3E-8_mGHPyCQW?usp=sharing)

### Notes:

1. I accidentally set the number of classes to 80 for `yolor_p6_voc.pt` (should be 20). For now, I manually remove boxes with invalid class labels in the code. 

2. `mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco_20210903_104808-b92c91f1.pth` is downloaded from [here](https://download.openmmlab.com/mmdetection/v2.0/swin/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco_20210903_104808-b92c91f1.pth). See the [mmdet repo](https://github.com/open-mmlab/mmdetection/tree/master/configs/swin) for more details.
   - Note that we only use the detection head of the mask rcnn (the mask head is not used). The architecture is identical to faster rcnn

