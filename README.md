# ObjectSeeker: Certifiably Robust Object Detection against Patch Hiding Attacks via Patch-agnostic Masking

By [Chong Xiang](http://xiangchong.xyz/), Alexander Valtchanov, [Saeed Mahloujifar](https://smahloujifar.github.io/), [Prateek Mittal](https://www.princeton.edu/~pmittal/)

Code for "[ObjectSeeker: Certifiably Robust Object Detection against Patch Hiding Attacks via Patch-agnostic Masking](https://arxiv.org/abs/2202.01811)".

#### Check out this [paper list for adversarial patch research](https://github.com/xiangchong1/adv-patch-paper-list) for fun!

## Files

```shell
├── README.md                        #this file 
├── requirement.txt                  #required packages
├── example_cmd.sh                   #example commands to run the code
| 
├── main_yolor.py                    #run ObjectSeeker with yolor 
├── main_mmdet.py                    #run ObjectSeeker with swin (mmdet)
├── clean_eval.py                    #scripts for clean AP evaluation; can be imported into main_yolor.py or main_mmdet.py
| 
├── objseeker                        #utils of objectseeker
|   ├── defense.py                   #implement the objectseeker defense
|   ├── coco_eval.py                 #utils for coco clean evaluation
|   └── voc_eval.py                  #utils for voc clean evaluation
|
├── yolor                            #scripts from the original yolor implementation
|   ├── data                        
|   ├── ....
|
├── mmdet                            #configs for mmdet/swin
|   ├── data                        
|   └── configs
|
└── checkpoints                      #directory for checkpoints
    ├── README.md                    #details of checkpoints
    └── ...                          #model checkpoints
```

## Datasets

[YOLOR](https://github.com/WongKinYiu/yolor) is based on [YOLOv5](https://github.com/ultralytics/yolov5), and thus needs pre-processed data from ultralytics. [mmdet](https://github.com/open-mmlab/mmdetection) (swin) uses official datasets directly. 

- [VOC](http://host.robots.ox.ac.uk/pascal/VOC/)
  - (mmdet.swin) The official dataset can be downloaded using `torchvision.datasets.VOCDetection`. we use test2007 for our evaluation.
    - specify the data directory in `mmdet/configs/_base_/datasets/voc0712.py`.
  - (yolor) see this [script](https://github.com/ultralytics/yolov5/blob/master/data/VOC.yaml) to download and generate the ultralytics-YOLO style VOC dataset.
    - specify the data directory in `yolor/data/voc.yaml`.
- [COCO](https://cocodataset.org/#home)
  - (mmdet.swin) Download the official dataset (both data and annotations). we use val2007 for evaluation.
    - specify the data directory in `mmdet/configs/_base_/datasets/coco_detection.py`.
  - (yolor) see this [script](https://github.com/ultralytics/yolov5/blob/master/data/scripts/get_coco.sh) for downloading ultralytics-YOLO style COCO dataset.
    - specify the data directory in `yolor/data/coco.yaml`.

## Dependency

- PyTorch
  - follow the [official website](https://pytorch.org/get-started/locally/) to install PyTorch.
- mmdet
  - follow this [tutorial](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/get_started.md) to install mmdet (and mmcv).
- Others
  - see [`requirement.txt`](requirement.txt); install via `pip install -r requirement.txt`. 

## Usage

- See **Files** for details of each file. 
- Setup the datasets and dependency as discussed above. 
- Download checkpoints from this Google Drive [link](https://drive.google.com/drive/folders/1kf4O42nohcQYfU9brDK3E-8_mGHPyCQW?usp=sharing) and move them to `checkpoints/`.
- See [`example_cmds.sh`](example_cmds.sh) for example commands for running the code.

If anything is unclear, please open an issue or contact Chong Xiang (cxiang@princeton.edu).

## Citations

If you find our work useful in your research, please consider citing:

```tex
@article{xiang2022objectseeker,
  title={ObjectSeeker: Certifiably Robust Object Detection against Patch Hiding Attacks via Patch-agnostic Masking},
  author={Xiang, Chong and Valtchanov, Alexander and Mahloujifar, Saeed and Mittal, Prateek},
  journal={arXiv preprint arXiv:2202.01811},
  year={2022}
}
```
