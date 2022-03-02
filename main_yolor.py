# adapted from yolor/test.py
import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

from yolor.utils.google_utils import attempt_load
from yolor.utils.datasets import create_dataloader
from yolor.utils.general import coco80_to_coco91_class, check_dataset, check_file, check_img_size, box_iou, \
    non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, clip_coords, set_logging, increment_path
from yolor.utils.loss import compute_loss
from yolor.utils.metrics import ap_per_class,compute_ap
from yolor.utils.plots import plot_images, output_to_target
from yolor.utils.torch_utils import select_device, time_synchronized

from yolor.models.models import *
def load_classes(path):
    # Loads *.names file at 'path'
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))  # filter removes empty strings (such as last line)

################################################################################
# load objseeker library
import joblib
from clean_eval import clean_eval
from objseeker.defense import YOLO_wrapper,ObjSeekerModel
#set confidence threshold for saving raw detection results
SAVE_RAW_BASE_CONF_THRES = 0.001#0.01#0.001
SAVE_RAW_MASK_CONF_THRES = 0.1#0.6#0.1
################################################################################

def test(data,
         weights=None,
         batch_size=16,
         imgsz=640,
         #conf_thres=0.001,
         #iou_thres=0.6,  # for NMS
         #save_json=False,
         single_cls=False,
         augment=False,
         verbose=False,
         model=None,
         dataloader=None,
         save_dir=Path(''),  # for saving images
         save_txt=False,  # for auto-labelling
         save_conf=False,
         plots=True,
         log_imgs=0,
         base_output_list=None,
         raw_masked_output_list=None,
         args=None):  # number of logged images

    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device = next(model.parameters()).device  # get model device

    else:  # called directly
        set_logging()
        if isinstance(args.device,str):
            device = select_device(args.device, batch_size=batch_size) 
        else:
            device = args.device
        save_txt = args.save_txt  # save *.txt labels

        # Directories
        #save_dir = Path(increment_path(Path(args.project) / args.name, exist_ok=args.exist_ok))  # increment run
        #(save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
        save_dir = None
        # Load model
        model = Darknet(args.cfg).to(device)

        # load model
        try:
            ckpt = torch.load(weights[0], map_location=device)  # load checkpoint
            ckpt['model'] = {k: v for k, v in ckpt['model'].items() if model.state_dict()[k].numel() == v.numel()}
            model.load_state_dict(ckpt['model'], strict=False)
        except:
            load_darknet_weights(model, weights[0])
        imgsz = check_img_size(imgsz, s=64)  # check img_size

    # Half
    half = device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    # Configure
    model.eval()
    is_coco = data.endswith('coco.yaml')  # is COCO dataset
    with open(data) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)  # model dict
    check_dataset(data)  # check
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Logging
    log_imgs, wandb = min(log_imgs, 100), None  # ceil
    try:
        import wandb  # Weights & Biases
    except ImportError:
        log_imgs = 0

    # Dataloader
    if not training:
        img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
        _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
        path = data['test'] if args.task == 'test' else data['val']  # path to val/test images
        dataloader = create_dataloader(path, imgsz, batch_size, 64, args, pad=0.5, rect=True)[0]

    seen = 0
    try:
        names = model.names if hasattr(model, 'names') else model.module.names
    except:
        names = load_classes(args.names)
    coco91class = coco80_to_coco91_class()
    s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    p, r, f1, mp, mr, map50, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class, wandb_images = [], [], [], [], []


    ################################################################################
    # basic objseeker setup
    args.device = device
    # build model
    model = YOLO_wrapper(model)
    if args.load_raw:
        model = None
    model = ObjSeekerModel(model,args)
    if args.certify:
        cr_res = [0,0,0,0,0]#far_vul_cnt_iou_total,far_vul_cnt_total,close_vul_cnt_total,over_vul_cnt_total,obj_cnt_total

    if not args.map:
        dataloader = tqdm(dataloader)

    for batch_i, (img, targets, paths, shapes) in enumerate(dataloader):
        img = img.to(device, non_blocking=True)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        nb, _, height, width = img.shape  # batch size, channels, height, width
        whwh = torch.Tensor([width, height, width, height]).to(device)
        # Disable gradients
        with torch.no_grad():
            # inference
            if args.one_line: # do inference with one-line function call
                output = model(img)
            else: #do inference with precomputed detections
                if args.load_raw: # if we already loaded the precomputed detection
                    raw_masked_output = raw_masked_output_list[batch_i]
                    base_output = base_output_list[batch_i]
                else: # otherwise we generate the precomputed detection now
                    raw_masked_output = model.get_raw_masked_boxes(img)
                    base_output = model.base_model(img,conf_thres=args.base_conf_thres,nms_iou_thres=args.base_nms_iou_thres)
                if args.save_raw: # if we want to save the raw detection to the disk
                    raw_masked_output_list.append(raw_masked_output)
                    base_output_list.append([x.detach().cpu() for x in base_output])
                # run the inference with precomputed detections
                output = model(img,raw_masked_output_precomputed=raw_masked_output,base_output_precomputed=base_output)

        if args.certify: # gather certification stats
            ground_truth = []
            for img_i in range(len(img)):
                labels = targets[targets[:, 0] == img_i, 1:] # ground truth labels for this image
                labels[:, 1:5] = xywh2xyxy(labels[:, 1:5]) * whwh
                labels = labels[:,[1,2,3,4,0]] # -> xyxy cls
                ground_truth.append(labels)

            res = model.certify(img,raw_masked_output,ground_truth,args.patch_size,args.certify_iou_thres,args.certify_ioa_thres)
            cr_res = [cr_res[x]+res[x] for x in range(5)]


        if args.save_det: # dump the detection results
            for si, pred in enumerate(output):
                labels = targets[targets[:, 0] == si, 1:]
                nl = len(labels)
                tcls = labels[:, 0].tolist() if nl else []  # target class
                seen += 1

                if len(pred) == 0:
                    if nl:
                        stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                    continue
                clip_coords(pred, (height, width))

                # Append to text file for voc
                path = Path(paths[si])
                if 'voc' in args.data:
                    x = pred.clone()
                    x[:, :4] = scale_coords(img[si].shape[1:], x[:, :4], shapes[si][0], shapes[si][1])  # to original
                    with open(os.path.join(args.clean_dir,path.stem + '.txt'), 'w') as f:
                        for *xyxy, conf, cls in x:
                            if 'voc' in args.data and cls>19:
                                continue
                            xyxy = torch.tensor(xyxy).tolist()
                            line = (cls, conf,*xyxy)
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                # Append to pycocotools JSON dictionary
                elif 'coco' in args.data:
                    # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
                    box = pred[:, :4].clone()  # xyxy
                    scale_coords(img[si].shape[1:], box, shapes[si][0], shapes[si][1])  # to original shape
                    box = xyxy2xywh(box)  # xywh
                    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                    for p, b in zip(pred.tolist(), box.tolist()):
                        jdict.append({'image_id': image_id,
                                      'category_id': coco91class[int(p[5])] if is_coco else int(p[5]),
                                      'bbox': [round(x, 3) for x in b],
                                      'score': round(p[4], 5)})


            if 'coco' in args.data:
                pred_json = os.path.join(args.clean_dir,'coco_predictions.bbox.json')  # predictions json
                if len(jdict)>0:
                    with open(pred_json, 'w') as f:
                        json.dump(jdict, f)


    if args.certify: # print robustness stats
        obj_cnt_total = cr_res[-1]
        cr_res = cr_res[:4]
        cr_res = [100-x/obj_cnt_total*100 for x in cr_res]
        print(cr_res)
        print('Certified recall results:')
        print('Far-patch (IoU): {:.2f}%; Far-patch (IoA): {:.2f}%; Close-patch (IoA): {:.2f}%; Over-patch (IoA): {:.2f}%'.format(*cr_res))
    else:
        cr_res = []

    return cr_res,base_output_list,raw_masked_output_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    # original arguments from yolo
    parser.add_argument('--weights', nargs='+', type=str, default='yolor_p6.pt', help='model.pt path(s)')
    parser.add_argument('--data', type=str, default='data/coco.yaml', help='*.data path')
    parser.add_argument('--batch-size', type=int, default=8, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=1280, help='inference size (pixels)')
    #parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold') # renamed to base-conf-thres (see below)
    #parser.add_argument('--iou-thres', type=float, default=0.65, help='IOU threshold for NMS') # renamed to base-nms-iou-thres (see below)
    parser.add_argument('--task', default='val', help="'val', 'test', 'study'")
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--project', default='runs/test', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--cfg', type=str, default='yolor/cfg/yolor_p6.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='yolor/data/coco.names', help='*.cfg path')
    
    # objectseeker argumenets
    # we call vanilla object detector "base detector"
    parser.add_argument('--base-conf-thres', type=float, default=0.6, help='conf thres of base detector')
    parser.add_argument('--base-nms-iou-thres', type=float, default=0.65, help='IoU threshold for NMS in base object detector')
    parser.add_argument('--num-line', type=int, default=30, help='number of lines $k$')
    parser.add_argument('--masked-conf-thres', type=float, default=0.8, help='conf thres for masked predictions ($tau_mask$)')
    parser.add_argument('--pruning-mode', type=str, default='ioa', help='ioa or iou')
    parser.add_argument('--ioa-prune-thres', type=float, default=0.6, help='ioa threshold for box filtering/pruning ($tau_ioa$)')
    parser.add_argument('--iou-prune-thres', type=float, default=0.8, help='iou threshold for box filtering/pruning ($tau_iou$; not used in the main body; see appendix)')
    parser.add_argument('--match-class', action='store_true', help='whether consider class label in the pruning (will affect robustness property)')
    parser.add_argument('--alpha', type=float, default=0.8, help='minimum masked confidence threshold (used for clean AP calculation; see appendix)')
    parser.add_argument('--beta', type=float, default=0.5, help='(used for clean AP calculation; see appendix)')
    
    # certification arguments
    parser.add_argument('--certify-iou-thres', type=float, default=0.0, help='iou thres for robustness certification')
    parser.add_argument('--certify-ioa-thres', type=float, default=0.5, help='ioa thres for robustness certification')
    parser.add_argument('--patch-size', type=float, default=0.01, help='patch size, in percentage of image pixels')

    # functional arguments
    parser.add_argument('--dump-dir', type=str, default='dump', help='root dir for precomputed raw detections')
    parser.add_argument('--clean-dir', type=str, default='clean_det', help='dir for saving clean detection results')
    parser.add_argument('--save-det', action='store_true', help='whether to save detection results')
    parser.add_argument('--save-raw', action='store_true', help='whether to save raw detection results')
    parser.add_argument('--load-raw', action='store_true', help='whether to load precomputed raw detection')
    parser.add_argument('--one-line', action='store_true', help='whether use one-line inference mode without any precomputing')
    parser.add_argument('--certify', action='store_true', help='whether to certification')
    parser.add_argument('--map', action='store_true', help='whether to calculate ap (need to change the confidence threshold)')

    args = parser.parse_args()
    #args.save_json |= args.data.endswith('coco.yaml')
    args.data = check_file(args.data)  # check file
    print(args)

    base_output_list = None 
    raw_masked_output_list = None

    # setup directory, load/save detection results
    if args.save_raw or args.load_raw:
        DUMP_DIR = args.dump_dir
        if not os.path.exists(DUMP_DIR):
            os.mkdir(DUMP_DIR)

        # a dumb way to extract model and dataset names
        model_name = args.weights[0].split('/')[-1].split('.')[0]
        dataset_name = args.data.split('/')[-1].split('.')[0]

        prefix = 'raw_masked_output_{}_{}_{}_{}_{}'.format(dataset_name,model_name,SAVE_RAW_MASK_CONF_THRES,args.base_nms_iou_thres,args.batch_size)
        DUMP_DIR_MASK = os.path.join(DUMP_DIR,prefix)
        if not os.path.exists(DUMP_DIR_MASK):
            os.mkdir(DUMP_DIR_MASK)  
    
        prefix = 'base_output_{}_{}_{}_{}_{}'.format(dataset_name,model_name,SAVE_RAW_BASE_CONF_THRES,args.base_nms_iou_thres,args.batch_size)
        DUMP_DIR_BASE = os.path.join(DUMP_DIR,prefix)
        if not os.path.exists(DUMP_DIR_BASE):
            os.mkdir(DUMP_DIR_BASE)  
        
        if args.load_raw:# load saved detection results
            base_output_list = joblib.load(os.path.join(DUMP_DIR_BASE,'base_output_list.z'))
            if args.num_line>0:
                raw_masked_output_list = joblib.load(os.path.join(DUMP_DIR_MASK,'raw_masked_output_list_{}.z'.format(args.num_line)))
            else: # vanilla predictions
                raw_masked_output_list = [None for i in range(len(base_output_list))]
        else:# prepare to gather raw detection results and save
            base_output_list = [] # detection results for vanilla object detectors on the original images
            raw_masked_output_list = [] # detection results on masked images
            # set the flags to the saving mode
            args.base_conf_thres = SAVE_RAW_BASE_CONF_THRES
            #args.conf_thres = SAVE_RAW_BASE_CONF_THRES
            args.masked_conf_thres = SAVE_RAW_MASK_CONF_THRES
    
    
    if args.map:
        conf_thres_list = np.linspace(0,0.99,100)[::-1] # the list of confidence thresholds to vary
    else:
        conf_thres_list = [args.base_conf_thres]

    args.save_det = args.save_det or args.map

    if args.save_det: #setup save directory
        CLEAN_DIR = args.clean_dir
        CLEAN_BASE_DIR = CLEAN_DIR
        if not os.path.exists(CLEAN_DIR):
            os.mkdir(CLEAN_DIR)
        CLEAN_DIR = os.path.join(CLEAN_DIR,dataset_name)
        if not os.path.exists(CLEAN_DIR):
            os.mkdir(CLEAN_DIR)
        match_class = 'cls' if args.match_class else 'nocls'
        CLEAN_DIR = os.path.join(CLEAN_DIR,'{}_{}_{}_{}_{}_{}'.format(model_name,args.num_line,args.ioa_prune_thres,args.iou_prune_thres,match_class,args.pruning_mode))
        
        if not os.path.exists(CLEAN_DIR):
            os.mkdir(CLEAN_DIR)


    for conf in tqdm(conf_thres_list):
        args.base_conf_thres = conf
        if args.map:
            args.masked_conf_thres = max(args.alpha,conf + (1-conf)*args.beta) # get masked confidence threshold # see appendix for more details

        if args.save_det:
            args.clean_dir = os.path.join(CLEAN_DIR,'{:.3f}_{:.5f}'.format(conf,args.masked_conf_thres))
            if not os.path.exists(args.clean_dir):
                os.mkdir(args.clean_dir) 

        cr_res,base_output_list,raw_masked_output_list = test(args.data,
             args.weights,
             args.batch_size,
             args.img_size,
             args.single_cls,
             args.augment,
             args.verbose,
             base_output_list=base_output_list,
             raw_masked_output_list=raw_masked_output_list,
             args=args
             )

    if args.save_raw: # dump detections
        joblib.dump(raw_masked_output_list,os.path.join(DUMP_DIR_MASK,'raw_masked_output_list_{}.z'.format(args.num_line)))
        joblib.dump(base_output_list,os.path.join(DUMP_DIR_BASE,'base_output_list.z'))
    if args.certify:
        match_class = 'cls' if args.match_class else 'nocls'
        res_dir = 'results_{}'.format(args.pruning_mode)
        if not os.path.exists(res_dir):
            os.mkdir(res_dir)
        if args.pruning_mode == 'ioa':
            joblib.dump(cr_res,'results_{}/cr_{}_{}_{}_{}_{}_{}_{}_{}.z'.format(args.pruning_mode,dataset_name,model_name,args.num_line,args.masked_conf_thres,args.ioa_prune_thres,args.certify_ioa_thres,args.patch_size,match_class))
        elif args.pruning_mode == 'iou':
            joblib.dump(cr_res,'results_{}/cr_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.z'.format(args.pruning_mode,dataset_name,model_name,args.num_line,args.masked_conf_thres,args.ioa_prune_thres,args.certify_ioa_thres,args.patch_size,match_class,args.iou_prune_thres,args.certify_iou_thres))



    if args.save_det:
        print('calling clean_eval.py...')
        args.save=True
        args.preprocess=True
        args.load=False
        args.single = not args.map
        args.dataset = dataset_name
        args.model = model_name
        args.clean_dir = CLEAN_BASE_DIR
        clean_eval(args)
