# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import time
import warnings

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

#from apis_test import multi_gpu_test, single_gpu_test
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector
import numpy as np 
from tqdm import tqdm

################################################################################
# load objseeker library
from objseeker.defense import FRCNN_wrapper,ObjSeekerModel
from clean_eval import clean_eval
import joblib
#set confidence threshold for saving raw detection results
SAVE_RAW_BASE_CONF_THRES = 0.05
SAVE_RAW_MASK_CONF_THRES = 0.1
################################################################################

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    # mmdet arguments
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--show-score-thr',
        type=float,
        default=0.3,
        help='score threshold (default: 0.3)')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)

    ################################################################################
    # objectseeker argumenets
    parser.add_argument('--conf-thres', type=float, default=0.05, help='object confidence threshold')#for vanilla object detectors

    # defense parameters
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
    ################################################################################

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both '
            'specified, --options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    if cfg.model.get('neck'):
        if isinstance(cfg.model.neck, list):
            for neck_cfg in cfg.model.neck:
                if neck_cfg.get('rfp_backbone'):
                    if neck_cfg.rfp_backbone.get('pretrained'):
                        neck_cfg.rfp_backbone.pretrained = None
        elif cfg.model.neck.get('rfp_backbone'):
            if cfg.model.neck.rfp_backbone.get('pretrained'):
                cfg.model.neck.rfp_backbone.pretrained = None

    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    rank, _ = get_dist_info()
    # allows not to create
    if args.work_dir is not None and rank == 0:
        mmcv.mkdir_or_exist(osp.abspath(args.work_dir))
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        json_file = osp.join(args.work_dir, f'eval_{timestamp}.json')

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu',strict=False)
    #print(cfg.model)
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES


    ################################################################################
    # basic setup
    if 'coco' in args.config:
        dataset_name = 'coco'
        num_cls = 80
        conf_rescale_a = 1.36 #determined by heurstics; used to rescale the confidence value 
        conf_rescale_b = 4.7
    elif 'voc' in args.config:
        dataset_name = 'voc'
        num_cls = 20
        conf_rescale_a = 1.53
        conf_rescale_b = 24.8
    #elif 'kitti' in args.config:
    #    dataset_name = 'kitti'
    #    num_cls = 3
    #    conf_rescale_a = 1.53
    #    conf_rescale_b = 24.8

    # setup directory, load/save detection results
    if args.save_raw or args.load_raw:
        DUMP_DIR = args.dump_dir
        if not os.path.exists(DUMP_DIR):
            os.mkdir(DUMP_DIR)

        model_name = 'swin_{}'.format(dataset_name)#hard-coded 

        batch_size = 1 # the code for frcnn is assumed we are using batch size of one
        prefix = 'raw_masked_output_{}_{}_{}_{}_{}'.format(dataset_name,model_name,SAVE_RAW_MASK_CONF_THRES,args.base_nms_iou_thres,batch_size)
        DUMP_DIR_MASK = os.path.join(DUMP_DIR,prefix)
        if not os.path.exists(DUMP_DIR_MASK):
            os.mkdir(DUMP_DIR_MASK)  
    
        prefix = 'base_output_{}_{}_{}_{}_{}'.format(dataset_name,model_name,SAVE_RAW_BASE_CONF_THRES,args.base_nms_iou_thres,batch_size)
        DUMP_DIR_BASE = os.path.join(DUMP_DIR,prefix)
        if not os.path.exists(DUMP_DIR_BASE):
            os.mkdir(DUMP_DIR_BASE)  
        

        if args.load_raw: # load saved detection results
            base_output_list = joblib.load(os.path.join(DUMP_DIR_BASE,'base_output_list.z'))
            if args.num_line>0:
                raw_masked_output_list = joblib.load(os.path.join(DUMP_DIR_MASK,'raw_masked_output_list_{}.z'.format(args.num_line)))
            else:
                raw_masked_output_list = [None for i in range(len(base_output_list))]
        else: # prepare to gather raw detection results and save
            base_output_list = [] # detection results for vanilla object detectors on the original images
            raw_masked_output_list = [] # detection results on masked images
            # set the flags to the saving mode
            args.base_conf_thres = SAVE_RAW_BASE_CONF_THRES
            args.conf_thres = SAVE_RAW_BASE_CONF_THRES
            args.masked_conf_thres = SAVE_RAW_MASK_CONF_THRES

    args.device = 'cuda'
    # build model
    model_ = MMDataParallel(model, device_ids=[0]) # add a mmdet wrapper...
    model_ = FRCNN_wrapper(model_,conf_rescale_a,conf_rescale_b)
    if args.load_raw:
        model_ = None
    if args.certify:
        cr_res = [0,0,0,0,0]#far_vul_cnt_iou_total,far_vul_cnt_total,close_vul_cnt_total,over_vul_cnt_total,obj_cnt_total

    if args.map:
        conf_thres_list = np.linspace(0,0.99,100)[::-1]
    else:
        conf_thres_list = [args.base_conf_thres]

    args.save_det = args.save_det or args.map
    if args.save_det:
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
        save_flg = False
        args.base_conf_thres = conf
        if args.map:
            args.masked_conf_thres = max(args.alpha,conf + (1-conf)*args.beta)

        if args.save_det:
            args.clean_dir = os.path.join(CLEAN_DIR,'{:.3f}_{:.5f}'.format(conf,args.masked_conf_thres))
            if not os.path.exists(args.clean_dir):
                os.mkdir(args.clean_dir) 

        model = ObjSeekerModel(model_,args)

        outputs = []
        dataset = data_loader.dataset
        if not args.map:
            data_loader = tqdm(data_loader)
        for batch_i, data in enumerate(data_loader):
            with torch.no_grad():

                img_metas = data['img_metas'][0].data[0]
                fn = img_metas[0]['filename']
                fn = fn.split('/')[-1]
                img_id = fn.split('.')[0]

                if args.one_line: # do not run the inference with precomputed detections
                    output = model(data)
                else:
                    if args.load_raw: # if we already loaded the precomputed detection
                        raw_masked_output = raw_masked_output_list[batch_i]
                        base_output = base_output_list[batch_i]
                    else: # otherwise we generate the precomputed detection now
                        raw_masked_output = model.get_raw_masked_boxes(data)
                        base_output = model.base_model(data,conf_thres=args.conf_thres)
                    if args.save_raw: # save the detection is needed
                        raw_masked_output_list.append(raw_masked_output)
                        base_output_list.append([x.detach().cpu() for x in base_output])
                    # run the inference with precomputed detections
                    output = model(data,raw_masked_output_precomputed=raw_masked_output,base_output_precomputed=base_output)
                save_flg = len(output)>0 or save_flg
                output = FRCNN_wrapper.convert(output,num_cls=num_cls,img_metas=data['img_metas'],rescale=True)
                outputs.extend(output)


                if ('voc' in dataset_name or 'kitti' in dataset_name) and args.save_det:
                    with open(os.path.join(args.clean_dir,'{}.txt'.format(img_id)),'w') as wf:
                        for cls_i,boxes in enumerate(output[0]):
                            for box_i,box in enumerate(boxes):
                                wf.write('{} {} {} {} {} {}\n'.format(cls_i,box[4],box[0],box[1],box[2],box[3]))

            if args.certify: # gather certification stats
                #get and re-format the ground-truth labels.
                ann = dataset.get_ann_info(batch_i)
                bboxes = torch.from_numpy(ann['bboxes']).to(args.device)
                labels = torch.from_numpy(ann['labels']).to(args.device)
                scale_factor = torch.from_numpy(data['img_metas'][0]._data[0][0]['scale_factor']).to(args.device)

                ground_truth = [torch.cat([bboxes*scale_factor,labels[:,None]],dim=1)]#xyxy cls

                res = model.certify(data,raw_masked_output,ground_truth,args.patch_size,args.certify_iou_thres,args.certify_ioa_thres)
                cr_res = [cr_res[x]+res[x] for x in range(5)]

        rank, _ = get_dist_info()
        if rank == 0:
            if save_flg and args.save_det and 'coco' in dataset_name:
                dataset.results2json(outputs, os.path.join(args.clean_dir,'coco_predictions'))
        
    if args.save_raw: # dump detections
        joblib.dump(raw_masked_output_list,os.path.join(DUMP_DIR_MASK,'raw_masked_output_list_{}.z'.format(args.num_line)))
        joblib.dump(base_output_list,os.path.join(DUMP_DIR_BASE,'base_output_list.z'))
    if args.certify: # print robustness stats
        obj_cnt_total = cr_res[-1]
        cr_res = cr_res[:4]
        cr_res = [100-x/obj_cnt_total*100 for x in cr_res]
        print('Certified recall results:')
        print('Far-patch (IoU): {:.2f}%; Far-patch (IoA): {:.2f}%; Close-patch (IoA): {:.2f}%; Over-patch (IoA): {:.2f}%'.format(*cr_res))

        res_dir = 'results_{}'.format(args.pruning_mode)
        match_class = 'cls' if args.match_class else 'nocls'
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
        
    ################################################################################
if __name__ == '__main__':
    main()
