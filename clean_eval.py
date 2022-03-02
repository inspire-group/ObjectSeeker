from objseeker.voc_eval import voc_eval,voc_ap
from objseeker.coco_eval import COCOeval_OS,compute_ap
from pycocotools.coco import COCO
from tqdm import tqdm
import numpy as np
import argparse
import os
import joblib
import shutil

def clean_eval(args):
	DATASET = args.dataset
	if DATASET == 'voc':
		DATA_ROOT = os.path.join('data','VOCdevkit','VOC2007') #set it to your data path
		annopath = os.path.join(DATA_ROOT,'Annotations','{}.xml')
		classes_name = ['aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']
		imagesetfile = os.path.join(DATA_ROOT,'ImageSets','Main','test.txt')
	elif DATASET == 'coco':
		with open(os.path.join('yolor','data','coco.names')) as f:#set it to your data path
			lines= f.readlines()
		classes_name = [x.strip() for x in lines]
		anno_json = os.path.join('yolor','data','coco','annotations','instances_val2017.json')#set it to your data path
	#elif DATASET == 'kitti':
	#	DATA_ROOT = os.path.join('data','kitti_yolo','raw')
	#	annopath = os.path.join(DATA_ROOT,'label_2','{}.txt')
	#	classes_name = ['car','person','cyclist']
	#	imagesetfile = os.path.join(DATA_ROOT,'val.txt')

	dataset_name = args.dataset
	model_name = args.model
	prec_list = []
	rec_list = []
	CLEAN_DIR = args.clean_dir
	CLEAN_DIR = os.path.join(CLEAN_DIR,dataset_name)
	match_class = 'cls' if args.match_class else 'nocls'
	CLEAN_DIR = os.path.join(CLEAN_DIR,'{}_{}_{}_{}_{}_{}'.format(model_name,args.num_line,args.ioa_prune_thres,args.iou_prune_thres,match_class,args.pruning_mode))


	conf_thres_list = np.linspace(0,0.99,100)[::-1] if not args.single else [args.base_conf_thres]

	if args.dataset in ['voc','kitti'] and args.preprocess:
		for conf in tqdm(conf_thres_list):
			masked_conf = max(args.alpha,conf + (1-conf)*args.beta) if not args.single else args.masked_conf_thres
			input_dir = os.path.join(CLEAN_DIR,'{:.3f}_{:.5f}'.format(conf,masked_conf))
			output_dir = os.path.join(input_dir,'det_cls')
			if os.path.exists(output_dir):
				shutil.rmtree(output_dir)
			os.mkdir(output_dir)
			fn_list = os.listdir(input_dir)
			for fn in fn_list:
				if 'txt' not in fn or fn.split('.')[0] in classes_name:
					continue
				#if :
				#	continue
				with open(os.path.join(input_dir,fn),'r') as rf:
					lines = rf.readlines()
				splitlines = [x.strip().split(" ") for x in lines]
				img_id = fn[:-4]
				for line in splitlines:
					with open(os.path.join(output_dir,classes_name[int(line[0])]+'.txt'),'a') as wf:
						wf.write('{} {} {} {} {} {}\n'.format(img_id,line[1],line[2],line[3],line[4],line[5]))

	if not args.load:

		prec = np.zeros([len(conf_thres_list),len(classes_name)]) # precision at different confidence threshold of Base Detector
		rec = np.zeros([len(conf_thres_list),len(classes_name)]) # recall at different confidence threshold of Base Detector
		if DATASET in ['voc','kitti']:
			eval_func = voc_eval #if DATASET == 'voc' else kitti_eval
			for i,conf in enumerate(tqdm(conf_thres_list)):
				masked_conf = max(args.alpha,conf + (1-conf)*args.beta) if not args.single else args.masked_conf_thres
				detpath = os.path.join(CLEAN_DIR,'{:.3f}_{:.5f}'.format(conf,masked_conf),'det_cls','{}.txt')

				for j,clss in enumerate(classes_name): # get precision and recall for each class
					if not os.path.exists(detpath.format(clss)):
						prec[i,j] = 1
						rec[i,j] = 0
					else:
						r,p = eval_func(detpath, annopath, imagesetfile, clss)
						prec[i,j]=p
						rec[i,j]=r

		elif DATASET == 'coco':
			for i,conf in enumerate(tqdm(conf_thres_list)):
				masked_conf = max(args.alpha,conf + (1-conf)*args.beta) if not args.single else args.masked_conf_thres
				pred_json = os.path.join(CLEAN_DIR,'{:.3f}_{:.5f}'.format(conf,masked_conf),'coco_predictions.bbox.json')
				if not os.path.exists(pred_json):
					prec[i] = 1
					rec[i] = 0				
					print('nores',conf)
				else:
					anno = COCO(anno_json)  # init annotations api
					pred = anno.loadRes(pred_json)  # init predictions api
					cocoeval = COCOeval_OS(anno, pred, 'bbox')
					cocoeval.params.maxDets = [100]
					cocoeval.params.areaRng = [[0 ** 2, 1e5 ** 2]]
					cocoeval.params.areaRngLbl = ['all']
					cocoeval.params.iouThrs = [0.5]
					cocoeval.params.recThrs = [0.0]
					cocoeval.evaluate()
					r,p = cocoeval.accumulate_rec_prec()
					prec[i] = p
					rec[i] = r

		if args.save:
			joblib.dump(prec,os.path.join(CLEAN_DIR,'prec_list_{}_{}.z'.format(args.alpha,args.beta)))
			joblib.dump(rec,os.path.join(CLEAN_DIR,'rec_list_{}_{}.z'.format(args.alpha,args.beta)))
		if args.single:
			rec = np.mean(rec,1) # reduce to averaged recall 
			prec = np.mean(prec,1) # reduce to averaged recall 
			print('prec',prec,'rec',rec)
	else:
		prec = joblib.load(os.path.join(CLEAN_DIR,'prec_list_{}_{}.z'.format(args.alpha,args.beta)))
		rec = joblib.load(os.path.join(CLEAN_DIR,'rec_list_{}_{}.z'.format(args.alpha,args.beta)))

	ap_list = []
	for j,clss in enumerate(classes_name): # get ap for each class
		if DATASET in ['voc','kitti']:
			ap = voc_ap(rec[:,j], prec[:,j]) 
		elif DATASET == 'coco':
			ap = compute_ap(rec[:,j], prec[:,j])
		ap_list.append(ap)
		print('{:15}: {:.3f}'.format(clss,ap))
	print('{:15}: {:.3f}'.format('mAP',np.mean(ap_list)))


	rec = np.mean(rec,1) # reduce to averaged recall 
	prec = np.mean(prec,1) # reduce to averaged recall 
	for r in np.arange(0.2,1,0.1): 
		try:
			idx = np.searchsorted(rec,r)
			print('Clean recall:',rec[idx],prec[idx],conf_thres_list[idx])
		except:
			print('clean recall:',r,'NA')

if __name__ == '__main__':
    
	parser = argparse.ArgumentParser()
	parser.add_argument("--dataset",default='voc',choices=['voc','coco'],type=str,help='dataset name')# haven't support kitti yet
	parser.add_argument("--clean-dir",default='clean_det',type=str)
	parser.add_argument("--model",default='yolor_p6_voc',type=str,help='type of base detector')

	# defense arguments
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

	# functional arguments
	parser.add_argument("--save",action='store_true',help='save the evaluation results (for further analysis)')
	parser.add_argument("--preprocess",action='store_true',help='preprocess the voc detection results')
	parser.add_argument("--load",action='store_true',help='load the evaluation results')
	parser.add_argument("--single",action='store_true',help='evaluate clean performance at a single confidence threshold')

	args = parser.parse_args()
	
	clean_eval(args)


