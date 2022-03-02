from sklearn.cluster import DBSCAN
import torch
from math import ceil
import numpy as np 
from collections import defaultdict
import torchvision
import itertools

try:
    #load helper functions from yolo repo...
    from yolor.utils.general import non_max_suppression
except:
    print('skipping yolor library')


"""
!!! Note on image/box coordinates: 
vanilla object detectors resize the original images to higher resolutions for better performance
all operations in this script work on the resized (large) image sizes/coordinates


all image are in the shape of BCHW
therefore, for a 2-D image, the first dimension is height, and the second is width

in contrast, the output of vanilla object detectors, as well as ground-truth annotation, has the format of xyxy; x is for the width dimension, y is for the height

"""



def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2


class YOLO_wrapper(object):
    # a wrapper for vanilla YOLO detector 
    def __init__(self, yolo_detector):
        self.yolo_detector = yolo_detector
        self.yolo_detector.eval()
    def __call__(self,img,conf_thres,mask=None,nms_iou_thres=0.65):
        # do vanilla object detection.
        if mask is None:
            return non_max_suppression(self.yolo_detector(img)[0].detach(),conf_thres=conf_thres,iou_thres=nms_iou_thres) 
        else:
            return non_max_suppression(self.yolo_detector(img*mask)[0].detach(),conf_thres=conf_thres,iou_thres=nms_iou_thres)
        # the detection result is a list of tensor; one tensor for one image
        # the format is `xyxy conf cls` # the coordinates is for the resize images, not for the original image size

class FRCNN_wrapper(object):
    # a wrapper for vanilla Faster-RCNN detector (also compatible with other mmdet detectors)
    # ATTN!! the implementation of mmdet frcnn assumes a batch size of 1
    def __init__(self, frcnn_detector,conf_rescale_a,conf_rescale_b):
        self.frcnn_detector = frcnn_detector
        self.frcnn_detector.eval()
        # the following two are used for rescaling the confidence -- FRCNN's confidence are too concentrated around 0.99
        # if we do not do rescaling, the precision-recall curve changes drastically around 0.99 and gives a bad AP computation result
        # note that conventional object detection does not have this issue since their AP calculation is only affected by relative magnitude
        self.conf_rescale_a = conf_rescale_a
        self.conf_rescale_b = conf_rescale_b

    def __call__(self,data,conf_thres,mask=None,rescale=False):
        # do vanilla object detection. convert the format to the yolo-style
        output = [] 
        #rescale = False
        if mask is None:
            result = self.frcnn_detector(img=data['img'],img_metas = data['img_metas'],return_loss=False, rescale=rescale)[0]
        else:
            result = self.frcnn_detector(return_loss=False, rescale=rescale,img_metas = data['img_metas'],img=[x.to('cuda')*mask for x in data['img']] )[0]
        # the `result` is a list of list, one list for one image
        # each list has num_cls numpy array in the format of `xyxy conf`
        # we are converting the `result` to 'xyxy conf cls' (the format used in yolo)
        # the coordinates are rescaled to the original image size if `rescale = True`
        # !!ATTN!! in our defense implementation, we set `rescale = False`. None of our operations are working on the original image coordinates...

        # convert output format
        # the format is `xyxy conf cls` 
        for cls_i,boxes in enumerate(result):
            boxes = torch.from_numpy(boxes[boxes[:,-1]>conf_thres]).to('cuda')###############
            if len(boxes)>0:
                boxes[:,4] = torch.tan(boxes[:,4]*self.conf_rescale_a)/self.conf_rescale_b # rescale the confidence value
                output.append(torch.cat([boxes,torch.ones((len(boxes),1),device=boxes.device)*cls_i],dim=1))
        if len(output)>0:
            output = [torch.cat(output)]
        else:
            output = [torch.zeros((0,6))]
        return output

    @staticmethod
    def convert(yolo_output,num_cls,rescale=True,img_metas=None):
        # convert the yolo-style `xyxy conf cls` back to the mmdet-style format
        # Here, `rescale` is set to True by default. The coordinates are rescaled to the original image sizes for evaluation
        ret = []
        for yolo_boxes,meta in zip(yolo_output,img_metas):
            if rescale:
                scale_factor = meta._data[0][0]['scale_factor']
                scale_factor = torch.tensor(scale_factor,device=yolo_boxes.device)
            tmp_list = []
            for cls_i in range(num_cls):
                tmp = yolo_boxes[yolo_boxes[:,-1]==cls_i][:,:5]
                if len(tmp)>0 and rescale:
                    bboxes = tmp[:,:4]
                    tmp[:,:4] = (bboxes.view(bboxes.size(0), -1, 4) / scale_factor).view(bboxes.size()[0], -1)
                tmp_list.append(tmp.detach().cpu().numpy())
            ret.append(tmp_list)
        return ret



class ObjSeekerModel(object):

    '''
    The defense class that implement inference and certification
    
    ---main methods---
    __init__():             take vanilla object detector as input and init the defense instance
    __call__():             do inference (Algorithm 1 and Algorithm 3)
        -> ioa_inference()  prune box with IoA
        -> iou_inference()  prune box with IoU
    certify():              perform certification analysis (Algorithm 2)
        -> ioa_certify()    certification with IoA
        -> iou_certify()    certification with IoU

    ---other helper functions---
    see each function for more details
    '''

    def __init__(self, base_model, args=None):
        super(ObjSeekerModel, self).__init__()
        
        # base detector setup (unrelated to the defense)
        self.base_model = base_model # base model
        self.base_conf_thres = args.base_conf_thres #the confidence threshold for base prediction boxes that will be used for box pruning 

        # mask parameters
        self.num_line = args.num_line #number of lines in one dimension
        self.masked_conf_thres = args.masked_conf_thres #only retained masked prediction boes whose confidence exceeds this threshold
      
        # box pruning parameters
        self.pruning_mode = args.pruning_mode # 'ioa' or 'iou'
        self.ioa_prune_thres = args.ioa_prune_thres # discard masked prediction boxes whose ioa with base prediction box exceeds this threshold
        self.iou_prune_thres = args.iou_prune_thres # nonoverlap box nms pruning (when args.pruning_mode=='iou')
        self.dbscan = DBSCAN(eps=0.1, min_samples=1,metric='precomputed') #dbscan instance for overlap boxes pruning
        self.match_class = args.match_class # whether consider class labels

        #misc
        self.device = args.device
        self.half = self.device!='cpu' # used for yolor -- yolor use half() mode for inference efficiency


    def __call__(self,img,base_output_precomputed=None,raw_masked_output_precomputed=None):
        # defense inference forward pass
        # img: input image [B,3,H,W] for yolo; or a dict for img and its meta information for mmdet
        # base_output_precomputed. precomputed base detector detection results (on the original imaeges). If it is None, compute the results on the fly
        # raw_masked_output_precomputed. precomputed detection results on masked image. If it is None, compute the results on the fly

        # return a list of detection results (the same format as the output of YOLO_Wrapper). one list element corresponds to one image
        if self.pruning_mode == 'ioa':  
            return self.ioa_inference(img,base_output_precomputed=base_output_precomputed,raw_masked_output_precomputed=raw_masked_output_precomputed)
        elif self.pruning_mode == 'iou':
            return self.iou_inference(img,base_output_precomputed=base_output_precomputed,raw_masked_output_precomputed=raw_masked_output_precomputed)


    def ioa_inference(self,img,base_output_precomputed=None,raw_masked_output_precomputed=None,save_dir=None,paths=None,names=None):
        # the same usage as __call__()

        # setup
        if isinstance(img,torch.Tensor): #yolo input is tensor
            num_img,_,height,width = img.shape
        else: # the input for mmdet is not torch.Tensor
            num_img,_,height,width = img['img'][0].shape

        # get coordinates of partition points
        idx_list = self.get_mask_idx(height,width)
        # for each image, get filtered masked predictions
        defense_output = [torch.zeros((0,6)).to(self.device) for i in range(num_img)] # to hold the final output 
        masked_output = [[] for i in range(num_img)] # to hold masked boxes from different masked images
        

        if base_output_precomputed is None:
            base_output_precomputed = self.base_model(img,conf_thres=self.base_conf_thres)
        
        base_output = [pred[pred[:,4]>self.base_conf_thres].to(self.device) for pred in base_output_precomputed] # each is xyxy conf cls # used as part of the output
        if self.num_line <=0: #vanilla inference
            return base_output

        for img_i in range(num_img):
            clip_coords(base_output[img_i], (height, width)) # clip negative values
            if self.match_class: # add class offset so that boxes of different classes can never have intersection
                base_output[img_i][:,:4] = base_output[img_i][:,:4]+ base_output[img_i][:, 5:6] * 4096            

        # next, we are going to gather masked box and perform box pruning 
        for ii,jj in idx_list: # for each partition point
            mask_list = self.gen_mask(ii,jj,height,width) if raw_masked_output_precomputed is None else list(range(4))
            
            for mask_i,mask in enumerate(mask_list): # for each mask (4 masks in total)
                if raw_masked_output_precomputed is None: # no precomputed detections. do detection now!
                    masked_output_ = self.base_model(img,mask=mask,conf_thres=self.masked_conf_thres)  # each is xyxy conf cls 
                
                for img_i in range(num_img): # for each image
                    if raw_masked_output_precomputed is None: # get masked boxes for this image, depending on if precomputed detection is available
                        masked_boxes = masked_output_[img_i]
                    else:
                        masked_boxes = raw_masked_output_precomputed[img_i][(ii,jj)][mask_i].to(self.device)
                        masked_boxes = masked_boxes[masked_boxes[:,4]>self.masked_conf_thres]

                    base_boxes = base_output[img_i]
                    clip_coords(masked_boxes, (height, width)) # clip negative values
                    if len(masked_boxes) >0:
                        if self.match_class: # class offset
                            masked_boxes[:,:4] = masked_boxes[:,:4] + masked_boxes[:, 5:6] * 4096
                        # calculate ioa, and filter boxes
                        if len(base_boxes)>0: ###False:
                            ioa = self.box_ioa(masked_boxes[:,:4],base_boxes[:,:4]) 
                            ioa = torch.max(ioa,dim=1)[0]
                            fil = ioa < self.ioa_prune_thres
                            masked_boxes = masked_boxes[fil] 
                        masked_output[img_i].append(masked_boxes)

        for img_i in range(num_img):
            masked_boxes = masked_output[img_i]
            masked_boxes = torch.cat(masked_boxes) if len(masked_boxes) > 0 else torch.zeros((0,6)).to(self.device)
            if len(masked_boxes)>1: ###False:
                masked_boxes = self.unionize_cluster(masked_boxes) # box unionizing
            base_boxes = base_output[img_i]
            if self.match_class: # remove class offset
                masked_boxes[:,:4] = masked_boxes[:,:4] - masked_boxes[:, 5:6] * 4096
                base_boxes[:,:4] = base_boxes[:,:4] - base_boxes[:, 5:6] * 4096
            defense_output[img_i] = torch.cat([base_boxes,masked_boxes])

        return defense_output

    def get_raw_masked_boxes(self,img):
        # generate masked detection boxes (i.e., generate precomputed masked prediction). 
        # then we can dump/load the masked boxes to save computation in our experiment
        if isinstance(img,torch.Tensor):
            num_img,_,height,width = img.shape
        else:
            num_img,_,height,width = img['img'][0].shape

        raw_masked_output = [defaultdict(list) for i in range(num_img)] 
        idx_list = self.get_mask_idx(height,width)
        for ii,jj in idx_list:
            mask_list = self.gen_mask(ii,jj,height,width) 
            for mask in mask_list:
                masked_output = self.base_model(img,mask=mask,conf_thres=self.masked_conf_thres)  # each is xyxy conf cls # 
                for i,pred in enumerate(masked_output):
                    if pred is None:
                        raw_masked_output[i][(ii,jj)].append(None)
                    else:
                        raw_masked_output[i][(ii,jj)].append(pred.detach().cpu())
        return raw_masked_output

    # helper functions for inference 
    

    def unionize_cluster(self,overlap_boxes):
        # box unionizing
        # cluster overlap_boxes, and merge each cluster into one box

        # calculate "distances" between boxes; the distances is based on ioa; the distance matrix will be used for cluster
        ioa = self.box_ioa(overlap_boxes[:,:4],overlap_boxes[:,:4])
        # calculate pair-wise distance
        distance = 1-torch.maximum(ioa,ioa.T)
        distance = distance.cpu().numpy()

        # dbscan clustering
        cluster = self.dbscan.fit_predict(distance)
        num_cluster = np.max(cluster)+1
        cluster = torch.from_numpy(cluster).to(self.device)

        unionized_boxes = torch.zeros((num_cluster,6)).to(self.device) #xyxy conf cls

        for cluster_i in range(num_cluster):
            boxes = overlap_boxes[cluster==cluster_i]
            unionized_boxes[cluster_i,:2] = torch.min(boxes[:,:2],dim=0)[0]
            unionized_boxes[cluster_i,2:4] = torch.max(boxes[:,2:4],dim=0)[0]
            unionized_boxes[cluster_i,4] = torch.max(boxes[:,4]) # take the highest confidence
            unionized_boxes[cluster_i,5] = boxes[0,5] # take a "random" class

        return unionized_boxes


    def gen_mask(self,ii,jj,height,width):

        #generate 4 mask tensors for location (ii,jj) on a (height,width) image

        mask_list = []
        mask = torch.ones((1,1,height,width)).to(self.device)
        mask[...,:ii,:] = 0
        mask = mask.half() if self.half else mask
        mask_list.append(mask)
        
        mask = torch.ones((1,1,height,width)).to(self.device)
        mask[...,ii:,:] = 0
        mask = mask.half() if self.half else mask
        mask_list.append(mask)

        mask = torch.ones((1,1,height,width)).to(self.device)
        mask[...,:,:jj] = 0
        mask = mask.half() if self.half else mask
        mask_list.append(mask)
        
        mask = torch.ones((1,1,height,width)).to(self.device)
        mask[...,:,jj:] = 0
        mask = mask.half() if self.half else mask
        mask_list.append(mask)

        return mask_list


    def get_mask_idx(self,height,width):
        # generate a list of (ii,jj) coordinates
        if self.num_line<=0:
            return []
        h_stride = ceil(height / (self.num_line+1))
        w_stride = ceil(width / (self.num_line+1))
        ii_idx = list(range(h_stride,height,h_stride))
        jj_idx = list(range(w_stride,width,w_stride))

        idx_list = zip(ii_idx,jj_idx)
        return idx_list


    def box_ioa(self,box1, box2):
        # slighlt modify the code for box_iou to calculate box_ioa
        # the output is inter / area1
        def box_area(box):
            # box = 4xn
            return (box[2] - box[0]) * (box[3] - box[1])

        area1 = box_area(box1.T)
        area2 = box_area(box2.T)

        # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
        inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
        return inter / (area1[:, None])



    def certify(self,img,raw_masked_output_precomputed,ground_truth,patch_size,certify_iou_thres,certify_ioa_thres,save_dir=None,paths=None,scale_factor=None):
        # img: input image [B,3,W,H]
        # raw_masked_output_precomputed: generated from get_raw_masked_boxes
        # ground_truth: the ground truth information 
        # patch_size: size of the patch
        # certify_iou_thres: the threshold to certify iou robustness
        # certify_ioa_thres: the threshold to certify ioa robustness
        # return far_vul_cnt_iou,far_vul_cnt,close_vul_cnt,over_vul_cnt : the number of vulnerable objects
        # return obj_cnt: the number of total objects
        if self.pruning_mode == 'ioa':
            return self.ioa_certify(img,raw_masked_output_precomputed,ground_truth,patch_size,certify_ioa_thres,scale_factor=scale_factor)
        elif self.pruning_mode == 'iou':
            return self.iou_certify(img,raw_masked_output_precomputed,ground_truth,patch_size,certify_iou_thres,certify_ioa_thres,scale_factor=scale_factor)


    def ioa_certify(self,img,raw_masked_output_precomputed,ground_truth,patch_size,certify_ioa_thres,save_dir=None,paths=None,scale_factor=None):
        # see certify() for usage

        if isinstance(img,torch.Tensor):
            num_img,_,height,width = img.shape
        else: # the input for mmdet is not torch.Tensor
            num_img,_,height,width = img['img'][0].shape

        aspect_ratio = 1# for different patch shapes...
        patch_size = (height*width*patch_size/aspect_ratio)**0.5 # patch_size % of image pixel

        patch_size_ii = int(patch_size)
        patch_size_jj = int(patch_size * aspect_ratio)

        obj_cnt = 0 
        far_vul_cnt,close_vul_cnt,over_vul_cnt = 0,0,0
        far_vul_cnt_iou = 0
        for img_i in range(num_img): # for each image
            # get masked prediction       
            raw_masked_dict = raw_masked_output_precomputed[img_i]
            labels = ground_truth[img_i].to(self.device) # ground truth labels for this image
            tbox = labels[:,:4]
            
            num_boxes = len(labels)
            obj_cnt += num_boxes
            
            # a binary map indicating our defense is robust to adversarial pixels at each !!pixel!! locations
            # note that we have one slice of robustness map (shape [height,width]) for every box and every mask
            robust_bitmap_ioa = torch.zeros((num_boxes,4,height,width)).to(self.device) # 4 is for four masks for one (ii,jj) coordinate

            for (ii,jj) in raw_masked_dict.keys():
                masked_boxes_list = raw_masked_dict[(ii,jj)] # a list for four masked predictions
                for mask_i,masked_boxes in enumerate(masked_boxes_list):
                    if masked_boxes is None or len(masked_boxes)==0:
                        continue
                    masked_boxes = masked_boxes[masked_boxes[:,4]>self.masked_conf_thres]
                    if len(masked_boxes)==0:
                        continue
                    masked_boxes = masked_boxes.to(self.device)
                    clip_coords(masked_boxes, (height, width))

                    if self.match_class: # add class offets
                        gt_boxes = tbox + labels[:, 4:5] * 4096
                        masked_boxes = masked_boxes[:,:4] + masked_boxes[:, 5:6] * 4096
                    else:
                        gt_boxes = tbox
                        masked_boxes = masked_boxes[:,:4] 

                    # use overlapping boxes for ioa certification 
                    flag_ioa = self.ioa_certify_flg(gt_boxes, masked_boxes,certify_ioa_thres=certify_ioa_thres)
                    
                    #update each robustness map based on the certification results at this particular mask location
                    self.update_map(robust_bitmap_ioa,flag_ioa,ii,jj,mask_i)

            # with robust_bitmap_ioa and robust_bitmap_iou generated, now we can determine the certified recall

            offset = 0.1 # the offset to distinguish between a close-patch and far-patch
            ii_offset = height*offset
            jj_offset = width*offset


            # add padding so that a large patch can step out of the borders of the image
            # note that if we do not perform padding, the far-patch will have fewer valid locations when we use a large patch size
            pad = True#False
            if pad:
                robust_bitmap_ioa = torch.nn.functional.pad(robust_bitmap_ioa, pad=(patch_size_jj,patch_size_jj,patch_size_ii,patch_size_ii), mode='constant', value=1.0)
            # use a sliding window (i.e., a patch) over the -robust_bitmap. 
            # The output is zero if at least one of the pixel location within the sliding window has zero
            vulnerable_map_ioa = torch.nn.functional.max_pool2d(-robust_bitmap_ioa,kernel_size=(patch_size_ii,patch_size_jj), stride=1,padding = 0)
            vulnerable_map_ioa = torch.sum(vulnerable_map_ioa,dim=1)==0

            #location map. valid patch locations are ones; others are zeros
            location_map = torch.ones_like(vulnerable_map_ioa,dtype=torch.bool,device=vulnerable_map_ioa.device)#.to(self.device)

            # far patch
            for box_i,box in enumerate(tbox):
                box = box.type(torch.int)
                if pad:
                    a,b = int(max(0,box[1]-ii_offset)),int(min(box[3]+1 + ii_offset + patch_size_ii,location_map.shape[1]))
                    c,d = int(max(0,box[0]-jj_offset)),int(min(box[2]+1 + jj_offset + patch_size_jj ,location_map.shape[2]))
                else:
                    a,b = int(max(0,box[1]-patch_size_ii-ii_offset)),int(min(box[3]+1 + ii_offset,location_map.shape[1]))
                    c,d = int(max(0,box[0]-patch_size_jj-jj_offset)),int(min(box[2]+1 + jj_offset,location_map.shape[2]))
                location_map[box_i,a:b,c:d] = False
            
            # use logical_and. our defense is vulnerable when there is a location such that, 1) the location is vulnerable 2) the location is a valid one
            vul_count_ioa = torch.any(torch.any(torch.logical_and(location_map,vulnerable_map_ioa),dim=-1),dim=-1)
            vul_count_ioa = torch.sum(vul_count_ioa)
            far_vul_cnt+=vul_count_ioa.item()

            # close patch
            for box_i,box in enumerate(tbox):
                box = box.type(torch.int)
                if pad:
                    a,b = max(0,box[1]),min(box[3]+1+patch_size_ii,location_map.shape[1])
                    c,d = max(0,box[0]),min(box[2]+1+patch_size_jj,location_map.shape[2])
                else:
                    a,b = max(0,box[1]-patch_size_ii),min(box[3]+1,location_map.shape[1])
                    c,d = max(0,box[0]-patch_size_jj),min(box[2]+1,location_map.shape[2])

                location_map[box_i,a:b,c:d] = True 
            location_map = ~location_map
            vul_count_ioa = torch.any(torch.any(torch.logical_and(location_map,vulnerable_map_ioa),dim=-1),dim=-1)
            vul_count_ioa = torch.sum(vul_count_ioa)
            close_vul_cnt+=vul_count_ioa.item()

            # over patch
            location_map = torch.zeros_like(location_map,dtype=torch.bool,device=vulnerable_map_ioa.device)#.to(self.device)
            for box_i,box in enumerate(tbox):
                box = box.type(torch.int)
                if pad:
                    a,b = max(0,box[1]),min(box[3]+1+patch_size_ii,location_map.shape[1])
                    c,d = max(0,box[0]),min(box[2]+1+patch_size_jj,location_map.shape[2])
                else:
                    a,b = max(0,box[1]-patch_size_ii),min(box[3]+1,location_map.shape[1])
                    c,d = max(0,box[0]-patch_size_jj),min(box[2]+1,location_map.shape[2])

                location_map[box_i,a:b,c:d] = True 
            vul_count_ioa = torch.any(torch.any(torch.logical_and(location_map,vulnerable_map_ioa),dim=-1),dim=-1)
            vul_count_ioa = torch.sum(vul_count_ioa)
            over_vul_cnt+=vul_count_ioa.item()

        far_vul_cnt_iou = obj_cnt
        return far_vul_cnt_iou,far_vul_cnt,close_vul_cnt,over_vul_cnt,obj_cnt

    def ioa_certify_flg(self, box1, box2, certify_ioa_thres):

        def box_area(box):
            # box = 4xn
            return (box[2] - box[0]) * (box[3] - box[1])
        
        area1 = box_area(box1.T)
        area2 = box_area(box2.T)

        # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
        inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
        area2_ind = area2 - inter
        inter = area2 * self.ioa_prune_thres - area2_ind
        ioa = inter / (area1[:, None])
        
        flag = ioa  > certify_ioa_thres
        flag_ioa = torch.any(flag,dim=1)        
        return flag_ioa


    def update_map(self,robust_bitmap,flag,ii,jj,mask_i):
        for box_i,ff in enumerate(flag):
            if ff:
                if mask_i == 0:
                    robust_bitmap[box_i,mask_i,:ii,:]=1
                elif mask_i == 1:
                    robust_bitmap[box_i,mask_i,ii:,:]=1
                elif mask_i == 2:
                    robust_bitmap[box_i,mask_i,:,:jj]=1
                elif mask_i == 3:
                    robust_bitmap[box_i,mask_i,:,jj:]=1 







#############################################################################################################################################################                
#############################################################################################################################################################                
#############################################################################################################################################################                
    #below are for iou inference and iou certification (discussed in the appendix)

    def iou_inference(self,img,base_output_precomputed=None,raw_masked_output_precomputed=None,save_dir=None,paths=None,names=None,scale_factor=None):
        # the same usage as __call__()

        # setup
        if isinstance(img,torch.Tensor): #yolo input is tensor
            num_img,_,height,width = img.shape
        else: # the input for mmdet is not torch.Tensor
            num_img,_,height,width = img['img'][0].shape

        # get coordinates of partition points
        idx_list = self.get_mask_idx(height,width)
        # for each image, get filtered masked predictions
        defense_output = [torch.zeros((0,6)).to(self.device) for i in range(num_img)] # to hold the final output 
        #defense_output = [None for i in range(num_img)] # final output
        
        nonoverlap_output = [[] for i in range(num_img)] # a list for nonoverlapping boxes
        overlap_output = [[] for i in range(num_img)] # a list of overlapping boxes

        # next, we are going to gather all nonoverlap_output and overlap_output
        for ii,jj in idx_list: # for each partition point
            mask_list = self.gen_mask(ii,jj,height,width) if raw_masked_output_precomputed is None else list(range(4))
            
            for mask_i,mask in enumerate(mask_list): # for each mask (4 masks in total)
                if raw_masked_output_precomputed is None: # no precomputed detections. do detection now!
                    masked_output_ = self.base_model(img,mask=mask,conf_thres=self.masked_conf_thres)#,nms_iou_thres=self.base_nms_iou_thres)  # each is xyxy conf cls 
                
                for img_i in range(num_img): # for each image
                    if raw_masked_output_precomputed is None: # get masked boxes for this image, depending on if precomputed detection is available
                        masked_boxes = masked_output_[img_i]
                    else:
                        masked_boxes = raw_masked_output_precomputed[img_i][(ii,jj)][mask_i].to(self.device)
                        masked_boxes = masked_boxes[masked_boxes[:,4]>self.masked_conf_thres]

                    if scale_factor is not None:
                        masked_boxes[:,:4] = masked_boxes[:,:4]*scale_factor

                    # divide masked_boxes to nonoverlap_boxes and overlap_boxes
                    nonoverlap_boxes,overlap_boxes = self.split_overlap_boxes(masked_boxes,ii,jj,height,width,mask_i)
                    nonoverlap_output[img_i].append(nonoverlap_boxes)
                    overlap_output[img_i].append(overlap_boxes)


        if base_output_precomputed is None:
            base_output_precomputed = self.base_model(img,conf_thres=self.base_conf_thres)#,nms_iou_thres=self.base_nms_iou_thres)
        
        base_output = [pred[pred[:,4]>self.base_conf_thres].to(self.device) for pred in base_output_precomputed] # each is xyxy conf cls # used as part of the output

        if scale_factor is not None:
            for i in range(len(base_output)):
                base_output[i][:,:4] = base_output[i][:,:4]*scale_factor

        # first deal with nonoverlap boxes
        # in this implementation, we made some simplifications and directly apply NMS on masked and base boxes.
        for img_i in range(num_img): # for each image
            nonoverlap_masked_boxes = nonoverlap_output[img_i]
            if len(nonoverlap_masked_boxes)>0 or len(base_output[img_i])>0: # there is at least one box 
                nonoverlap_boxes = torch.cat(nonoverlap_masked_boxes+[base_output[img_i]]) # combine base boxes and nonoverlapping boxes 
                clip_coords(nonoverlap_boxes, (height, width)) # clip negative values
                # perform nms
                c = nonoverlap_boxes[:, 5:6] * 4096  # classes
                boxes, scores = nonoverlap_boxes[:, :4] + c, nonoverlap_boxes[:, 4]  # boxes (offset by class), scores
                keep = torchvision.ops.nms(boxes,scores,self.iou_prune_thres) 
                nonoverlap_output[img_i] = nonoverlap_boxes[keep]
        

        # now let's deal with overlapping boxes; similar to what we did in ioa_inference()
        for img_i in range(num_img):
            nonoverlap_boxes = nonoverlap_output[img_i]
            #nonoverlap_boxes = nonoverlap_boxes[nonoverlap_boxes[:,4]>self.base_conf_thres]
            overlap_boxes = torch.cat(overlap_output[img_i])
            clip_coords(nonoverlap_boxes, (height, width)) # clip negative values
            clip_coords(overlap_boxes, (height, width)) # clip negative values
            if len(overlap_boxes) >0:
                if self.match_class: # class offset
                    overlap_boxes[:,:4] = overlap_boxes[:,:4] + overlap_boxes[:, 5:6] * 4096
                    nonoverlap_boxes[:,:4] = nonoverlap_boxes[:,:4]+ nonoverlap_boxes[:, 5:6] * 4096
                if len(nonoverlap_boxes)>0:
                    ioa  = self.box_ioa(overlap_boxes[:,:4],nonoverlap_boxes[:,:4])
                    ioa = torch.max(ioa,dim=1)[0]
                    fil = ioa < self.ioa_prune_thres
                    overlap_boxes = overlap_boxes[fil]
                # merge remaining boxes
                if len(overlap_boxes)>1:
                    overlap_boxes = self.unionize_cluster(overlap_boxes)

                if self.match_class: # remove class offset
                    overlap_boxes[:,:4] = overlap_boxes[:,:4] - overlap_boxes[:, 5:6] * 4096
                    nonoverlap_boxes[:,:4] = nonoverlap_boxes[:,:4] - nonoverlap_boxes[:, 5:6] * 4096

            #generate final defense output by concatenating overlap boxes and nonoverlap boxes
            defense_output[img_i] = torch.cat([overlap_boxes,nonoverlap_output[img_i]])

        return defense_output

    def split_overlap_boxes(self,boxes,ii,jj,height,width,mask_i,offset=0.05):
        #divide boxes into overlap and nonoverlap boxes

        def check_mask_overlap(boxes,ii,jj,height,width,mask_i,offset):
            #helper function, to generate a bool tensor, the element of which indicates if the corresponding box overlaps with the mask
            # we increase the size of mask slightly in case the box is very close to the mask but with no overlapping 
            ii_offset = height*offset
            jj_offset = width*offset

            if mask_i == 0:
                mask_box = [0,0,ii+ii_offset,width]
            elif mask_i == 1:
                mask_box = [ii-ii_offset,0,height,width]
            elif mask_i == 2:
                mask_box = [0,0,height,jj+jj_offset]
            elif mask_i ==3:
                mask_box = [0,jj-jj_offset,height,width]

            mask_box = [mask_box[1],mask_box[0],mask_box[3],mask_box[2]] # get the xyxy coordinates
            flag1 = torch.logical_and(boxes[:,0] <= mask_box[2],mask_box[0]<=boxes[:,2])
            flag2 = torch.logical_and(boxes[:,1] <= mask_box[3],mask_box[1]<=boxes[:,3])
            return torch.logical_and(flag1,flag2)

        overlap = check_mask_overlap(boxes,ii,jj,height,width,mask_i,offset)
        return boxes[~overlap],boxes[overlap]
    
    def iou_certify(self,img,raw_masked_output_precomputed,ground_truth,patch_size,certify_iou_thres,certify_ioa_thres,save_dir=None,paths=None,scale_factor=None):
        # the same usage as certify()
        ioa=True
        iou=True
        if isinstance(img,torch.Tensor):
            num_img,_,height,width = img.shape
        else: # the input for mmdet is not torch.Tensor
            num_img,_,height,width = img['img'][0].shape
        aspect_ratio = 1#8#1#8.# for different patch shapes...
        patch_size = (height*width*patch_size/aspect_ratio)**0.5 # patch_size % of image pixel

        patch_size_ii = int(patch_size)
        patch_size_jj = int(patch_size * aspect_ratio)
        #print(patch_size_ii,patch_size_jj,aspect_ratio,height,width)

        obj_cnt = 0 
        far_vul_cnt,close_vul_cnt,over_vul_cnt = 0,0,0
        far_vul_cnt_iou = 0
        for img_i in range(num_img): # for each image
            # get masked prediction       
            raw_masked_dict = raw_masked_output_precomputed[img_i]
            labels = ground_truth[img_i].to(self.device) # ground truth labels for this image
            tbox = labels[:,:4]
            
            num_boxes = len(labels)
            obj_cnt += num_boxes
            
            # a binary map indicating our defense is robust to adversarial pixels at each pixel locations
            # one for ioa robustness, another for iou robustness
            # note that we have one slice of robustness map (shape [height,width]) for every box and every mask
            if ioa:
                robust_bitmap_ioa = torch.zeros((num_boxes,4,height,width)).to(self.device)
            if iou:
                robust_bitmap_iou = torch.zeros((num_boxes,4,height,width)).to(self.device)

            for (ii,jj) in raw_masked_dict.keys():
                masked_boxes_list = raw_masked_dict[(ii,jj)] # a list for four masked predictions
                for mask_i,masked_boxes in enumerate(masked_boxes_list):

                    if masked_boxes is None or len(masked_boxes)==0:
                        continue
                    masked_boxes = masked_boxes[masked_boxes[:,4]>self.masked_conf_thres]
                    if len(masked_boxes)==0:
                        continue
                    masked_boxes = masked_boxes.to(self.device)
                    clip_coords(masked_boxes, (height, width))

                    if scale_factor is not None:
                        masked_boxes[:,:4] = masked_boxes[:,:4]*scale_factor


                    nonoverlap_boxes,overlap_boxes = self.split_overlap_boxes(masked_boxes,ii,jj,height,width,mask_i,offset=0.05)

                    if self.match_class: # add class offets
                        gt_boxes = tbox + labels[:, 4:5] * 4096
                        nonoverlap_boxes = nonoverlap_boxes[:,:4] + nonoverlap_boxes[:, 5:6] * 4096
                        overlap_boxes = overlap_boxes[:,:4] + overlap_boxes[:, 5:6] * 4096
                    else:
                        gt_boxes = tbox 
                        nonoverlap_boxes = nonoverlap_boxes[:,:4] 
                        overlap_boxes = overlap_boxes[:,:4] 

                    # use nonoverlapping boxes for certification (it can certify iou and ioa robustness)
                    #if iou:
                    nonoverlap_flag_iou,nonoverlap_flag_ioa = self.iou_ioa_certify_flg(gt_boxes, nonoverlap_boxes, certify_iou_thres=certify_iou_thres,certify_ioa_thres=certify_ioa_thres)
                    # use overlapping boxes for certification (it can only certify ioa robustness)
                    if ioa:
                        overlap_flag_ioa = self.ioa_certify_flg(gt_boxes, overlap_boxes,certify_ioa_thres=certify_ioa_thres)
                    
                    #update each robustness map based on the certification results at this particular mask location
                    if ioa:
                        self.update_map(robust_bitmap_ioa,torch.logical_or(nonoverlap_flag_ioa,overlap_flag_ioa),ii,jj,mask_i)
                    if iou:
                        self.update_map(robust_bitmap_iou,nonoverlap_flag_iou,ii,jj,mask_i)

            # with robust_bitmap_ioa and robust_bitmap_iou generated, now we can determine the certified recall

            offset = 0.1 # the offset to distinguish between a close-patch and far-patch
            ii_offset = height*offset
            jj_offset = width*offset
            if ioa:
                robust_bitmap_ioa = torch.nn.functional.pad(robust_bitmap_ioa, pad=(patch_size_jj,patch_size_jj,patch_size_ii,patch_size_ii), mode='constant', value=1.0)
            if iou:
                robust_bitmap_iou = torch.nn.functional.pad(robust_bitmap_iou, pad=(patch_size_jj,patch_size_jj,patch_size_ii,patch_size_ii), mode='constant', value=1.0)

            # use a sliding window (i.e., a patch) over the -robust_bitmap_iou. 
            # The output is zero if at least one of the pixel location within the sliding windowhas zero
            # note the max pooling with a stride of 1 is memory-consuming
            if iou:
                vulnerable_map_iou = torch.nn.functional.max_pool2d(-robust_bitmap_iou,kernel_size=(patch_size_ii,patch_size_jj), stride=1,padding = 0)
                vulnerable_map_iou = torch.sum(vulnerable_map_iou,dim=1)==0 # the defense is vulnerable if all four mask robustness map give an output of zero
            if ioa:
                vulnerable_map_ioa = torch.nn.functional.max_pool2d(-robust_bitmap_ioa,kernel_size=(patch_size_ii,patch_size_jj), stride=1,padding = 0)
                vulnerable_map_ioa = torch.sum(vulnerable_map_ioa,dim=1)==0

            #location map. valid patch locations are ones; others are zeros
            if iou:
                location_map = torch.ones_like(vulnerable_map_iou,dtype=torch.bool).to(self.device)
            if ioa:
                location_map = torch.ones_like(vulnerable_map_ioa,dtype=torch.bool).to(self.device)


            # far patch
            for box_i,box in enumerate(tbox):
                box = box.type(torch.int)
                #a,b = int(max(0,box[1]-patch_size-ii_offset)),int(min(box[3]+1 + ii_offset,location_map.shape[1]))
                #c,d = int(max(0,box[0]-patch_size-jj_offset)),int(min(box[2]+1 + jj_offset,location_map.shape[2]))
                a,b = int(max(0,box[1]-ii_offset)),int(min(box[3]+1 + ii_offset + patch_size_ii,location_map.shape[1]))
                c,d = int(max(0,box[0]-jj_offset)),int(min(box[2]+1 + jj_offset + patch_size_jj ,location_map.shape[2]))
                location_map[box_i,a:b,c:d] = False
            
            # use logical_and. our defense is vulnerable when there is a location such that, 1) the location is vulnerable 2) the location is a valid one
            if iou:
                vul_count_iou = torch.any(torch.any(torch.logical_and(location_map,vulnerable_map_iou),dim=-1),dim=-1)
                vul_count_iou = torch.sum(vul_count_iou)
                far_vul_cnt_iou+=vul_count_iou.item()
            if ioa:
            
                vul_count_ioa = torch.any(torch.any(torch.logical_and(location_map,vulnerable_map_ioa),dim=-1),dim=-1)
                vul_count_ioa = torch.sum(vul_count_ioa)
                far_vul_cnt+=vul_count_ioa.item()
                
                # close patch
                for box_i,box in enumerate(tbox):
                    box = box.type(torch.int)
                    #a,b = max(0,box[1]-patch_size),min(box[3]+1,location_map.shape[1])
                    #c,d = max(0,box[0]-patch_size),min(box[2]+1,location_map.shape[2])
                    a,b = max(0,box[1]),min(box[3]+1+patch_size_ii,location_map.shape[1])
                    c,d = max(0,box[0]),min(box[2]+1+patch_size_jj,location_map.shape[2])
                    location_map[box_i,a:b,c:d] = True 
                location_map = ~location_map
                vul_count_ioa = torch.any(torch.any(torch.logical_and(location_map,vulnerable_map_ioa),dim=-1),dim=-1)
                vul_count_ioa = torch.sum(vul_count_ioa)
                close_vul_cnt+=vul_count_ioa.item()

                # over patch
                location_map = torch.zeros_like(location_map,dtype=torch.bool).to(self.device)
                for box_i,box in enumerate(tbox):
                    box = box.type(torch.int)
                    #a,b = max(0,box[1]-patch_size),min(box[3]+1,location_map.shape[1])
                    #c,d = max(0,box[0]-patch_size),min(box[2]+1,location_map.shape[2])
                    a,b = max(0,box[1]),min(box[3]+1+patch_size_ii,location_map.shape[1])
                    c,d = max(0,box[0]),min(box[2]+1+patch_size_jj,location_map.shape[2])
                    location_map[box_i,a:b,c:d] = True 
                vul_count_ioa = torch.any(torch.any(torch.logical_and(location_map,vulnerable_map_ioa),dim=-1),dim=-1)
                vul_count_ioa = torch.sum(vul_count_ioa)
                over_vul_cnt+=vul_count_ioa.item()
            
        return far_vul_cnt_iou,far_vul_cnt,close_vul_cnt,over_vul_cnt,obj_cnt

    def iou_ioa_certify_flg(self, box1, box2, certify_iou_thres,certify_ioa_thres):
        # the nonoverlapping boxes can be used for ioa and iou certification
        def box_area(box):
            # box = 4xn
            return (box[2] - box[0]) * (box[3] - box[1])

        area1 = box_area(box1.T)
        area2 = box_area(box2.T)
        # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
        inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)

        union = (area1[:, None] + area2 - inter) 

        area2_ind = area2 - inter

        tmp = self.iou_prune_thres * inter - (1-self.iou_prune_thres) * area2_ind
        flag1 = tmp >0
        flag2 = tmp / union > certify_iou_thres
        flag = torch.logical_and(flag1,flag2)
        flag_iou = torch.any(flag,dim=1)


        flag2 = (tmp / area1[:, None]) > certify_ioa_thres
        flag = torch.logical_and(flag1,flag2)
        flag_ioa = torch.any(flag,dim=1)        

        return flag_iou,flag_ioa
