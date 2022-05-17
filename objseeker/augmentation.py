import random

from defense import clip_coords

import numpy as np
import torch

def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw  # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh  # top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw  # bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh  # bottom right y
    return y


def xyxy2xywhn(x, w=640, h=640, clip=False, eps=0.0):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] normalized where xy1=top-left, xy2=bottom-right
    if clip:
        clip_coords(x, (h - eps, w - eps))  # warning: inplace clip
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = ((x[:, 0] + x[:, 2]) / 2) / w  # x center
    y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) / h  # y center
    y[:, 2] = (x[:, 2] - x[:, 0]) / w  # width
    y[:, 3] = (x[:, 3] - x[:, 1]) / h  # height
    return y

def box_ioa(box1, box2):

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    #area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None])  


#random.seed(12342343456)
def mask_augment(imgs, targets, ioa_thres=0.5):
    num_imgs, _, H, W = imgs.size()

    # copy images 4 times for each mask
    imgs = imgs.repeat((4,1,1,1))
    
    # copy targets 4 times for each mask, relabel img indices
    num_targets = targets.size(0)
    targets = targets.repeat((4, 1))
    for i in range(4):
        targets[i*num_targets:(i+1)*num_targets,0] += i * num_imgs
    targets[:,-4:] = xywhn2xyxy(targets[:,-4:], w=W, h=H)
    targets_crop = targets.detach().clone()

    retain_mask = torch.ones((len(targets)),dtype=torch.bool,device=targets.device)

    # for each original image
    for i in range(num_imgs):
        # select a random object
        img_objs = targets[targets[:,0] == i]
        if img_objs.size(0) == 0: continue
        obj = img_objs[random.randint(0, img_objs.size(0)-1), -4:]

        # generate random partioning point in the object
        ratio = 0.15
        obj_w = obj[2] - obj[0]
        obj_h = obj[3] - obj[1]
        x = random.randint(int(obj_w*ratio), int(obj_w*(1-ratio))) + int(round(float(obj[0])))
        y = random.randint(int(obj_h*ratio), int(obj_h*(1-ratio))) + int(round(float(obj[1])))
        crops = torch.tensor([[0,0,x,H], [x,0,W,H], [0,0,W,y], [0,y,W,H]], device=targets.device, dtype=int)

        # keep a vanilla image 
        keep_idx = random.randint(0, 3)
        
        # for each mask
        for j, crop in enumerate(crops):
            if j == keep_idx:
                crop = torch.tensor([0, 0, W, H], device=targets.device) # keep the whole image
            
            mask = torch.zeros((1,1,H,W), device=imgs.device)
            x1, y1, x2, y2 = crop
            mask[...,y1:y2, x1:x2] = 1

            intersections = torch.squeeze(box_ioa(targets[targets[:,0] == i + j*num_imgs, -4:], crop.unsqueeze(0)), dim=1)
            retain_mask[targets[:,0] == i + j*num_imgs] = intersections > ioa_thres
            
            imgs[i + j*num_imgs] = mask * imgs[i + j*num_imgs]

            # crop the ground truth objects
            diff = torch.cat([crop[:2], crop[:2]])
            crop_boxes = torch.sub(targets_crop[targets_crop[:,0] == i + j*num_imgs, -4:], diff)
            clip_coords(crop_boxes, (crop[3]-crop[1], crop[2]-crop[0]))
            crop_boxes = torch.add(crop_boxes, diff)
            targets_crop[targets_crop[:,0] == i + j*num_imgs, -4:] = crop_boxes

    targets[:,-4:] = xyxy2xywhn(targets[:,-4:], w=W, h=H)
    targets_crop[:,-4:] = xyxy2xywhn(targets_crop[:,-4:], w=W, h=H)
    
    return imgs, targets[retain_mask], targets_crop[retain_mask]
