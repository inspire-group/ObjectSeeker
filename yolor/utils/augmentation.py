import random

from yolor.utils.general import clip_coords

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

# takes  n1 x 4, n2 x 4 sets of boxes and returns an n1 x n2 array of pair-wise intersection-over-area scores
def IoA(a, b):
    a = a[:,:,None]
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        x_overlap = np.maximum(0, np.minimum(a[:,2], b[:,2])- np.maximum(a[:,0], b[:,0]))
        y_overlap = np.maximum(0, np.minimum(a[:,3], b[:,3])- np.maximum(a[:,1], b[:,1]))
        return x_overlap*y_overlap/(np.abs(a[:,2] - a[:,0])*np.abs(a[:,3] - a[:,1]))
    elif isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        z = torch.tensor(0, device=a.device)
        x_overlap = torch.maximum(z, torch.minimum(a[:,2], b[:,2])- torch.maximum(a[:,0], b[:,0]))
        y_overlap = torch.maximum(z, torch.minimum(a[:,3], b[:,3])- torch.maximum(a[:,1], b[:,1]))
        return x_overlap*y_overlap/(torch.abs(a[:,2] - a[:,0])*torch.abs(a[:,3] - a[:,1]))
    else:
        raise Exception('both inputs must be of type np.ndarray or torch.Tensor, simultaneously')
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
def mask_augment(imgs, targets):
    B, _, H, W = imgs.size()
        
    targets[:,-4:] = xywhn2xyxy(targets[:,-4:], w=W, h=H)

    retain_mask = torch.ones((len(targets)),dtype=torch.bool,device=targets.device)
    for i in range(B):
        if random.randint(1,10) > 3:
            r = random.randint(0, 3)
            ratio = 0.2
            x = random.randint(int(W*ratio), int(W*(1-ratio)))
            y = random.randint(int(H*ratio), int(H*(1-ratio)))

            if r == 0: 
                crop = torch.tensor([0, 0, x, H-1])
            elif r == 1:
                crop = torch.tensor([x, 0, W-1, H-1])
            elif r == 2:
                crop = torch.tensor([0, 0, W-1, y])
            elif r == 3:
                crop = torch.tensor([0, y, W-1, H-1])
            crop = crop.to(targets.device)

            # calculate which boxes are still visible within the crop
            intersections = torch.squeeze(box_ioa(targets[targets[:,0] == i, -4:],crop.unsqueeze(0)),dim=1)
            retain_mask[targets[:,0] == i] = intersections > 0.2

            # mask image
            mask = torch.zeros((1,1,H,W), device=imgs.device)
            x1, y1, x2, y2 = crop
            mask[...,y1:y2, x1:x2] = 1
            imgs[i] =mask* imgs[i]

            # crop boxes
            diff = torch.cat([crop[:2], crop[:2]])
            crop_boxes = torch.sub(targets[targets[:,0] == i, -4:], diff)
            clip_coords(crop_boxes, (crop[3]-crop[1], crop[2]-crop[0]))
            crop_boxes = torch.add(crop_boxes, diff)
            targets[targets[:,0] == i, -4:] = crop_boxes
        #else:


    targets[:,-4:] = xyxy2xywhn(targets[:,-4:], w=W, h=H)
    #return imgs, targets[torch.cat(retain_mask)]
    return imgs, targets[retain_mask]
