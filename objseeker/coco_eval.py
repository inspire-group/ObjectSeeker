
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import time
import numpy as np 

class COCOeval_OS(COCOeval): # coco evaluation class for objectseeker

    def accumulate_rec_prec(self, p = None):
        # modified from https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py#L315
        # only compute AP@0.5
        '''
        Accumulate per image evaluation results and store the result in self.eval
        :param p: input params for evaluation
        :return: None
        '''
        print('Accumulating evaluation results...')
        tic = time.time()
        if not self.evalImgs:
            print('Please run evaluate() first')
        # allows input customized parameters
        if p is None:
            p = self.params
        p.catIds = p.catIds if p.useCats == 1 else [-1]
        T           = len(p.iouThrs)
        R           = len(p.recThrs)
        K           = len(p.catIds) if p.useCats else 1
        A           = len(p.areaRng)
        M           = len(p.maxDets)
        precision   = np.ones((T,K,A,M)) # -1 for the precision of absent categories
        recall      = np.zeros((T,K,A,M))
        #scores      = -np.ones((T,R,K,A,M))

        # create dictionary for future indexing
        _pe = self._paramsEval
        catIds = _pe.catIds if _pe.useCats else [-1]
        setK = set(catIds)
        setA = set(map(tuple, _pe.areaRng))
        setM = set(_pe.maxDets)
        setI = set(_pe.imgIds)
        # get inds to evaluate
        k_list = [n for n, k in enumerate(p.catIds)  if k in setK]
        m_list = [m for n, m in enumerate(p.maxDets) if m in setM]
        a_list = [n for n, a in enumerate(map(lambda x: tuple(x), p.areaRng)) if a in setA]
        i_list = [n for n, i in enumerate(p.imgIds)  if i in setI]
        I0 = len(_pe.imgIds)
        A0 = len(_pe.areaRng)
        # retrieve E at each category, area range, and max number of detections
        for k, k0 in enumerate(k_list):
            Nk = k0*A0*I0
            for a, a0 in enumerate(a_list):
                Na = a0*I0
                for m, maxDet in enumerate(m_list):
                    E = [self.evalImgs[Nk + Na + i] for i in i_list]
                    E = [e for e in E if not e is None]
                    if len(E) == 0:
                        continue
                    dtScores = np.concatenate([e['dtScores'][0:maxDet] for e in E])

                    # different sorting method generates slightly different results.
                    # mergesort is used to be consistent as Matlab implementation.
                    inds = np.argsort(-dtScores, kind='mergesort')
                    dtScoresSorted = dtScores[inds]

                    dtm  = np.concatenate([e['dtMatches'][:,0:maxDet] for e in E], axis=1)[:,inds]
                    dtIg = np.concatenate([e['dtIgnore'][:,0:maxDet]  for e in E], axis=1)[:,inds]
                    gtIg = np.concatenate([e['gtIgnore'] for e in E])
                    npig = np.count_nonzero(gtIg==0 )
                    if npig == 0:
                        continue
                    tps = np.logical_and(               dtm,  np.logical_not(dtIg) )
                    fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg) )

                    tp_sum = np.sum(tps, axis=1).astype(dtype=float)
                    fp_sum = np.sum(fps, axis=1).astype(dtype=float)
                    for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                        npred = fp+tp
                        rc = tp / npig
                        pr = tp / npred if npred>0 else 1
                        recall[t,k,a,m] = rc
                        precision[t,k,a,m] = pr
        return recall[0,:,0,-1],precision[0,:,0,-1]           




def compute_ap(recall, precision):

    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'continuous'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap#, mpre, mrec

if __name__ == "__main__":
    import os
    pred_json = '/home/cxiang/vanilla_detection_dump/yolor_coco/yolor_p6_coco_predictions.json'
    anno_json = os.path.join('..','data','coco','annotations','instances_val2017.json')

    anno = COCO(anno_json)  # init annotations api
    pred = anno.loadRes(pred_json)  # init predictions api
    cocoeval = COCOeval_DM(anno, pred, 'bbox')

    cocoeval.params.maxDets = [100]
    cocoeval.params.areaRng = [[0 ** 2, 1e5 ** 2]]
    cocoeval.params.areaRngLbl = ['all']
    cocoeval.params.iouThrs = [0.5]
    cocoeval.params.recThrs = [0.0]

    #if is_coco:
    #    eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.img_files]  # image IDs to evaluate
    cocoeval.evaluate()
    r,p = cocoeval.accumulate_rec_prec()

    print(r,p)