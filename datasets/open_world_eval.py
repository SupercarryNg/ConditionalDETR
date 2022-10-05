import numpy as np
import os
import json
import xml.etree.ElementTree as ET


class OWEvaluator:
    def __init__(self, voc_gt, iou_types):
        self.lines = None
        self.lines_cls = None
        self.voc_gt.CLASS_NAMES = None
        self.known_classes = self.voc_gt.CLASS_NAMES
        # imageset txt file
        self.voc_gt.image_set = None
        self.all_recs = None
        self.tp_plus_fp_cs = None
        self.fp_os = None

    def accumulate(self):
        for class_label_ind, class_label in enumerate(self.voc_gt.CLASS_NAMES):
            '''
                self.lines.append(f"{image_id} {score:.3f} {xmin:.1f} {ymin:.1f} {xmax:.1f} {ymax:.1f}")
                self.lines_cls.append(cls)
            '''
            lines_by_class = [l + '\n' for l, c in zip(self.lines, self.lines_cls.tolist()) if c == class_label_ind]
            if len(lines_by_class) == 0:
                lines_by_class = []
            print(class_label + " has " + str(len(lines_by_class)) + " predictions.")
            ovthresh = 50

            self.rec, self.tp_plus_fp_closed_set, self.fp_open_set = voc_eval(lines_by_class, self.voc_gt.annotations, \
                                                     self.voc_gt.image_set, class_label,ovthresh= .5,known_classes=self.known_classes)
            self.all_recs[ovthresh].append(self.rec)
            self.tp_plus_fp_cs[ovthresh].append(self.tp_plus_fp_closed_set)
            self.fp_os[ovthresh].append(self.fp_open_set)

'''
lines_by_class, self.voc_gt.annotations, self.voc_gt.image_set,\
class_label, ovthresh=ovthresh / 100.0,known_classes=self.known_classes
'''
# assumes detections are in detpath.format(classname)
# assumes annotations are in annopath.format(imagename)
# assumes imagesetfile is a text file with each line an image name
def parse_json(imagenames, root_dir='voc_data/annotation/test'):
    """
    :param root_dir: 'voc_data/annotation/test'
    :param imagenames: ['2007_0032.jpg', '2007_0033.jpg']
    :return: {'2007_0032.jpg': [{'name': 'person', 'difficult': 0, 'bbox': [234, 345, 245, 456]},
                                {'name': 'person', 'difficult': 0, 'bbox': [234, 345, 245, 456]}]}
    """
    recs = {}
    for imagename in imagenames:
        json_path = os.path.join(root_dir, imagename[:-4] + '.json')
        objects = []
        with open(json_path) as f:
            img_rec = json.load(f)

        for obj in img_rec.get('annotations'):
            objects.append({'name': obj.get('category_id'), 'bbox': obj.get('bbox'), 'difficult': obj.get('difficult')})

        recs[img_rec.get('images')['file_name']] = objects

    return recs

def iou(BBGT, bb):
    ixmin = np.maximum(BBGT[:, 0], bb[0])
    iymin = np.maximum(BBGT[:, 1], bb[1])
    ixmax = np.minimum(BBGT[:, 2], bb[2])
    iymax = np.minimum(BBGT[:, 3], bb[3])
    iw = np.maximum(ixmax - ixmin + 1., 0.)
    ih = np.maximum(iymax - iymin + 1., 0.)
    inters = iw * ih

    # union
    uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
           (BBGT[:, 2] - BBGT[:, 0] + 1.) *
           (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

    overlaps = inters / uni
    ovmax = np.max(overlaps)
    jmax = np.argmax(overlaps)
    return ovmax, jmax

def voc_eval(detpath,
             classname,
             imagesetfile,
             annopath = '../voc_data/annotation/test',
             ovthresh=0.5,
             known_classes=None):


    imagenames = ['2007_000033.jpg']
    # load  annotations
    '''
    recs['2007.jpg'] =
    [{'name':'person'(0),'difficult':0/1,'bbox':[23,33,345,233]}, {'name':'bird','difficult':0/1,'bbox':[23,33,345,233]},]
    '''
    recs = parse_json(imagenames, root_dir= annopath)

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        # 對於每一張圖片的所有object做循環，有classname的拿出來存在R中
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        # R = [{'name':'person'(0),'difficult':0/1,'bbox':[23,33,345,233]},...], all the obj are 'person'
        bbox = np.array([x['bbox'] for x in R])
        # bbox = [[23,33,345,233],[23,33,345,233],...]
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        # difficult = [0,0,1,0...]
        det = [False] * len(R)
        # det = [false,false,...]
        npos = npos + sum(~difficult)
        # npos = the number of objects in this class 'Person'
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}
    # class_recs['2007.jpg'] = {'bbox':,'difficult':,'det':}

    # read detections
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()

    # lines belong to the same class

    # get image_id, score, bounding box
    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    # ['2007.jpg','2008.jpg','2007.jpg'...]
    confidence = np.array([float(x[1]) for x in splitlines])
    # ['0.4','0.5',...]
    if len(splitlines) == 0:
        BB = np.array([[float(z) for z in x[2:]] for x in splitlines]).reshape(-1, 4)
    else:
        BB = np.array([[float(z) for z in x[2:]] for x in splitlines])#.reshape(-1, 4)


    # sort by confidence, 降序排列返回index
    sorted_ind = np.argsort(-confidence)
    # sorted index = [5,4,1,65,...]
    BB = BB[sorted_ind, :]

    # import pdb;pdb.set_trace()
    image_ids = [image_ids[x] for x in sorted_ind]
    # image_ids = ['2008.jpg','2007.jpg','2007.jpg',...],從大到小

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)

    for d in range(nd):
        R = class_recs[image_ids[d]]
        # R = class_recs['2008.jpg'] = {'bbox':[[23,33,345,233],[23,33,345,233],...],'difficult':,'det':} targts
        bb = BB[d, :].astype(float)
        # bb = [23,45,342,454] prediction
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)

        if BBGT.size > 0:
            ovmax, jmax = iou(BBGT, bb)
        # ovmax 是最大的iou值，jmax是所對應的index

        if ovmax > ovthresh:
            if not R['difficult'][jmax]:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult

    # Finding GT of unknown objects
    unknown_class_recs = {}
    n_unk = 0
    for imagename in imagenames:
        # 'unknown' -1
        R = [obj for obj in recs[imagename] if obj["name"] == 'unknown']
        bbox = np.array([x["bbox"] for x in R])
        difficult = np.array([x["difficult"] for x in R]).astype(np.bool)
        det = [False] * len(R)
        n_unk = n_unk + sum(~difficult)
        unknown_class_recs[imagename] = {"bbox": bbox, "difficult": difficult, "det": det}

    if classname == 'unknown':
        return rec, None, None, None

    # Go down each detection and see if it has an overlap with an unknown object.
    # If so, it is an unknown object that was classified as known.
    is_unk = np.zeros(nd)
    for d in range(nd):
        R = unknown_class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R["bbox"].astype(float)

        if BBGT.size > 0:
            ovmax, jmax = iou(BBGT, bb)

        if ovmax > ovthresh:
            is_unk[d] = 1.0

    is_unk_sum = np.sum(is_unk)
    # is_unk_sum 就是 A-OSE
    tp_plus_fp_closed_set = tp+fp
    fp_open_set = np.cumsum(is_unk)

    return rec, is_unk_sum, tp_plus_fp_closed_set, fp_open_set

if __name__ == '__main__':
    rec, is_unk_sum, tp_plus_fp_closed_set, fp_open_set = voc_eval('../predictions/pred_{}.txt',7,None)





