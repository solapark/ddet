import numpy as np
import copy
import json
import os
import matplotlib.pyplot as plt
import datetime

class Data_to_monitor :
    def __init__(self, name, names) :
        self.name = name
        self.names = names
        self.num_data = len(names)
        self.reset()

    def add(self, data):
        self.data = np.concatenate([self.data, np.array(data).reshape(-1, self.num_data)])

    def mean(self):
        return np.nanmean(self.data, axis=0)

    def reset(self):
        self.data = np.zeros((0, self.num_data))

    def get_name(self):
        return self.names

    def get_best(self):
        best_idx = np.argmax(self.data)
        best = self.data[best_idx]
        return best_idx[0], best

    def get_length(self):
        return len(self.data)

    def get_data(self):
        return self.data

    def load(self, path):
        self.data = np.load(path).reshape((-1, self.num_data))

    def save(self, path):
        np.save(path, self.data)

    def plot(self, path):
        epoch = len(self.data)
        axis = np.linspace(1, epoch, epoch)
        fig = plt.figure()
        plt.title(self.name)
        for n, d in zip(self.names, self.data.T) :
            plt.plot(axis, d, label=n)
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel(self.name)
        plt.grid(True)
        plt.savefig(path)
        plt.close(fig)

    def display(self):
        log = ['%s: %.4f'%(n, v) for n, v in zip(self.names, self.mean())]
        return ' '.join(log)
        

class Log_manager:
    def __init__(self, output_dir, class_list):
        self.dir = output_dir
        os.makedirs(self.dir, exist_ok=True)

        self.log_file = self.get_log_file()

        self.ap_names = class_list
        self.ap = Data_to_monitor('ap', self.ap_names)
        self.map = Data_to_monitor('map', ['map'])
        self.iou = Data_to_monitor('iou', ['iou'])

        self.best_map = 0
        self.best_map_epoch = 0

        '''
        if args.resume:
            self.ap.load(self.get_path('ap.npy'))
            self.map.load(self.get_path('map.npy'))
            self.iou.load(self.get_path('iou.npy'))
            
            self.best_map_epoch, self.best_map = slef.map.get_best()
        '''

    def get_path(self, *subdir):
        return os.path.join(self.dir, *subdir)

    def get_log_file(self):
        #self.log_file_name = 'log_%s.txt' % (self.args.mode)
        self.log_file_name = 'log.txt'
        open_type = 'a' if os.path.exists(self.get_path(self.log_file_name))else 'w'
        log_file = open(self.get_path(self.log_file_name), open_type)
        return log_file

    def save(self):
        self.ap.save(self.get_path('ap'))
        self.map.save(self.get_path('map'))
        self.iou.save(self.get_path('iou'))
        
        self.ap.plot(self.get_path('ap.pdf'))
        self.map.plot(self.get_path('map.pdf'))
        self.iou.plot(self.get_path('iou.pdf'))

    def add(self, data, name):
        if name == 'ap':
            self.ap.add(data)
            mAP = sum(data)/len(data)
            if(mAP > self.best_map):
                self.best_map = mAP
                self.best_map_epoch = self.map.get_length() + 1
            self.map.add(mAP)
        
        elif name == 'iou' :
            self.iou.add(data)

    def epoch_done(self):
        self.loss_every_epoch.add(self.loss_every_iter.mean())
        self.num_calssifier_pos_samples_every_epoch.add(self.num_calssifier_pos_samples_every_iter.mean())

        self.loss_every_iter.reset()
        self.num_calssifier_pos_samples_every_iter.reset()
        self.save()

    def write_log(self, log, refresh=True):
        print(log)
        self.log_file.write(log + '\n')
        if refresh:
            self.log_file.close()
            self.log_file = open(self.get_path(self.log_file_name), 'a')

    def write_cur_time(self):
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        self.write_log(now)

    def done(self):
        self.log_file.close()

    def plot(self, data, label):
        epoch = len(data)
        axis = np.linspace(1, epoch, epoch)
        fig = plt.figure()
        plt.title(label)
        if(label == 'ap'):
            for ap_name, ap in zip(self.ap_names, self.ap.T) :
                plt.plot( axis, ap, label=ap_name)
        else :
            plt.plot( axis, data, label=label)
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel(label)
        plt.grid(True)
        plt.savefig(self.get_path('%s.pdf')%(label))
        plt.close(fig)

    def get_best_map(self):
        return self.best_map

    def get_best_map_idx(self):
        return self.best_map_idx



class Map_calculator:
    def __init__(self, num_views, class_list, eval_rpn_only=False, cls_thresh=0):
        self.num_valid_cam = num_views
        self.class_list_wo_bg = class_list
        self.reset()

        self.min_overlap = 0.5
        self.eval_rpn_only = eval_rpn_only
        self.cls_thresh = cls_thresh
        self.metric = ['MODA', 'MODP', 'F1', 'Recall', 'Precision']

    def reset(self):
        self.TP = {cls : [] for cls in self.class_list_wo_bg}
        self.FP = {cls : [] for cls in self.class_list_wo_bg}

        self.prob = {cls : [] for cls in self.class_list_wo_bg}

        self.iou = {cls : [] for cls in self.class_list_wo_bg}

        self.gt_counter_per_class = {cls : 0 for cls in self.class_list_wo_bg}

        self.iou_result = 0
        self.cnt = 0

    '''
    def get_gt_batch(self, labels_batch):
        return [self.get_gt(labels) for labels in labels_batch]

    def get_gt(self, labels):
        gts = [[] for _ in range(self.num_valid_cam)]
        for inst in labels :
            gt_cls = self.class_list_wo_bg[inst['cls']]
            gt_boxes = inst['resized_box']
            for cam_idx in range(self.num_valid_cam) : 
                if cam_idx in gt_boxes :
                    x1, y1, x2, y2 = gt_boxes[cam_idx]
                    info = {'class':gt_cls, 'x1':x1, 'y1':y1, 'x2':x2, 'y2':y2}
                    gts[cam_idx].append(info)
        return gts
    '''

    def get_iou(self):
        return self.iou_result/self.cnt

    def get_map(self) :
        return np.mean(np.array(self.all_aps))

    def get_aps(self):
        all_aps = [average_precision_score(t, p) if len(t) else 0 for t, p in zip(self.all_T.values(), self.all_P.values())]
        #all_aps = [ap if not math.isnan(ap) else 0 for ap in all_aps]
        all_aps = [0 if ap == 1.0 else ap for ap in all_aps]
        all_aps = [0 if math.isnan(ap) else ap for ap in all_aps]
        self.all_aps = all_aps
        return all_aps

    def get_aps_dict(self):
        return {cls : ap for cls, ap in zip(self.class_list_wo_bg, self.all_aps)}

    def add_img_tp(self, dets, gts):
        self.cnt += 1
        T, P, iou = self.get_img_tp(dets, gts)
        self.iou_result += iou
        for key in T.keys():
            self.all_T[key].extend(T[key])
            self.all_P[key].extend(P[key])

    def get_img_tp(self, pred, gt):
        T = {}
        P = {}
        iou_result = 0

        for bbox in gt:
            bbox['bbox_matched'] = False

        pred_probs = np.array([s['prob'] for s in pred])
        #print(pred)
        #print(pred_probs)
        box_idx_sorted_by_prob = np.argsort(pred_probs)[::-1]

        for box_idx in box_idx_sorted_by_prob:
            pred_box = pred[box_idx]
            pred_class = pred_box['class']
            pred_x1 = pred_box['x1']
            pred_x2 = pred_box['x2']
            pred_y1 = pred_box['y1']
            pred_y2 = pred_box['y2']
            pred_prob = pred_box['prob']
            if pred_class not in P:
                P[pred_class] = []
                T[pred_class] = []
            P[pred_class].append(pred_prob)
            found_match = False

            for gt_box in gt:
                gt_class = gt_box['class']
                gt_x1 = gt_box['x1']
                gt_x2 = gt_box['x2']
                gt_y1 = gt_box['y1']
                gt_y2 = gt_box['y2']
                gt_seen = gt_box['bbox_matched']
                if gt_class != pred_class:
                    continue
                if gt_seen:
                    continue
                iou = 0
                iou = data_generators.iou((pred_x1, pred_y1, pred_x2, pred_y2), (gt_x1, gt_y1, gt_x2, gt_y2))
                iou_result += iou
                #print('IoU = ' + str(iou))
                if iou >= 0.5:
                    found_match = True
                    gt_box['bbox_matched'] = True
                    break
                else:
                    continue

            T[pred_class].append(int(found_match))
        for gt_box in gt:
            if not gt_box['bbox_matched']: # and not gt_box['difficult']:
                if gt_box['class'] not in P:
                    P[gt_box['class']] = []
                    T[gt_box['class']] = []

                T[gt_box['class']].append(1)
                P[gt_box['class']].append(0)

        #import pdb
        #pdb.set_trace()
        return T, P, iou_result

    def get_x1y1x2y2(self, box):
        x1 = box['x1']
        x2 = box['x2']
        y1 = box['y1']
        y2 = box['y2']
        return x1, y1, x2, y2

    def get_dr_data(self, pred):
        dr_data = {cls : [] for cls in self.class_list_wo_bg}
        pred_probs = np.array([s['prob'] for s in pred])
        box_idx_sorted_by_prob = np.argsort(pred_probs)[::-1]

        for box_idx in box_idx_sorted_by_prob:
            pred_box = pred[box_idx]
            #cls = pred_box['class']
            cls = pred_box['class'] if not self.eval_rpn_only else 'water1'
            prob = pred_box['prob']
            bbox = self.get_x1y1x2y2(pred_box)
            dr_data[cls].append({"confidence":prob, "bbox":bbox})
        
        return dr_data

    def get_ground_truth_data(self, gt):
        ground_truth_data = {cls : [] for cls in self.class_list_wo_bg}

        for gt_box in gt:
            #cls = gt_box['class']
            cls = gt_box['class'] if not self.eval_rpn_only else 'water1'
            bbox = self.get_x1y1x2y2(gt_box)
            bbox = list(map(round, bbox))
            ground_truth_data[cls].append({"bbox":bbox, "used":False})
            self.gt_counter_per_class[cls] += 1
        
        return ground_truth_data

    def thresholding(self, dets) :
        result_det = dict()
        for cls_name, cur_cls_det in dets.items():
            result_det[cls_name] = []
            for det in cur_cls_det : 
                if det['confidence'] > self.cls_thresh :
                    result_det[cls_name].append(det)
                else :
                    pass
        return result_det

    def add_tp_fp(self, pred, gt):
        dr_data_dict = self.get_dr_data(pred)
        ground_truth_data = self.get_ground_truth_data(gt)

        dr_data_dict = self.thresholding(dr_data_dict)

        for class_name in dr_data_dict.keys():
            dr_data = dr_data_dict[class_name]
            nd = len(dr_data)
            tp = [0] * nd 
            fp = [0] * nd
            prob = [0] * nd
            iou = [0] * nd

            for idx, detection in enumerate(dr_data):
                ovmax = -1
                gt_match = -1
                # load detected object bounding-box
                bb = detection["bbox"]
                prob[idx] = detection["confidence"]
                for obj in ground_truth_data[class_name]:
                    bbgt = obj["bbox"]
                    bi = [max(bb[0],bbgt[0]), max(bb[1],bbgt[1]), min(bb[2],bbgt[2]), min(bb[3],bbgt[3])]
                    iw = bi[2] - bi[0] + 1
                    ih = bi[3] - bi[1] + 1
                    if iw > 0 and ih > 0:
                        # compute overlap (IoU) = area of intersection / area of union
                        ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (bbgt[2] - bbgt[0] + 1) * (bbgt[3] - bbgt[1] + 1) - iw * ih
                        ov = iw * ih / ua
                        if ov > ovmax:
                            ovmax = ov
                            gt_match = obj

                # assign detection as true positive/don't care/false positive
                if ovmax >= self.min_overlap:
                    if not bool(gt_match["used"]):
                        # true positive
                        tp[idx] = 1
                        gt_match["used"] = True
                        iou[idx] = ovmax
                        #count_true_positives[class_name] += 1
                    else:
                        # false positive (multiple detection)
                        fp[idx] = 1
                else:
                    # false positive
                    fp[idx] = 1
                    #if ovmax > 0:
                    #    status = "INSUFFICIENT OVERLAP"

            self.prob[class_name].extend(prob)
            self.TP[class_name].extend(tp)
            self.FP[class_name].extend(fp)
            self.iou[class_name].extend(iou)

    def sort_tp_fp(self, prob, tp, fp):
        whole = np.column_stack([np.array(prob), np.array(tp), np.array(fp)])
        whole = whole[(-whole[:, 0]).argsort()]
        prob, tp, fp = whole.T
        return prob.tolist(), tp.tolist(), fp.tolist()

    def get_recall_precision(self, tp, fp, gt_counter_per_class):
        tp = copy.deepcopy(tp)
        fp = copy.deepcopy(fp)
        # compute precision/recall
        cumsum = 0
        for idx, val in enumerate(fp):
            fp[idx] += cumsum
            cumsum += val
        cumsum = 0
        for idx, val in enumerate(tp):
            tp[idx] += cumsum
            cumsum += val
        #print(tp)

        rec = tp[:]
        for idx, val in enumerate(tp):
            #rec[idx] = float(tp[idx]) / gt_counter_per_class[class_name]
            rec[idx] = float(tp[idx]) / gt_counter_per_class
        #print(rec)
        prec = tp[:]
        for idx, val in enumerate(tp):
            prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])
        #print(prec)
        
        return rec, prec

    def voc_ap(self, rec, prec):
        """
        --- Official matlab code VOC2012---
        mrec=[0 ; rec ; 1];
        mpre=[0 ; prec ; 0];
        for i=numel(mpre)-1:-1:1
                mpre(i)=max(mpre(i),mpre(i+1));
        end
        i=find(mrec(2:end)~=mrec(1:end-1))+1;
        ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
        """
        rec.insert(0, 0.0) # insert 0.0 at begining of list
        rec.append(1.0) # insert 1.0 at end of list
        mrec = rec[:]
        prec.insert(0, 0.0) # insert 0.0 at begining of list
        prec.append(0.0) # insert 0.0 at end of list
        mpre = prec[:]
        """
         This part makes the precision monotonically decreasing
            (goes from the end to the beginning)
            matlab: for i=numel(mpre)-1:-1:1
                        mpre(i)=max(mpre(i),mpre(i+1));
        """
        # matlab indexes start in 1 but python in 0, so I have to do:
        #     range(start=(len(mpre) - 2), end=0, step=-1)
        # also the python function range excludes the end, resulting in:
        #     range(start=(len(mpre) - 2), end=-1, step=-1)
        for i in range(len(mpre)-2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i+1])
        """
         This part creates a list of indexes where the recall changes
            matlab: i=find(mrec(2:end)~=mrec(1:end-1))+1;
        """
        i_list = []
        for i in range(1, len(mrec)):
            if mrec[i] != mrec[i-1]:
                i_list.append(i) # if it was matlab would be i + 1
        """
         The Average Precision (AP) is the area under the curve
            (numerical integration)
            matlab: ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
        """
        ap = 0.0
        for i in i_list:
            ap += ((mrec[i]-mrec[i-1])*mpre[i])
        return ap, mrec, mpre

    def get_aps_dict(self):
        return self.all_aps_dict

    def get_eval(self):
        self.all_moda_dict = {class_name : -1 for class_name in self.class_list_wo_bg}
        self.all_modp_dict = {class_name : -1 for class_name in self.class_list_wo_bg}
        self.all_f1_dict = {class_name : -1 for class_name in self.class_list_wo_bg}
        self.all_recall_dict = {class_name : -1 for class_name in self.class_list_wo_bg}
        self.all_precision_dict = {class_name : -1 for class_name in self.class_list_wo_bg}

        for class_name in self.class_list_wo_bg :
            self.prob[class_name], self.TP[class_name], self.FP[class_name] = self.sort_tp_fp(self.prob[class_name], self.TP[class_name], self.FP[class_name])
            gt_counter_per_class =  self.gt_counter_per_class[class_name]

            if gt_counter_per_class : 
                num_tp = np.array(self.TP[class_name]).sum()
                num_fp = np.array(self.FP[class_name]).sum()
                num_fn = gt_counter_per_class - num_tp

                tp_iou_sum = np.array(self.iou[class_name]).sum()

                rec = num_tp / gt_counter_per_class
                prec = num_tp / (num_fp + num_tp)
                
                f1 = 2 * (rec * prec) / (rec + prec)

                moda = 1 - (num_fn + num_fp) / gt_counter_per_class
                modp = tp_iou_sum / num_tp

                self.all_moda_dict[class_name] = moda if moda > 0 else 0
                self.all_modp_dict[class_name] = modp if modp > 0 else 0
                self.all_f1_dict[class_name] = f1 if f1 > 0 else 0
                self.all_recall_dict[class_name] = rec if rec > 0 else 0
                self.all_precision_dict[class_name] = prec if prec > 0 else 0

        self.all_moda = list(self.all_moda_dict.values())
        self.all_modp = list(self.all_modp_dict.values())
        self.all_f1 = list(self.all_f1_dict.values())
        self.all_recall = list(self.all_recall_dict.values())
        self.all_precision = list(self.all_precision_dict.values())

        return self.all_moda_dict, self.all_modp_dict, self.all_f1_dict, self.all_recall_dict, self.all_precision_dict

    def get_aps(self):
        self.all_aps_dict = {}
        for class_name in self.class_list_wo_bg :
            self.prob[class_name], self.TP[class_name], self.FP[class_name] = self.sort_tp_fp(self.prob[class_name], self.TP[class_name], self.FP[class_name])
            gt_counter_per_class =  self.gt_counter_per_class[class_name]
            if gt_counter_per_class : 
            
                recall, precision = self.get_recall_precision(self.TP[class_name], self.FP[class_name], gt_counter_per_class)

                self.all_aps_dict[class_name], _, _ = self.voc_ap(recall, precision)
            else :
                self.all_aps_dict[class_name] = -1
                
        self.all_aps = list(self.all_aps_dict.values())
        return self.all_aps

    def get_valid_mean(self, metric):
        metric = np.array(metric)
        #metric = metric[metric >= 0]
        return np.mean(metric)

    def get_mean_eval(self) :
        return list(map(self.get_valid_mean, [self.all_moda, self.all_modp, self.all_f1, self.all_recall, self.all_precision]))

    def get_map(self) :
        return self.get_valid_mean(self.all_aps)

    '''
    def get_map(self) :
        self.all_aps = np.array(self.all_aps)
        valid_aps = self.all_aps[self.all_aps >= 0]
        return np.mean(valid_aps)
    '''

class DataLoader:
    def __init__(self, num_views, visible_thresh=0, reid_thresh=0):
        self.num_valid_cam = num_views
        self.visible_thresh = visible_thresh
        self.reid_thresh = reid_thresh

    def get_gt(self, all_bboxes, all_is_valids, all_cls):
        num_samples = len(all_bboxes)
        data = [[] for _ in range(self.num_valid_cam)]
        for sample_idx in range(num_samples) :
            cls = all_cls[sample_idx]
            for cam_idx in range(self.num_valid_cam) :
                if all_is_valids[sample_idx, cam_idx] :
                    cx, cy, w, h = all_bboxes[sample_idx][cam_idx]
                    x1, y1, x2, y2 = (cx - w/2), (cy - h/2), (cx + w/2), (cy + h/2)
                    info = {'class':cls, 'x1':x1, 'y1':y1, 'x2':x2, 'y2':y2}
                    data[cam_idx].append(info)
        return data 

    def get_det(self, all_bboxes, all_is_valids, all_cls, all_score, all_reid_score=[]):
        num_samples = len(all_bboxes)
        data = [[] for _ in range(self.num_valid_cam)]
        for sample_idx in range(num_samples) :
            cls = all_cls[sample_idx]
            score = all_score[sample_idx]
            if self.reid_thresh :
                reid_score = all_reid_score[sample_idx]
                if reid_score < self.reid_thresh :
                    continue

            for cam_idx in range(self.num_valid_cam) :
                is_valid = all_is_valids[sample_idx, cam_idx]
                if is_valid >= self.visible_thresh :
                    cx, cy, w, h = all_bboxes[sample_idx][cam_idx]
                    x1, y1, x2, y2 = (cx - w/2), (cy - h/2), (cx + w/2), (cy + h/2)
                    info = {'class':cls, 'x1':x1, 'y1':y1, 'x2':x2, 'y2':y2, 'prob':score, 'is_valid':is_valid}
                    data[cam_idx].append(info)
        return data 

class MessytableEval:
    def __init__(self, num_views, class_list, det_path, gt, output_dir, min_overlap=.5, cls_thresh=0, visible_thresh=0, reid_thresh=0, save_query=False):
        self.num_valid_cam = num_views
        self.save_query = save_query

        self.gt = gt
        with open(det_path) as f:
            self.det = json.load(f)

        self.visible_thresh = visible_thresh
        self.reid_thresh = reid_thresh
        self.DataLoader = DataLoader(num_views, visible_thresh, reid_thresh)
        self.Map_calculator = Map_calculator(num_views, class_list, cls_thresh=cls_thresh)
        self.Log_manager = Log_manager(output_dir, class_list)

    def main(self) :
        self.Log_manager.write_cur_time()
        result_list = []
        for gt in self.gt:
            scene_id = gt['scene_id']
            
            gt_bboxes = gt['cam_instances'][:self.num_valid_cam].transpose(1, 0, 2) #(7, num_views, 4)
            gt_is_valids = gt['cam_instances_valid_flags'][:self.num_valid_cam].transpose(1, 0) #(7, num_views)
            gt_cls = np.array(gt['gt_names']) #(7, )

            det = self.det[scene_id] #(300, )
            num_det = len(det) # 300

            det_bboxes = []
            det_is_valids = []
            det_cls = [] 
            det_score = []
            det_reid_score = []
            det_query = []
            for inst in det :
                det_bboxes.append(inst['camera_instances'])
                det_is_valids.append(inst['camera_instances_valid_flag'])
                det_cls.append(inst['detection_name'])
                det_score.append(inst['detection_score'])
                if self.reid_thresh : 
                    det_reid_score.append(inst['reid_score'])
                if self.save_query : 
                    det_query.append(inst['query2d'])
            det_bboxes = np.array(det_bboxes).reshape((num_det, self.num_valid_cam, -1)) #(300, num_views, 4)
            det_is_valids = np.array(det_is_valids).reshape(num_det, self.num_valid_cam) #(300, num_views)
            det_cls = np.array(det_cls) #(300,)
            det_score = np.array(det_score) #(300,)
            if self.reid_thresh : 
                det_reid_score = np.array(det_reid_score) #(300,)

            if self.save_query : 
                det_query = np.array(det_query) #(300,)

            gt = self.DataLoader.get_gt(gt_bboxes, gt_is_valids, gt_cls)
            det = self.DataLoader.get_det(det_bboxes, det_is_valids, det_cls, det_score, det_reid_score)

            for cam_idx in range(self.num_valid_cam) :
                self.Map_calculator.add_tp_fp(det[cam_idx], gt[cam_idx])

            result = [scene_id, gt_bboxes.transpose(1, 0, 2), gt_is_valids.transpose(1, 0), gt_cls, det_bboxes.transpose(1, 0, 2), det_is_valids.transpose(1, 0), det_cls, det_score]

            if self.save_query :
                result.append(det_query.transpose(1, 0, 2))

            result_list.append(result)

        all_aps = self.Map_calculator.get_aps()
        #iou_avg = self.Map_calculator.get_iou()

        self.Log_manager.add(all_aps, 'ap')
        #self.Log_manager.add(iou_avg, 'iou')
        self.Log_manager.save()

        all_ap_dict = self.Map_calculator.get_aps_dict()
        cur_map = self.Map_calculator.get_map()

        return all_ap_dict, cur_map, result_list

    def log_eval(self) :
        eval = self.Map_calculator.get_eval()
        mean_eval = self.Map_calculator.get_mean_eval()
        metric = self.Map_calculator.metric
        valid_cls = list(eval[0].keys())
        '''
        for metric_name, ev, m_ev in zip(metric, eval, mean_eval) :
            log_manager.write_log('%s\t%.2f'%(metric_name, m_ev))
            for _, cls in enumerate(valid_cls):
                e = ev[cls]
                #if e<0: continue
                log_manager.write_log('%s\t%.2f'%(cls, e))
            log_manager.write_log('\n')
        '''

        metric_name = '\t'.join(metric)
        metric_value = ['%.2f'%(m) for m in mean_eval]
        metric_value = '\t'.join(metric_value)
        self.Log_manager.write_log('metric\t%s'%(metric_name))
        self.Log_manager.write_log('ALL\t%s'%(metric_value))

        for _, cls in enumerate(valid_cls):
            ev = [e[cls] for e in eval]
            ev = ['%.2f'%(e) for e in ev]
            ev = '\t'.join(ev)
            #if e<0: continue
            self.Log_manager.write_log('%s\t%s'%(cls, ev))
        self.Log_manager.write_log('\n')

