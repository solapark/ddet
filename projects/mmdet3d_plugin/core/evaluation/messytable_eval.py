import numpy as np
import copy

class MessytableEval:
    #def __init__(self, args, fx, fy):
    def __init__(self, args):
        self.num_valid_cam = args.num_valid_cam
        #self.fx, self.fy = fx, fy         
        self.class_list_wo_bg = args.class_list[:-1]
        self.reset()

        self.min_overlap = 0.5
        self.args = args

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
    '''

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
            cls = pred_box['class'] if not self.args.eval_rpn_only else 'water1'
            prob = pred_box['prob']
            bbox = self.get_x1y1x2y2(pred_box)
            dr_data[cls].append({"confidence":prob, "bbox":bbox})
        
        return dr_data

    def get_ground_truth_data(self, gt):
        ground_truth_data = {cls : [] for cls in self.class_list_wo_bg}

        for gt_box in gt:
            #cls = gt_box['class']
            cls = gt_box['class'] if not self.args.eval_rpn_only else 'water1'
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
                if det['confidence'] > self.args.eval_thresh :
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

    def main(self):
        pass

def log_eval(map_calculator, log_manager) :
    eval = map_calculator.get_eval()
    mean_eval = map_calculator.get_mean_eval()
    metric = map_calculator.metric
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
    log_manager.write_log('metric\t%s'%(metric_name))
    log_manager.write_log('ALL\t%s'%(metric_value))

    for _, cls in enumerate(valid_cls):
        ev = [e[cls] for e in eval]
        ev = ['%.2f'%(e) for e in ev]
        ev = '\t'.join(ev)
        #if e<0: continue
        log_manager.write_log('%s\t%s'%(cls, ev))
    log_manager.write_log('\n')

