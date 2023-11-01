import numpy as np
import copy
import json
import os
import matplotlib.pyplot as plt
import datetime
import tempfile

from .utils.json_writer import Json_saver
from .utils.log_manager import Log_manager
from .utils.dataloader import DataLoader
from .detection.map import Map_calculator
from .reid.map import Reid_evaluator

class MessytableEval:
    def __init__(self, num_views, class_list, det_path, gt, output_dir, img_dir, min_overlap=.5, cls_thresh=0, visible_thresh=0, reid_thresh=0, save_query=False, gt_path=None):
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

        self.json_save_path = os.path.join(output_dir, 'det.json')
        self.Json_saver = Json_saver(img_dir, self.json_save_path, num_views, gt_path)

        #self.Reid_evaluator = Reid_evaluator(img_dir, self.json_save_path)

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

            self.Json_saver.add_data(scene_id, det)

            result = [scene_id, gt_bboxes.transpose(1, 0, 2), gt_is_valids.transpose(1, 0), gt_cls, det_bboxes.transpose(1, 0, 2), det_is_valids.transpose(1, 0), det_cls, det_score]

            if self.save_query :
                result.append(det_query.transpose(1, 0, 2))

            result_list.append(result)

        self.Json_saver.close()
        #reid_eval_result = self.Reid_evaluator.eval()

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


