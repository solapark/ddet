import numpy as np
import json
import os
import pickle
from projects.mmdet3d_plugin.core.bbox.util import cxcywh2x1y1x2y2

class ReidPickleSaver:
    def __init__(self, num_views, det_path, output_dir, eval_thresh, reid_thresh, visible_thresh):
        self.num_valid_cam = num_views
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        self.eval_thresh = eval_thresh
        self.reid_thresh = reid_thresh
        self.visible_thresh = visible_thresh
        self.rpn_stride = 16
        self.resize_ratio = 0.5555555555555556
        with open(det_path) as f:
            self.det = json.load(f)

    def main(self) :
        for scene_id in self.det:
            det = self.det[scene_id] #(300, )
            num_det = len(det) # 300

            det_bboxes = []
            det_is_valids = []
            det_score = []
            det_reid_score = []
            for inst in det :
                det_bboxes.append(inst['camera_instances'])
                det_is_valids.append(inst['camera_instances_valid_flag'])
                det_score.append(inst['detection_score'])
                det_reid_score.append(inst['reid_score'])
            det_bboxes = np.array(det_bboxes).reshape((1, num_det, self.num_valid_cam, -1)) #(1, 300, num_views, 4)
            det_bboxes = cxcywh2x1y1x2y2(det_bboxes)
            det_bboxes = np.rint(det_bboxes * self.resize_ratio / self.rpn_stride).astype(np.int32)
            det_is_valids = np.array(det_is_valids).reshape(1, num_det, self.num_valid_cam) #(1, 300, num_views)
            det_score = np.array(det_score).reshape(1, num_det) #(1, 300)
            det_reid_score = np.array(det_reid_score).reshape(1, num_det) #(1, 300)

            det_is_valids = (det_is_valids>self.visible_thresh)
            det_bboxes[~det_is_valids] = -1
            det_is_valids = det_is_valids.astype(np.int32)

            det_score_valid = (det_score > self.eval_thresh)
            det_reid_score_valid = (det_reid_score > self.reid_thresh)
            is_valid_query = det_score_valid & det_reid_score_valid
            det_bboxes = np.expand_dims(det_bboxes[is_valid_query], 0) #(1, G, 3, 4)
            det_is_valids = np.expand_dims(det_is_valids[is_valid_query], 0) #(1, G, 3)

            save_path = os.path.join(self.output_dir, scene_id) + '.pickle'
            #self.save_pickle(save_path, [det_bboxes, det_is_valids, emb_batch])
            self.save_pickle(save_path, [det_bboxes, det_is_valids])

    def save_pickle(self, path, content):
        with open(path, 'wb') as f:
            pickle.dump(content, f)        
