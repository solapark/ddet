import numpy as np
import json
import os
import pickle
from projects.mmdet3d_plugin.core.bbox.util import cxcywh2x1y1x2y2

class ReidPickleSaver:
    def __init__(self, num_views, det_path, output_dir, visible_thresh=0):
        self.num_valid_cam = num_views
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
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
            for inst in det :
                det_bboxes.append(inst['camera_instances'])
                det_is_valids.append(inst['camera_instances_valid_flag'])
            det_bboxes = np.array(det_bboxes).reshape((1, num_det, self.num_valid_cam, -1)) #(1, 300, num_views, 4)
            det_bboxes = cxcywh2x1y1x2y2(det_bboxes)
            det_bboxes = np.rint(det_bboxes * self.resize_ratio / self.rpn_stride).astype(np.int32)
            det_is_valids = np.array(det_is_valids).reshape(1, num_det, self.num_valid_cam) #(1, 300, num_views)
            det_is_valids = (det_is_valids>self.visible_thresh)
            det_bboxes[~det_is_valids] = -1
            det_is_valids = det_is_valids.astype(np.int32)
            #emb_batch = np.zeros_like(det_is_valids) #(1, 300, 3)

            save_path = os.path.join(self.output_dir, scene_id) + '.pickle'
            #self.save_pickle(save_path, [det_bboxes, det_is_valids, emb_batch])
            self.save_pickle(save_path, [det_bboxes, det_is_valids])

    def save_pickle(self, path, content):
        with open(path, 'wb') as f:
            pickle.dump(content, f)        
