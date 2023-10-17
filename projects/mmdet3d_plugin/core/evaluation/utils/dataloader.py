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
                    info = {'class':cls, 'x1':x1, 'y1':y1, 'x2':x2, 'y2':y2, 'inst_idx':sample_idx+1}
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
                    info = {'class':cls, 'x1':x1, 'y1':y1, 'x2':x2, 'y2':y2, 'prob':score, 'is_valid':is_valid, 'inst_idx':sample_idx+1}
                    data[cam_idx].append(info)
        return data 


