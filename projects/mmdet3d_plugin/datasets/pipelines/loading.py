# ------------------------------------------------------------------------
# Copyright (c) 2022 Toyota Research Institute, Dian Chen. All Rights Reserved.
# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
import math
import pickle

import mmcv
import numpy as np
from mmdet.datasets.builder import PIPELINES
from einops import rearrange
from itertools import combinations

from ...core.bbox.util import x1y1x2y22cxcywh


@PIPELINES.register_module()
class LoadMapsFromFiles(object):

    def __init__(self, k=None):
        self.k = k

    def __call__(self, results):
        map_filename = results['map_filename']
        maps = np.load(map_filename)
        map_mask = maps['arr_0'].astype(np.float32)

        maps = map_mask.transpose((2, 0, 1))
        results['gt_map'] = maps
        maps = rearrange(maps, 'c (h h1) (w w2) -> (h w) c h1 w2 ', h1=16, w2=16)
        maps = maps.reshape(256, 3 * 256)
        results['map_shape'] = maps.shape
        results['maps'] = maps
        return results


@PIPELINES.register_module()
class LoadMultiViewImageFromMultiSweepsFiles(object):
    """Load multi channel images from a list of separate channel files.
    Expects results['img_filename'] to be a list of filenames.
    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(
        self,
        sweeps_num=5,
        to_float32=False,
        file_client_args=dict(backend='disk'),
        pad_empty_sweeps=False,
        sweep_range=[3, 27],
        time_range=-1,
        sweeps_id=None,
        color_type='unchanged',
        sensors=['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'],
        test_mode=True,
        prob=1.0,
    ):

        self.sweeps_num = sweeps_num
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.pad_empty_sweeps = pad_empty_sweeps
        self.sensors = sensors
        self.test_mode = test_mode
        self.sweeps_id = sweeps_id
        self.sweep_range = sweep_range
        self.time_range = time_range
        self.prob = prob
        if self.sweeps_id:
            assert len(self.sweeps_id) == self.sweeps_num

    def __call__(self, results):
        """Call function to load multi-view image from files.
        Args:
            results (dict): Result dict containing multi-view image filenames.
        Returns:
            dict: The result dict containing the multi-view image data. \
                Added keys and values are described below.
                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        sweep_imgs_list = []
        timestamp_imgs_list = []
        imgs = results['img']
        img_timestamp = results['img_timestamp']
        lidar_timestamp = results['timestamp']
        img_timestamp = [lidar_timestamp - timestamp for timestamp in img_timestamp]
        sweep_imgs_list.extend(imgs)
        timestamp_imgs_list.extend(img_timestamp)
        nums = len(imgs)
        if self.pad_empty_sweeps and len(results['cam_sweeps']) == 0:
            for i in range(self.sweeps_num):
                sweep_imgs_list.extend(imgs)
                mean_time = (self.sweep_range[0] + self.sweep_range[1]) / 2.0 * 0.083
                timestamp_imgs_list.extend([time + mean_time for time in img_timestamp])
                for j in range(nums):
                    results['filename'].append(results['filename'][j])
                    results['lidar2img'].append(np.copy(results['lidar2img'][j]))
                    results['intrinsics'].append(np.copy(results['intrinsics'][j]))
                    results['extrinsics'].append(np.copy(results['extrinsics'][j]))
        else:
            if self.sweeps_id:
                choices = self.sweeps_id
            elif len(results['cam_sweeps']) <= self.sweeps_num:
                choices = np.arange(len(results['cam_sweeps']))
            elif self.test_mode:
                # choices = [int((self.sweep_range[0] + self.sweep_range[1]) / 2) - 1]
                max_range = min(self.sweep_range[1], len(results['cam_sweeps']))
                if max_range - self.sweep_range[0] < self.sweeps_num:
                    choices = list(range(self.sweep_range[0], max_range))
                    choices = (choices * math.ceil(self.sweeps_num / len(choices)))[:self.sweeps_num]
                else:
                    interval = int((max_range - self.sweep_range[0]) / (self.sweeps_num + 1))
                    choices = [self.sweep_range[0] + interval * (i + 1) for i in range(self.sweeps_num)]
            else:
                if np.random.random() < self.prob:
                    max_range = min(self.sweep_range[1], len(results['cam_sweeps']))
                    sweep_range = list(range(self.sweep_range[0], max_range))
                    choices = np.random.choice(
                        sweep_range, self.sweeps_num, replace=max_range - self.sweep_range[0] < self.sweeps_num)

                else:
                    choices = [int((self.sweep_range[0] + self.sweep_range[1]) / 2) - 1]

            choices = sorted(choices)
            for idx in choices:
                sweep_idx = min(idx, len(results['cam_sweeps']) - 1)
                sweep = results['cam_sweeps'][sweep_idx]
                if len(sweep.keys()) < len(self.sensors):
                    sweep = results['cam_sweeps'][sweep_idx - 1]
                results['filename'].extend([sweep[sensor]['data_path'] for sensor in self.sensors])

                img = np.stack([mmcv.imread(sweep[sensor]['data_path'], self.color_type) for sensor in self.sensors],
                               axis=-1)

                if self.to_float32:
                    img = img.astype(np.float32)
                img = [img[..., i] for i in range(img.shape[-1])]
                sweep_imgs_list.extend(img)
                sweep_ts = [lidar_timestamp - sweep[sensor]['timestamp'] / 1e6 for sensor in self.sensors]
                timestamp_imgs_list.extend(sweep_ts)
                for sensor in self.sensors:
                    results['lidar2img'].append(sweep[sensor]['lidar2img'])
                    results['intrinsics'].append(sweep[sensor]['intrinsics'])
                    # due to inverse convention in our repo
                    results['extrinsics'].append(np.linalg.inv(sweep[sensor]['extrinsics']).T)
        results['img'] = sweep_imgs_list
        if self.time_range > 0:
            timestamp_imgs_list = [time / self.time_range for time in timestamp_imgs_list]
        results['timestamp'] = timestamp_imgs_list

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32}, '
        repr_str += f"color_type='{self.color_type}')"
        return repr_str

@PIPELINES.register_module()
class LoadMultiViewRpnFromFiles(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(self):
        pass 

    def __call__(self, results):
        pickle_path = results['pickle_path']
        with open(pickle_path, 'rb') as f:
            pred_box, pred_box_emb, pred_box_prob = pickle.load(f) 

        results['pickle_path'] = pickle_path
        results['rpn_x1y1x2y2'] = pred_box.astype(np.float32) #(300,3,4) x1y1x2y2
        results['rpn_cxcywh'] = x1y1x2y22cxcywh(results['rpn_x1y1x2y2']) #(300,3,4) cxcywh
        results['rpn_emb'] = pred_box_emb.astype(np.float32) #(300,3,128)
        results['rpn_prob'] = pred_box_prob #(300,3)
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        return repr_str

@PIPELINES.register_module()
class LoadMultiViewGTFeatFromFiles(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(self, num_input, num_views, feat_size, model, shuffle=True, empty_emb_is_one=False):
        self.num_input = num_input
        self.num_views = num_views
        self.feat_size = feat_size
        self.model = model
        self.shuffle = shuffle
        self.empty_emb_is_one = empty_emb_is_one

    def __call__(self, results):
        pickle_path = results['pickle_path']

        pred_box = np.zeros((self.num_input, self.num_views, 4)) #(num_sample, 9, 4)
        if self.empty_emb_is_one :
            pred_box_emb = np.ones((self.num_input, self.num_views, self.feat_size)) #(num_sample, 9, 1024)
        else :
            pred_box_emb = np.zeros((self.num_input, self.num_views, self.feat_size)) #(num_sample, 9, 1024)
        pred_box_prob = np.zeros((self.num_input, self.num_views)) #(num_sample, 9)

        with open(pickle_path, 'rb') as f:
            data = pickle.load(f) 

        if self.model == 'asnet' :
            box, app_feat, sur_feat = data #(num_sample, 9, 4), (num_sample, 9, 512), (num_sample, 9, 512)
            num_sample, _, app_feat_size = app_feat.shape
           
            pred_box_idx = results['pred_box_idx_org'].astype('int') #(num_gt, 9)
            if self.shuffle:
                is_valid = (pred_box_idx != -1)
                num_gt = len(pred_box_idx) #num_gt

                view_idx = np.arange(self.num_views).repeat(num_gt).reshape(self.num_views, num_gt).transpose() #(num_gt, 9)

                old_idx = (pred_box_idx.flatten(), view_idx.flatten())
                old2new_dict = np.array([np.random.choice(range(self.num_input), self.num_input, replace=False) for _ in range(self.num_views)]).transpose() #(num_sample, num_views)
                new_idx = old2new_dict[old_idx] #(num_gt * 9)
                new_pred_box_idx = new_idx.reshape(num_gt, self.num_views) #(num_gt, self.num_views)
                pred_box_idx[is_valid] = new_pred_box_idx[is_valid]
                
                dst_idx = (new_pred_box_idx.flatten(), view_idx.flatten())
                pred_box[dst_idx] = box[old_idx]
                pred_box_emb[dst_idx] = np.concatenate([app_feat[old_idx], sur_feat[old_idx]], -1)[..., :self.feat_size]
                pred_box_prob[dst_idx] = 1.
            else :
                pred_box[:num_sample] = box
                pred_box_emb[:num_sample] = np.concatenate([app_feat, sur_feat], -1)[..., :self.feat_size]
                pred_box_prob[:num_sample] = 1.
     
        results['pred_box_idx'] = pred_box_idx
        results['rpn_x1y1x2y2'] = pred_box.astype(np.float32) #(100,9,4) x1y1x2y2
        results['rpn_cxcywh'] = x1y1x2y22cxcywh(results['rpn_x1y1x2y2']) #(100,9,4) cxcywh
        results['rpn_emb'] = pred_box_emb.astype(np.float32) #(100,9,128)
        results['rpn_prob'] = pred_box_prob.astype(np.float32) #(100,9)
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        return repr_str


