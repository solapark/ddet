# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
import pyquaternion
import tempfile
from nuscenes.utils.data_classes import Box as NuScenesBox
from os import path as osp
import torch
import time 

from mmdet.datasets import DATASETS
#from mmdet.core.visualization import imshow_gt_det_bboxes
from ..core.visualization.show_result_mtv2d import show_result_mtv2d
#from ..core.bbox import Box3DMode, Coord3DMode, LiDARInstance3DBoxes
from mmdet3d.core.bbox import LiDARInstance3DBoxes
#from ..core.bbox.structures.mtv_boxes2d import MtvBoxes2D
from .messytable_dataset import CustomMessytableDataset
from mmdet3d.datasets.pipelines import Compose


@DATASETS.register_module()
class CustomMessytableRpnDataset(CustomMessytableDataset):
    r"""Messytable Dataset.

    This class serves as the API for experiments on the Messytable Dataset.

    Args:
        ann_file (str): Path of annotation file.
        pipeline (list[dict], optional): Pipeline used for data processing.
            Defaults to None.
        data_root (str): Path of dataset root.
        classes (tuple[str], optional): Classes used in the dataset.
            Defaults to None.
        load_interval (int, optional): Interval of loading the dataset. It is
            used to uniformly sample the dataset. Defaults to 1.
        with_velocity (bool, optional): Whether include velocity prediction
            into the experiments. Defaults to True.
        modality (dict, optional): Modality to specify the sensor data used
            as input. Defaults to None.
        box_type_3d (str, optional): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'LiDAR' in this dataset. Available options includes.
            - 'LiDAR': Box in LiDAR coordinates.
            - 'Depth': Box in depth coordinates, usually for indoor dataset.
            - 'Camera': Box in camera coordinates.
        filter_empty_gt (bool, optional): Whether to filter empty GT.
            Defaults to True.
        test_mode (bool, optional): Whether the dataset is in test mode.
            Defaults to False.
        eval_version (bool, optional): Configuration version of evaluation.
            Defaults to  'detection_cvpr_2019'.
        use_valid_flag (bool): Whether to use `use_valid_flag` key in the info
            file as mask to filter gt_boxes and gt_names. Defaults to False.
    """
    ErrNameMapping = {
        'det_err': 'mAP',
        'reid_err': 'AP',
    }

    CLASSES = ('water1', 'water2', 'pepsi', 'coca1', 'coca2', 'coca3', 'coca4', 'tea1', 'tea2', 'yogurt', 'ramen1', 'ramen2', 'ramen3', 'ramen4', 'ramen5', 'ramen6', 'ramen7', 'juice1', 'juice2', 'can1', 'can2', 'can3', 'can4', 'can5', 'can6', 'can7', 'can8', 'can9', 'ham1', 'ham2', 'pack1', 'pack2', 'pack3', 'pack4', 'pack5', 'pack6', 'snack1', 'snack2', 'snack3', 'snack4', 'snack5', 'snack6', 'snack7', 'snack8', 'snack9', 'snack10', 'snack11', 'snack12', 'snack13', 'snack14', 'snack15', 'snack16', 'snack17', 'snack18', 'snack19', 'snack20', 'snack21', 'snack22', 'snack23', 'snack24', 'green_apple', 'red_apple', 'tangerine', 'lime', 'lemon', 'yellow_quince', 'green_quince', 'white_quince', 'fruit1', 'fruit2', 'peach', 'banana', 'fruit3', 'pineapple', 'fruit4', 'strawberry', 'cherry', 'red_pimento', 'green_pimento', 'carrot', 'cabbage1', 'cabbage2', 'eggplant', 'bread', 'baguette', 'sandwich', 'hamburger', 'hotdog', 'donuts', 'cake', 'onion', 'marshmallow', 'mooncake', 'shirimpsushi', 'sushi1', 'sushi2', 'big_spoon', 'small_spoon', 'fork', 'knife', 'big_plate', 'small_plate', 'bowl', 'white_ricebowl', 'blue_ricebowl', 'black_ricebowl', 'green_ricebowl', 'black_mug', 'gray_mug', 'pink_mug', 'green_mug', 'blue_mug', 'blue_cup', 'orange_cup', 'yellow_cup', 'big_wineglass', 'small_wineglass', 'glass1', 'glass2', 'glass3')

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - scene_id (str): sample_idx.
                - img_filename (str, optional): Image filename.
                - world2img (list[np.ndarray], optional): Transformations \
                    from world to different cameras's image.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
        # standard protocal modified from SECOND.Pytorch
        input_dict = dict(
            scene_id=info['scene_id'],
            scene_token=info['scene_token'],
            pickle_path=info['pickle_path'],
            cams=info['cams'],
            valid_flags=info['valid_flags'],
            gt_bboxes_3d=info['gt_ids'], 
            gt_names=info['gt_names'],
            cam_instances=info['cam_instances'],
            cam_instances_valid_flags=info['cam_instances_valid_flags'],
            inst_3dp=info['inst_3dp'],  #(num_inst, 3) #cx,cy,cz
            inst_proj_2dp=info['inst_proj_2dp'], #(num_inst, num_cam, 2) #cx,cy
            pred_box_idx=info['pred_box_idx'], #(num_inst, num_cam) #cam1_idx,cam2_idx,cam3_idx
            probs=info['probs'] #(num_inst, num_cam)    
        )

        image_paths = []
        world2img_rts = []
        intrinsics = []
        extrinsics = []
        for i, (cam_type, cam_info) in enumerate(info['cams'].items()):
            if i == self.num_views : break
            image_paths.append(cam_info['img_path'])
            intrinsics.append(cam_info['intrinsic'])
            extrinsics.append(cam_info['extrinsic'])
            world2img_rts.append(cam_info['world2img'])

        input_dict.update(
            dict(
                img_filename=image_paths,
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                world2img=world2img_rts,
            ))

        #if not self.test_mode:
        #    annos = self.get_ann_info(index)
        #    input_dict['ann_info'] = annos

        annos = self.get_ann_info(index)
        input_dict['ann_info'] = annos

        return input_dict


    def _format_bbox(self, results, jsonfile_prefix=None):
        """Convert the results to the standard format.

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str): The prefix of the output jsonfile.
                You can specify the output directory/filename by
                modifying the jsonfile_prefix. Default: None.

        Returns:
            str: Path of the output json file.
        """
        messytable_annos = {}
        mapped_class_names = self.CLASSES

        print('Start to convert detection format...')
        for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
            annos = []
            boxes = output_to_messytable_box(det)
            scene_id = self.data_infos[sample_id]['scene_id']

            for i, box in enumerate(boxes):
                messytable_anno = dict(
                    scene_id=scene_id,
                    camera_instances=box['boxes'],
                    camera_instances_valid_flag=box['valid'],
                    detection_name=mapped_class_names[box['label']],
                    reid_score=box['reid_score'],
                    detection_score=box['cls_score'],
                    )
                annos.append(messytable_anno)
            messytable_annos[scene_id] = annos

        messytable_submissions = messytable_annos 

        mmcv.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, 'results_messytable.json')
        print('Results writes to', res_path)
        mmcv.dump(messytable_submissions, res_path)
        return res_path

def output_to_messytable_box(detection):
    """Convert the output to the box class in the nuScenes.

    Args:
        detection (dict): Detection results.

            - boxes_3d (:obj:`BaseInstance3DBoxes`): Detection bbox.
            - scores_3d (torch.Tensor): Detection scores.
            - labels_3d (torch.Tensor): Predicted box labels.

    Returns:
        list[:obj:`NuScenesBox`]: List of standard NuScenesBoxes.
    """
    boxmtv2d = detection['boxes_mtv2d'].numpy()
    visibles = detection['visibles_mtv2d'].numpy()
    reid_scores = detection['reid_scores_mtv2d'].numpy()
    cls_scores = detection['cls_scores_mtv2d'].numpy()
    labels = detection['labels_mtv2d'].numpy()

    box_list = []
    for i in range(len(boxmtv2d)):
        box = dict(
            boxes = boxmtv2d[i],
            valid = visibles[i],
            label = labels[i],
            reid_score = reid_scores[i],
            cls_score = cls_scores[i],
            )
        box_list.append(box)
    return box_list
