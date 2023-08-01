# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
import pyquaternion
import tempfile
from nuscenes.utils.data_classes import Box as NuScenesBox
from os import path as osp
import torch

from mmdet.datasets import DATASETS
#from mmdet.core.visualization import imshow_gt_det_bboxes
from ..core.visualization.show_result_mtv2d import show_result_mtv2d
#from ..core.bbox import Box3DMode, Coord3DMode, LiDARInstance3DBoxes
from mmdet3d.core.bbox import LiDARInstance3DBoxes
#from ..core.bbox.structures.mtv_boxes2d import MtvBoxes2D
from .custom_mtv2d import CustomMtv2DDataset
from mmdet3d.datasets.pipelines import Compose


@DATASETS.register_module()
class CustomMessytableDataset(CustomMtv2DDataset):
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


    def __init__(self,
                 ann_file,
                 pipeline=None,
                 data_root=None,
                 classes=None,
                 load_interval=1,
                 filter_empty_gt=True,
                 test_mode=False,
                 use_valid_flag=False):
        self.load_interval = load_interval
        self.use_valid_flag = use_valid_flag
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=classes,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode)

    def get_cat_ids(self, idx):
        """Get category distribution of single scene.

        Args:
            idx (int): Index of the data_info.

        Returns:
            dict[list]: for each category, if the current scene
                contains such boxes, store a list containing idx,
                otherwise, store empty list.
        """
        info = self.data_infos[idx]
        if self.use_valid_flag:
            mask = info['valid_flag']
            gt_names = set(info['gt_names'][mask])
        else:
            gt_names = set(info['gt_names'])

        cat_ids = []
        for name in gt_names:
            if name in self.CLASSES:
                cat_ids.append(self.cat2id[name])
        return cat_ids

    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations sorted by timestamps.
        """
        data = mmcv.load(ann_file)
        #data_infos = list(sorted(data['infos'], key=lambda e: e['timestamp']))
        data_infos = list(data['infos'])
        data_infos = data_infos[::self.load_interval]
        self.metadata = data['metadata']
        self.version = self.metadata['version']
        return data_infos

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
            cams=info['cams'],
            valid_flags=info['valid_flags'],
            gt_bboxes_3d=info['gt_ids'], 
            gt_names=info['gt_names'],
            cam_instances=info['cam_instances'],
            cam_instances_valid_flags=info['cam_instances_valid_flags'],
        )

        image_paths = []
        world2img_rts = []
        intrinsics = []
        extrinsics = []
        for cam_type, cam_info in info['cams'].items():
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

    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): \
                    3D ground truth bboxes
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_names (list[str]): Class names of ground truths.
        """
        info = self.data_infos[index]
        mask = info['valid_flags'].astype('bool')
        gt_ids = np.array(info['gt_ids'])[mask]
        gt_names = np.array(info['gt_names'])[mask]
        gt_labels = []
        for cat in gt_names:
            if cat in self.CLASSES:
                gt_labels.append(self.CLASSES.index(cat))
            else:
                gt_labels.append(-1)
        gt_labels = np.array(gt_labels)

        gt_bboxes_3d=LiDARInstance3DBoxes(
            np.reshape(gt_ids, (-1, 1)).astype('int'), 
            box_dim=1, 
            with_yaw=False)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels,
            gt_names=gt_names)

        return anns_results

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
                    detection_score=box['score'],
                    )
                annos.append(messytable_anno)
            messytable_annos[scene_id] = annos

        messytable_submissions = messytable_annos 

        mmcv.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, 'results_messytable.json')
        print('Results writes to', res_path)
        mmcv.dump(messytable_submissions, res_path)
        return res_path

    def _evaluate_single(self,
                         result_path,
                         logger=None,
                         metric='bbox',
                         result_name='pts_bbox'):
        """Evaluation for a single model in nuScenes protocol.

        Args:
            result_path (str): Path of the result file.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            metric (str): Metric name used for evaluation. Default: 'bbox'.
            result_name (str): Result name in the metric prefix.
                Default: 'pts_bbox'.

        Returns:
            dict: Dictionary of evaluation details.
        """
        from ..core.evaluation.messytable_eval import MessytableEval

        output_dir = osp.join(*osp.split(result_path)[:-1])
        messytable_eval = MessytableEval(
            result_path=result_path,
            output_dir=output_dir,
            verbose=False)
        messytable_eval.main(render_curves=False)

        # record metrics
        metrics = mmcv.load(osp.join(output_dir, 'metrics_summary.json'))
        detail = dict()
        metric_prefix = f'{result_name}_Messytable'
        detail['{}/mAP'.format(metric_prefix)] = metrics['mean_ap']
        return detail

    def format_results(self, results, jsonfile_prefix=None):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: Returns (result_files, tmp_dir), where `result_files` is a \
                dict containing the json filepaths, `tmp_dir` is the temporal \
                directory created for saving json files when \
                `jsonfile_prefix` is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None

        # currently the output prediction results could be in two formats
        # 1. list of dict('boxes_3d': ..., 'scores_3d': ..., 'labels_3d': ...)
        # 2. list of dict('pts_bbox' or 'img_bbox':
        #     dict('boxes_3d': ..., 'scores_3d': ..., 'labels_3d': ...))
        # this is a workaround to enable evaluation of both formats on nuScenes
        # refer to https://github.com/open-mmlab/mmdetection3d/issues/449
        if not ('pts_bbox' in results[0] or 'img_bbox' in results[0]):
            result_files = self._format_bbox(results, jsonfile_prefix)
        else:
            # should take the inner dict out of 'pts_bbox' or 'img_bbox' dict
            result_files = dict()
            for name in results[0]:
                print(f'\nFormating bboxes of {name}')
                results_ = [out[name] for out in results]
                tmp_file_ = osp.join(jsonfile_prefix, name)
                result_files.update(
                    {name: self._format_bbox(results_, tmp_file_)})
        return result_files, tmp_dir

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 result_names=['pts_bbox'],
                 show=False,
                 out_dir=None,
                 pipeline=None):
        """Evaluation in nuScenes protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            show (bool): Whether to visualize.
                Default: False.
            out_dir (str): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict[str, float]: Results of each evaluation metric.
        """
        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)

        '''
        if isinstance(result_files, dict):
            results_dict = dict()
            for name in result_names:
                print('Evaluating bboxes of {}'.format(name))
                ret_dict = self._evaluate_single(result_files[name])
            results_dict.update(ret_dict)
        elif isinstance(result_files, str):
            results_dict = self._evaluate_single(result_files)

        if tmp_dir is not None:
            tmp_dir.cleanup()

        if show:
            self.show(results, out_dir, pipeline=pipeline)
        '''

        results_dict = dict()
        return results_dict

    def _build_default_pipeline(self):
        """Build the default pipeline for this dataset."""
        pipeline = [
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5,
                file_client_args=dict(backend='disk')),
            dict(
                type='LoadPointsFromMultiSweeps',
                sweeps_num=10,
                file_client_args=dict(backend='disk')),
            dict(
                type='DefaultFormatBundle3D',
                class_names=self.CLASSES,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ]
        return Compose(pipeline)

    def show(self, results, out_dir, show=True, pipeline=None, wait_time=0):
        """Results visualization.

        Args:
            results (list[dict]): List of bounding boxes results.
            out_dir (str): Output directory of visualization result.
            show (bool): Visualize the results online.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.
        """
        assert out_dir is not None, 'Expect out_dir, got none.'
        pipeline = self._get_pipeline(pipeline)
        for i, result in enumerate(results):
            if 'pts_bbox' in result.keys():
                result = result['pts_bbox']
            data_info = self.data_infos[i]
            scene_id = data_info['scene_id']

            show_gt_ids = self.get_ann_info(i)['gt_ids'].tensor.numpy()
            show_gt_names = self.get_ann_info(i)['gt_names'].tensor.numpy()
            show_gt_cam_instances = self.get_ann_info(i)['cam_instances'].tensor.numpy()
            show_gt_cam_instances_valid_flags = self.get_ann_info(i)['cam_instances_valid_flags'].tensor.numpy()
            show_gt_cams = self.get_ann_info(i)['cams'].tensor.numpy()

            inds = result['scores_3d'] > 0.1
            show_pred_bboxes = result['boxes_3d'][inds].tensor.numpy()

            out_file = os.path.join(out_dir, scene_id+'.jpg')
            show_result_mtv2D(show_gt_cam_instances, show_gt_cam_instances_valid_flags, show_gt_cams, show_pred_bboxes, out_file=out_file, show=show, wait_time=wait_time) 

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
    boxmtv2d = detection['boxes_mtv2d']
    visibles = detection['visibles_mtv2d'].numpy()
    scores = detection['scores_mtv2d'].numpy()
    labels = detection['labels_mtv2d'].numpy()

    num_views = visibles.shape[-1]
    num_codes = boxmtv2d.shape[-1] // (num_views+1)
    boxmtv2d = boxmtv2d[:, num_codes:]
    cx = boxmtv2d[:, 0::num_codes] 
    cy = boxmtv2d[:, 1::num_codes] 
    w = boxmtv2d[:, 3::num_codes] 
    h = boxmtv2d[:, 5::num_codes] 
    boxmtv2d = torch.cat([cx, cy, w, h], dim=-1)
    boxmtv2d = boxmtv2d.reshape(*boxmtv2d.shape[:-1], 4, num_views).transpose(-1, -2).flatten(-2).numpy()

    box_list = []
    for i in range(len(boxmtv2d)):
        box = dict(
            boxes = boxmtv2d[i],
            valid = visibles[i],
            label = labels[i],
            score = scores[i],
            )
        box_list.append(box)
    return box_list
