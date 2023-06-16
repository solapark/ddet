import mmcv
import numpy as np
from os import path as osp

from mmdet.core.visualization import imshow_gt_det_bboxes

def show_result_mtv2d(show_gt_cam_instances,
                show_gt_cam_instances_valid_flags, 
                show_gt_cams, 
                show_pred_bboxes, 
                out_file, 
                show=True,
                wait_time=0):
    """Convert results into format that is directly readable for meshlab.

    Args:
        points (np.ndarray): Points.
        gt_bboxes (np.ndarray): Ground truth boxes.
        pred_bboxes (np.ndarray): Predicted boxes.
        out_dir (str): Path of output directory
        filename (str): Filename of the current frame.
        show (bool): Visualize the results online. Defaults to False.
        snapshot (bool): Whether to save the online results. Defaults to False.
    """
    result_path = osp.join(out_dir, filename)
    mmcv.mkdir_or_exist(result_path)

    img_list = []
    for gt_cam, cam_gt_boxes, cam_valid_falgs, cam_pred_boxes in zip(show_gt_cams, show_gt_cam_instances, show_gt_cam_instances_valid_flags, show_pred_bboxes) : 
        img_path = gt_cam['img_path']
        img = mmvc.imread(img_path)
        cam_gt_boxes = cam_gt_boxes[cam_valid_falgs]
        img = imshow_gt_det_bboxes(img, cam_gt_boxes, cam_pred_boxes, show=False)
        img_list.append(img)
    img = np.concatenate(img_list, axis=0)

    plt.imshow(img)

    if show:
        if wait_time == 0:
            plt.show()
        else:
            plt.show(block=False)
            plt.pause(wait_time)

    if out_file is not None:
        mmcv.imwrite(img, out_file)

    plt.close()
