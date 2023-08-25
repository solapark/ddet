import mmcv
import numpy as np
from os import path as osp
import cv2

from mmdet.core.visualization import imshow_gt_det_bboxes

def show_result_mtv2d(data_root, 
                out_dir, 
                result,
                eval_thresh, 
                show=True,
                show_gt=True,
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
    scene_id, gt_bboxes, gt_is_valids, gt_cls, det_bboxes, det_is_valids, det_cls, det_score = result

    
    img_list = []
    for i, (cam_gt_bboxes, cam_gt_is_valids, cam_det_bboxes, cam_det_is_valids) in enumerate(zip(gt_bboxes, gt_is_valids, det_bboxes, det_is_valids)) : 
        img_path = osp.join(data_root, '%s-%02d.jpg'%(scene_id, i+1))
        img = mmcv.imread(img_path)

        cam_gt_is_valids = cam_gt_is_valids > 0 #(num_gt, )
        cam_gt_bboxes = cam_gt_bboxes[cam_gt_is_valids] #(num_gt, 4) #(cx cy w h)
        cam_gt_cls = gt_cls[cam_gt_is_valids] #(num_gt, ) #str

        cam_det_is_valids =  (cam_det_is_valids>.5) & (det_score > eval_thresh) #(num_pred, )
        cam_det_bboxes = cam_det_bboxes[cam_det_is_valids] #(num_pred, 4) #(cx cy w h)
        cam_det_cls = det_cls[cam_det_is_valids] #(num_pred,) #str 
        cam_det_score = det_score[cam_det_is_valids] #(num_pred,) #float

        img = imshow_gt_det_bboxes(img, cam_gt_bboxes, cam_gt_cls, cam_det_bboxes, cam_det_cls, cam_det_score, show_gt)
        img_list.append(img)
    img = np.concatenate(img_list, axis=0)

    result_path = osp.join(out_dir, '%s.jpg'%(scene_id))
    mmcv.imwrite(img, result_path)

def imshow_gt_det_bboxes(img, cam_gt_bboxes, cam_gt_cls, cam_det_bboxes, cam_det_cls, cam_det_score, show_gt=True):
    # Make a copy of the image to draw bounding boxes on
    img_with_bboxes = img.copy()

    if show_gt :
        # Draw ground truth bounding boxes
        for bbox, cls in zip(cam_gt_bboxes, cam_gt_cls):
            x, y, w, h = bbox
            color = (0, 255, 0)  # Green color for ground truth
            cv2.rectangle(img_with_bboxes, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), color, 2)
            cv2.putText(img_with_bboxes, cls, (int(x - w / 2), int(y - h / 2) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Draw detected bounding boxes
    for bbox, cls, score in zip(cam_det_bboxes, cam_det_cls, cam_det_score):
        x, y, w, h = bbox
        color = (0, 0, 255)  # Red color for detected
        cv2.rectangle(img_with_bboxes, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), color, 2)
        cv2.putText(img_with_bboxes, f'{cls}: {score:.2f}', (int(x - w / 2), int(y - h / 2) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return img_with_bboxes
