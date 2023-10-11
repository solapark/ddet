import torch

def bboxmtv2result(bboxes, visibles, scores, labels):
    """Convert detection results to a list of numpy arrays.

    Args:
        bboxes (torch.Tensor): Bounding boxes with shape of (n, 5).
        labels (torch.Tensor): Labels with shape of (n, ).
        scores (torch.Tensor): Scores with shape of (n, ).
        attrs (torch.Tensor, optional): Attributes with shape of (n, ). \
            Defaults to None.

    Returns:
        dict[str, torch.Tensor]: Bounding box results in cpu mode.

            - boxes_3d (torch.Tensor): 3D boxes.
            - scores (torch.Tensor): Prediction scores.
            - labels_3d (torch.Tensor): Box labels.
            - attrs_3d (torch.Tensor, optional): Box attributes.
    """
    result_dict = dict(
        boxes_mtv2d=bboxes.to('cpu'),
        visibles_mtv2d=visibles.cpu(),
        scores_mtv2d=scores.cpu(),
        labels_mtv2d=labels.cpu())

    return result_dict

def bboxmtvreid2result(bboxes, visibles, reid_scores, cls_scores, labels):
    """Convert detection results to a list of numpy arrays.

    Args:
        bboxes (torch.Tensor): Bounding boxes with shape of (n, 5).
        labels (torch.Tensor): Labels with shape of (n, ).
        scores (torch.Tensor): Scores with shape of (n, ).
        attrs (torch.Tensor, optional): Attributes with shape of (n, ). \
            Defaults to None.

    Returns:
        dict[str, torch.Tensor]: Bounding box results in cpu mode.

            - boxes_3d (torch.Tensor): 3D boxes.
            - scores (torch.Tensor): Prediction scores.
            - labels_3d (torch.Tensor): Box labels.
            - attrs_3d (torch.Tensor, optional): Box attributes.
    """
    result_dict = dict(
        boxes_mtv2d=bboxes.to('cpu'),
        visibles_mtv2d=visibles.cpu(),
        reid_scores_mtv2d=reid_scores.cpu(),
        cls_scores_mtv2d=cls_scores.cpu(),
        labels_mtv2d=labels.cpu())

    return result_dict
