import torch
from copy import deepcopy
from .array_converter import array_converter


@array_converter(apply_to=('points', 'cam2img'))
def points_img2cam(points, cam2img):
    """Project points in image coordinates to camera coordinates.

    Args:
        points (torch.Tensor): 2.5D points in 2D images, [N, 3],
            3 corresponds with x, y in the image and depth.
        cam2img (torch.Tensor): Camera intrinsic matrix. The shape can be
            [3, 3], [3, 4] or [4, 4].

    Returns:
        torch.Tensor: points in 3D space. [N, 3],
            3 corresponds with x, y, z in 3D space.
    """
    assert cam2img.shape[0] <= 4
    assert cam2img.shape[1] <= 4
    assert points.shape[1] == 3

    xys = points[:, :2]
    depths = points[:, 2].view(-1, 1)
    unnormed_xys = torch.cat([xys * depths, depths], dim=1)

    pad_cam2img = torch.eye(4, dtype=xys.dtype, device=xys.device)
    pad_cam2img[:cam2img.shape[0], :cam2img.shape[1]] = cam2img
    inv_pad_cam2img = torch.inverse(pad_cam2img).transpose(0, 1)

    # Do operation in homogeneous coordinates.
    num_points = unnormed_xys.shape[0]
    homo_xys = torch.cat([unnormed_xys, xys.new_ones((num_points, 1))], dim=1)
    points3D = torch.mm(homo_xys, inv_pad_cam2img)[:, :3]

    return points3D

def normalize_pc3d(pc, pc_range) : 
    divider = torch.tensor([pc_range[3] - pc_range[0], pc_range[4] - pc_range[1], pc_range[5] - pc_range[2]], device=pc.device)
    subtract = torch.tensor([pc_range[0], pc_range[1], pc_range[2]], device=pc.device)
    return (pc - subtract) / (divider + 1e-6)

def denormalize_pc3d(pc, pc_range) : 
    divider = torch.tensor([pc_range[3] - pc_range[0], pc_range[4] - pc_range[1], pc_range[5] - pc_range[2]], device=pc.device)
    subtract = torch.tensor([pc_range[0], pc_range[1], pc_range[2]], device=pc.device)
    return pc * divider + subtract

def normalize_bbox(bboxes, pc_range):
    include_velocity = (bboxes.shape[-1] % 9 == 0)
    num_properties = 9 if include_velocity else 7
    num_views = bboxes.shape[-1] // num_properties

    cx = bboxes[..., 0::num_properties]
    cy = bboxes[..., 1::num_properties]
    cz = bboxes[..., 2::num_properties]
    w = bboxes[..., 3::num_properties].log()
    l = bboxes[..., 4::num_properties].log()
    h = bboxes[..., 5::num_properties].log()

    rot = bboxes[..., 6::num_properties]
    if include_velocity:
        vx = bboxes[..., 7::num_properties]
        vy = bboxes[..., 8::num_properties]
        # (..., 10 x V)
        normalized_bboxes = torch.cat((cx, cy, w, l, cz, h, rot.sin(), rot.cos(), vx, vy), dim=-1)
    else:
        # (..., 8 x V)
        normalized_bboxes = torch.cat((cx, cy, w, l, cz, h, rot.sin(), rot.cos()), dim=-1)

    normalized_bboxes = normalized_bboxes.reshape(*normalized_bboxes.shape[:-1], num_properties + 1, num_views)
    # (..., V x P)
    normalized_bboxes = normalized_bboxes.transpose(-1, -2).flatten(-2)

    return normalized_bboxes


def denormalize_bbox(normalized_bboxes, pc_range):
    include_velocity = (normalized_bboxes.shape[-1] % 10 == 0)
    num_properties = 10 if include_velocity else 8
    num_views = normalized_bboxes.shape[-1] // num_properties

    # rotation
    rot_sin = normalized_bboxes[..., 6::num_properties]
    rot_cos = normalized_bboxes[..., 7::num_properties]
    rot = torch.atan2(rot_sin, rot_cos)

    # center in the bev
    cx = normalized_bboxes[..., 0::num_properties]
    cy = normalized_bboxes[..., 1::num_properties]
    cz = normalized_bboxes[..., 4::num_properties]

    # size
    w = normalized_bboxes[..., 2::num_properties]
    l = normalized_bboxes[..., 3::num_properties]
    h = normalized_bboxes[..., 5::num_properties]

    w = w.exp()
    l = l.exp()
    h = h.exp()
    if include_velocity:
        # velocity
        vx = normalized_bboxes[:, 8::num_properties]
        vy = normalized_bboxes[:, 9::num_properties]
        # (..., 9 x V)
        denormalized_bboxes = torch.cat([cx, cy, cz, w, l, h, rot, vx, vy], dim=-1)
    else:
        # (..., 7 x V)
        denormalized_bboxes = torch.cat([cx, cy, cz, w, l, h, rot], dim=-1)

    denormalized_bboxes = denormalized_bboxes.reshape(*denormalized_bboxes.shape[:-1], num_properties - 1, num_views)
    # (..., V * P)
    denormalized_bboxes = denormalized_bboxes.transpose(-1, -2).flatten(-2)

    return denormalized_bboxes

def cxcywh2x1y1x2y2(boxes):
    """
    Convert bounding boxes from (cx, cy, w, h) format to (x1, y1, x2, y2) format.

    Args:
        boxes (torch.Tensor): Bounding boxes in (cx, cy, w, h) format.

    Returns:
        torch.Tensor: Bounding boxes in (x1, y1, x2, y2) format.
    """
    new_bboxes = deepcopy(boxes)
    new_bboxes[..., 0] = boxes[..., 0] - boxes[..., 2] / 2
    new_bboxes[..., 1] = boxes[..., 1] - boxes[..., 3] / 2
    new_bboxes[..., 2] = boxes[..., 0] + boxes[..., 2] / 2
    new_bboxes[..., 3] = boxes[..., 1] + boxes[..., 3] / 2

    return new_bboxes

def x1y1x2y22cxcywh(boxes):
    """
    Convert bounding boxes from (x1, y1, x2, y2) format to (cx, cy, w, h) format.

    Args:
        boxes (torch.Tensor): Bounding boxes in (x1, y1, x2, y2) format.

    Returns:
        torch.Tensor: Bounding boxes in (cx, cy, w, h) format.
    """
    new_bboxes = deepcopy(boxes)
    new_bboxes[..., 0] = (boxes[..., 0] + boxes[..., 2]) / 2
    new_bboxes[..., 1] = (boxes[..., 1] + boxes[..., 3]) / 2
    new_bboxes[..., 2] = boxes[..., 2] - boxes[..., 0]
    new_bboxes[..., 3] = boxes[..., 3] - boxes[..., 1]

    return new_bboxes

def get_box_form_pred_idx(bboxes, idx, num_views):
    """
    Get bounding boxes from idx.

    Args:
        bboxes (torch.Tensor): Bounding boxes in (cx, cy, w, h) format.
            shape #(num_bboxs, num_view, 4)
        idx (torch.Tensor): Bounding boxes idx in (idx in 1st view, idx in 2nd view, idx in 3rd view) format.
            shape #(num_idx, num_view)
    Returns:
        torch.Tensor: Bounding boxes in (cx, cy, w, h) format.
            shape #(num_idx, num_view, 4)
    """
    num_target = idx.size(0) 
    cam_idx = torch.arange(num_views).repeat(num_target, 1).flatten(0,1) #(3, )->(num_target, 3)->(num_target*3, )
    target_idx = idx.flatten(0,1) #(num_target*3, )
    bbox_targets = bboxes[target_idx, cam_idx].reshape(num_target, num_views, -1) #(num_target*3, 4)->(num_target, 3, 4)
    return bbox_targets


