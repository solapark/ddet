"""Four ReID experiments from attention, nearest, and random det.json files.

A sample is a directed view pair. A view is valid when that camera was written
for the query, which is the visibility >= 0.5 gate used to build det.json.
Queries also need classification score > 0.1 and reid score > 0.1.

The RP order stored in each file is that method's ranking. An RP's instance id
is the GT instance in the same view with IoU > 0.5 and the largest IoU.

Example:
    python tools/PAA_rebuttal/run_detjson_reid_analysis.py \
        --iou-sweep \
        --out-dir result/detjson_reid_analysis_iou
"""

import argparse
import importlib.util
import json
import os
import time

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
METHODS = ('attention', 'nearest', 'random')


def load_metrics():
    path = os.path.join(ROOT, 'projects/mmdet3d_plugin/core/evaluation/attention_reid_metrics.py')
    spec = importlib.util.spec_from_file_location('attention_reid_metrics', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def iter_scenes(path):
    """Yield ``(scene_id, {camera: {instance_id: info}})`` from a det or GT json."""
    scene_id = None
    cameras = {}
    cam = None
    in_instances = False
    collecting = False
    depth = 0
    buf = []
    inst_id = None
    with open(path) as handle:
        for line in handle:
            if not collecting and line.startswith('        "') and line.rstrip().endswith(': {'):
                if scene_id is not None:
                    yield scene_id, cameras
                scene_id = line.strip()[1:].split('"', 1)[0]
                cameras = {}
                cam = None
                in_instances = False
                continue
            if scene_id is None:
                continue
            if not collecting and line.startswith('                "') and line.rstrip().endswith(': {'):
                key = line.strip()[1:].split('"', 1)[0]
                if key.isdigit():
                    cam = key
                    cameras[cam] = {}
                    in_instances = False
                continue
            if cam is None:
                continue
            if not collecting and line.strip().startswith('"instances"'):
                in_instances = True
                continue
            if not in_instances:
                continue
            if not collecting and line.startswith('                        "') and line.rstrip().endswith(': {'):
                inst_id = line.strip()[1:].split('"', 1)[0]
                buf = ['{']
                collecting = True
                depth = 1
                continue
            if collecting:
                buf.append(line)
                depth += line.count('{') - line.count('}')
                if depth == 0:
                    text = ''.join(buf).rstrip()
                    if text.endswith(','):
                        text = text[:-1]
                    cameras[cam][inst_id] = json.loads(text)
                    collecting = False
    if scene_id is not None:
        yield scene_id, cameras


def box_iou(boxes, gt_boxes):
    """IoU of xyxy boxes. ``boxes`` is (N, 4) and ``gt_boxes`` is (G, 4)."""
    boxes = np.asarray(boxes, dtype=np.float64)
    gt_boxes = np.asarray(gt_boxes, dtype=np.float64)
    lt = np.maximum(boxes[:, None, :2], gt_boxes[None, :, :2])
    rb = np.minimum(boxes[:, None, 2:], gt_boxes[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = np.clip(boxes[:, 2] - boxes[:, 0], 0, None) * np.clip(boxes[:, 3] - boxes[:, 1], 0, None)
    area_b = np.clip(gt_boxes[:, 2] - gt_boxes[:, 0], 0, None) * np.clip(gt_boxes[:, 3] - gt_boxes[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.clip(union, 1e-6, None)


def match_ids(boxes, gt_boxes, gt_ids):
    """Return the best GT id and its IoU for each box. Id is None when the view has no GT."""
    count = len(boxes)
    ids = [None] * count
    ious = np.zeros(count, dtype=np.float64)
    if count == 0 or len(gt_boxes) == 0:
        return ids, ious
    overlap = box_iou(boxes, gt_boxes)
    best = overlap.argmax(axis=1)
    best_iou = overlap[np.arange(count), best]
    for index, gt_index in enumerate(best):
        ids[index] = gt_ids[gt_index]
        ious[index] = best_iou[index]
    return ids, ious


def ids_above(ids, ious, iou_thresh):
    """Drop matches whose IoU does not exceed the threshold."""
    return [inst_id if inst_id is not None and iou > iou_thresh else None for inst_id, iou in zip(ids, ious)]


def anchor_distance(query_xy, gt_items):
    """Distance from the reference GT to the nearest other GT of the same class.

    The reference GT is the one whose center is closest to the query projection.
    The result is inf when that GT has no same-class neighbor, and NaN when the
    view has no GT.
    """
    if not gt_items:
        return np.nan
    centers = np.asarray([item[3] for item in gt_items], dtype=np.float64)
    labels = np.asarray([item[1] for item in gt_items])
    nearest = np.linalg.norm(centers - np.asarray(query_xy, dtype=np.float64), axis=1).argmin()
    same = labels == labels[nearest]
    same[nearest] = False
    if not np.any(same):
        return np.inf
    return float(np.linalg.norm(centers[same] - centers[nearest], axis=1).min())


def gt_view(cam):
    items = []
    for inst_id, info in cam.items():
        pos = [float(value) for value in info['pos']]
        center = (0.5 * (pos[0] + pos[2]), 0.5 * (pos[1] + pos[3]))
        items.append((str(inst_id), int(info['subcls']), pos, center))
    return items


def det_view(cam):
    slim = {}
    for inst_id, info in cam.items():
        boxes = [[float(value) for value in item['pos']] for item in info['topk']]
        slim[str(inst_id)] = {
            'prob': float(info['prob']),
            'reid': float(info['reid_prob']),
            'query': [float(value) for value in info['query']],
            'attn': float(info['attn_prob']),
            'boxes': boxes,
        }
    return slim


def rank_of(query_id, other_ids):
    if query_id is None:
        return 0
    for index, other_id in enumerate(other_ids):
        if other_id == query_id:
            return index + 1
    return 0


def add_scene(analyzers, det_views, gt_cams, cls_thresh, reid_thresh):
    """Add one scene to every IoU-threshold analyzer.

    ``analyzers`` is a list of ``(iou_thresh, analyzer)``. Matching keeps the
    best GT and its IoU, then each threshold drops matches that do not exceed it.
    """
    attention = det_views['attention']
    cameras = sorted(attention, key=int)
    gt = {cam: gt_view(gt_cams.get(cam, {})) for cam in cameras}
    matched = {name: {} for name in METHODS}
    for name in METHODS:
        for cam in cameras:
            cam_dets = det_views[name].get(cam, {})
            if not cam_dets:
                matched[name][cam] = {}
                continue
            order = list(cam_dets)
            boxes = []
            spans = []
            for inst_id in order:
                inst_boxes = cam_dets[inst_id]['boxes']
                spans.append((inst_id, len(boxes), len(boxes) + len(inst_boxes)))
                boxes.extend(inst_boxes)
            gt_boxes = [item[2] for item in gt[cam]]
            gt_ids = [item[0] for item in gt[cam]]
            ids, ious = match_ids(boxes, gt_boxes, gt_ids)
            matched[name][cam] = {
                inst_id: (ids[start:stop], ious[start:stop]) for inst_id, start, stop in spans
            }

    queries = set()
    for cam in cameras:
        queries.update(attention[cam])
    kept = 0
    attns = []
    distances = []
    ranks = {(thresh, name): [] for thresh, _ in analyzers for name in METHODS}
    pair_ious = {(thresh, name): [] for thresh, _ in analyzers for name in METHODS}
    for inst_id in queries:
        views = [cam for cam in cameras if inst_id in attention[cam]]
        views = [
            cam for cam in views
            if all(inst_id in det_views[name].get(cam, {}) for name in METHODS)
        ]
        if not views:
            continue
        sample = attention[views[0]][inst_id]
        if not (sample['prob'] > cls_thresh and sample['reid'] > reid_thresh):
            continue
        if len(views) < 2:
            continue
        kept += 1
        for anchor in views:
            for other in views:
                if anchor == other:
                    continue
                anchor_det = attention[anchor][inst_id]
                attns.append(anchor_det['attn'])
                distances.append(anchor_distance(anchor_det['query'], gt[anchor]))
                for thresh, _ in analyzers:
                    for name in METHODS:
                        anchor_ids, anchor_ious = matched[name][anchor][inst_id]
                        other_ids, other_ious = matched[name][other][inst_id]
                        anchor_kept = ids_above(anchor_ids, anchor_ious, thresh)
                        other_kept = ids_above(other_ids, other_ious, thresh)
                        query_id = anchor_kept[0] if anchor_kept else None
                        rank = rank_of(query_id, other_kept)
                        if rank == 1:
                            pair_iou = 0.5 * (float(anchor_ious[0]) + float(other_ious[0]))
                        else:
                            pair_iou = np.nan
                        ranks[(thresh, name)].append(rank)
                        pair_ious[(thresh, name)].append(pair_iou)
    if attns:
        attn_array = np.asarray(attns, dtype=np.float64)
        distance_array = np.asarray(distances, dtype=np.float64)
        for thresh, analyzer in analyzers:
            analyzer._attn.append(attn_array)
            analyzer._distance.append(distance_array)
            for name in METHODS:
                analyzer._rank[name].append(np.asarray(ranks[(thresh, name)], dtype=np.int16))
                analyzer._iou[name].append(np.asarray(pair_ious[(thresh, name)], dtype=np.float64))
    for _, analyzer in analyzers:
        analyzer.num_scenes += 1
        analyzer.num_queries += kept


def default_paths():
    result = os.path.join(ROOT, 'result')
    return {
        'attention': os.path.join(result, 'tmvreid_messytable_rpn59_reproduce_261019/20261025_202936/det.json'),
        'nearest': os.path.join(result, 'tmvreid_messytable_rpn59_reproduce_261019_nearest/20261026_111859/det.json'),
        'random': os.path.join(result, 'tmvreid_messytable_rpn59_reproduce_261019_random/20261026_112913/det.json'),
        'gt': '/home/sapark/VEDet_org/data/MessyTable/labels/test.json',
        'out': os.path.join(result, 'detjson_reid_analysis_iou'),
    }


def parse_args():
    defaults = default_paths()
    parser = argparse.ArgumentParser(description='ReID analysis from three det.json files')
    parser.add_argument('--attention', default=defaults['attention'])
    parser.add_argument('--nearest', default=defaults['nearest'])
    parser.add_argument('--random', default=defaults['random'])
    parser.add_argument('--gt', default=defaults['gt'])
    parser.add_argument('--out-dir', default=defaults['out'])
    parser.add_argument('--cls-thresh', type=float, default=0.1)
    parser.add_argument('--reid-thresh', type=float, default=0.1)
    parser.add_argument('--iou-thresh', type=float, nargs='+', default=[0.5],
                        help='one or more IoU thresholds; a match must strictly exceed the value')
    parser.add_argument('--iou-sweep', action='store_true',
                        help='evaluate IoU thresholds 0, 0.1, ..., 1.0')
    parser.add_argument('--max-k', type=int, default=20)
    parser.add_argument('--max-scenes', type=int, default=0, help='stop after this many scenes; 0 means all')
    return parser.parse_args()


def iou_dirname(iou_thresh):
    return 'iou_%s' % ('%.1f' % iou_thresh)


def main():
    args = parse_args()
    if args.iou_sweep:
        args.iou_thresh = [index / 10.0 for index in range(11)]
    metrics = load_metrics()
    analyzers = []
    for iou_thresh in args.iou_thresh:
        analyzers.append((iou_thresh, metrics.AttentionReIDAnalyzer(
            cls_thresh=args.cls_thresh,
            reid_thresh=args.reid_thresh,
            iou_thresh=iou_thresh,
            max_k=args.max_k,
        )))
    print('reading GT', args.gt, flush=True)
    with open(args.gt) as handle:
        gt_scenes = json.load(handle)['scenes']
    print('GT scenes', len(gt_scenes), flush=True)

    streams = {
        'attention': iter_scenes(args.attention),
        'nearest': iter_scenes(args.nearest),
        'random': iter_scenes(args.random),
    }
    started = time.time()
    scene_count = 0
    while True:
        if args.max_scenes and scene_count >= args.max_scenes:
            break
        loaded = {}
        done = False
        for name in METHODS:
            try:
                loaded[name] = next(streams[name])
            except StopIteration:
                done = True
                break
        if done:
            break
        scene_id = loaded['attention'][0]
        for name in METHODS[1:]:
            if loaded[name][0] != scene_id:
                raise RuntimeError('scene mismatch: %s has %s, %s has %s' % (
                    'attention', scene_id, name, loaded[name][0]))
        if scene_id not in gt_scenes:
            raise RuntimeError('GT has no scene %s' % scene_id)
        det_views = {name: {cam: det_view(cams) for cam, cams in loaded[name][1].items()} for name in METHODS}
        gt_cams = {
            cam: camera.get('instances', {})
            for cam, camera in gt_scenes[scene_id]['cameras'].items()
        }
        add_scene(
            analyzers,
            det_views,
            gt_cams,
            args.cls_thresh,
            args.reid_thresh,
        )
        scene_count += 1
        if scene_count % 20 == 0:
            elapsed = time.time() - started
            done_analyzer = analyzers[0][1]
            print('scenes %d  queries %d  %.1fs' % (
                done_analyzer.num_scenes, done_analyzer.num_queries, elapsed), flush=True)
    done_analyzer = analyzers[0][1]
    print('scenes %d  queries %d  %.1fs' % (
        done_analyzer.num_scenes, done_analyzer.num_queries, time.time() - started), flush=True)
    multiple = len(analyzers) > 1
    for iou_thresh, analyzer in analyzers:
        out_dir = os.path.join(args.out_dir, iou_dirname(iou_thresh)) if multiple else args.out_dir
        print('saving IoU > %.1f to %s' % (iou_thresh, out_dir), flush=True)
        analyzer.save(out_dir)
    print('wrote', args.out_dir, flush=True)


if __name__ == '__main__':
    main()
