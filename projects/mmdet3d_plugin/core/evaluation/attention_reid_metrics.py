"""Pair-wise attention ReID analysis.

A sample is a directed pair of valid views of one query, (anchor -> other).
A view is valid when its visibility score is above ``vis_thresh``. Queries are
kept when the classification score and the reid score are above their
thresholds. ``reid_thresh`` is the vla threshold and ``vis_thresh`` is the
per-view visibility threshold.

The instance id of a region proposal is the ground-truth box in that view with
IoU above ``iou_thresh`` and the largest IoU. A pair is correct at K when the
other view's top-K list contains that anchor top-1 id.
"""

import csv
import json
import os

import numpy as np


METHODS = ('attention', 'nearest', 'random')
METHOD_LABELS = {
    'attention': 'Attention',
    'nearest': 'Nearest',
    'random': 'Random',
}
FAILURE_X_MAX = 1000.0
LINE_COLORS = ('#d62728', '#2ca02c', '#1f77b4')
PLOT_DPI = 300


def _use_plot_style(plt):
    """Use Times New Roman, or the closest installed serif."""
    from matplotlib import font_manager
    names = {font.name for font in font_manager.fontManager.ttflist}
    if 'Times New Roman' in names:
        serif = 'Times New Roman'
    elif 'Liberation Serif' in names:
        serif = 'Liberation Serif'
    elif 'Nimbus Roman' in names:
        serif = 'Nimbus Roman'
    else:
        serif = 'DejaVu Serif'
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = [serif, 'DejaVu Serif']
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 15
    plt.rcParams['axes.labelsize'] = 18
    plt.rcParams['axes.titlesize'] = 18
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14
    plt.rcParams['legend.fontsize'] = 15
    plt.rcParams['axes.linewidth'] = 1.2
    plt.rcParams['lines.linewidth'] = 2.4
    plt.rcParams['lines.markersize'] = 8


def _legend(ax, *args, **kwargs):
    kwargs['frameon'] = False
    return ax.legend(*args, **kwargs)


def cxcywh_to_xyxy(boxes):
    """Convert center-size boxes to corner boxes."""
    out = np.empty_like(boxes, dtype=np.float64)
    half_w = boxes[..., 2] * 0.5
    half_h = boxes[..., 3] * 0.5
    out[..., 0] = boxes[..., 0] - half_w
    out[..., 1] = boxes[..., 1] - half_h
    out[..., 2] = boxes[..., 0] + half_w
    out[..., 3] = boxes[..., 1] + half_h
    return out


def box_iou(boxes_a, boxes_b):
    """IoU between two sets of cxcywh boxes. Returns ``(Na, Nb)``."""
    a = cxcywh_to_xyxy(np.asarray(boxes_a, dtype=np.float64))
    b = cxcywh_to_xyxy(np.asarray(boxes_b, dtype=np.float64))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.clip(union, 1e-6, None)


def _match_rps(rp_cxcywh, gt_cxcywh, gt_valid, iou_thresh):
    """Match every RP to a GT instance in each view.

    Returns
    -------
    rp_id : (V, R) int, -1 when no GT has IoU above the threshold
    rp_iou : (V, R) float, IoU with the matched GT, else 0
    """
    num_rpn, num_views, _ = rp_cxcywh.shape
    rp_id = np.full((num_views, num_rpn), -1, dtype=np.int32)
    rp_iou = np.zeros((num_views, num_rpn), dtype=np.float64)
    for view in range(num_views):
        valid = np.asarray(gt_valid[:, view], dtype=bool)
        if not np.any(valid):
            continue
        ious = box_iou(rp_cxcywh[:, view], gt_cxcywh[valid, view])
        best = ious.argmax(axis=1)
        best_iou = ious[np.arange(num_rpn), best]
        global_ids = np.flatnonzero(valid)[best]
        keep = best_iou > iou_thresh
        rp_id[view, keep] = global_ids[keep]
        rp_iou[view, keep] = best_iou[keep]
    return rp_id, rp_iou


def _anchor_distance(query_xy, gt_cxcy, gt_valid, gt_label):
    """Pixel distance from the reference GT to the nearest same-class GT.

    The reference GT is the valid GT whose center is closest to the query
    projection in that view. The distance is infinite when that GT has no
    other box of the same class in the view, and NaN when the view has no
    valid GT.
    """
    num_query, num_views, _ = query_xy.shape
    distances = np.full((num_query, num_views), np.nan, dtype=np.float64)
    for view in range(num_views):
        valid = np.asarray(gt_valid[:, view], dtype=bool)
        if not np.any(valid):
            continue
        centers = gt_cxcy[valid, view, :2]
        labels = gt_label[valid]
        delta = query_xy[:, view, None, :] - centers[None, :, :]
        nearest = np.linalg.norm(delta, axis=-1).argmin(axis=1)
        ref_center = centers[nearest]
        ref_label = labels[nearest]
        ref_delta = ref_center[:, None, :] - centers[None, :, :]
        ref_dist = np.linalg.norm(ref_delta, axis=-1)
        same = labels[None, :] == ref_label[:, None]
        same[np.arange(num_query), nearest] = False
        ref_dist = np.where(same, ref_dist, np.inf)
        distances[:, view] = ref_dist.min(axis=1)
    return distances


def _ranks(attn, query_xy, rp_cxcy, rng):
    """Best-to-worst RP order for each method. Shape ``(method, N, V, R)``."""
    num_query, num_views, num_rpn = attn.shape
    attention_order = np.argsort(-attn, axis=-1, kind='mergesort')
    centers = rp_cxcy[:, :, :2].transpose(1, 0, 2)
    delta = query_xy[:, :, None, :] - centers[None, :, :, :]
    nearest_order = np.argsort(np.linalg.norm(delta, axis=-1), axis=-1, kind='mergesort')
    random_order = np.argsort(rng.random((num_query, num_views, num_rpn)), axis=-1, kind='mergesort')
    return {
        'attention': attention_order,
        'nearest': nearest_order,
        'random': random_order,
    }


def _pair_rank(ordered_ids, anchor, other):
    """1-based rank of the anchor top-1 id in the other view. 0 means absent."""
    query_id = ordered_ids[:, anchor, 0]
    other_ids = ordered_ids[:, other, :]
    matched = other_ids == query_id[:, None]
    matched[query_id < 0] = False
    found = matched.any(axis=-1)
    rank = matched.argmax(axis=-1) + 1
    return np.where(found, rank, 0).astype(np.int16)


class AttentionReIDAnalyzer(object):
    """Accumulate directed view pairs and summarize the four experiments."""

    def __init__(self,
                 cls_thresh=0.1,
                 reid_thresh=0.1,
                 vis_thresh=0.5,
                 iou_thresh=0.5,
                 max_k=20,
                 num_attn_bins=10,
                 num_x_bins=10,
                 random_seed=0):
        self.cls_thresh = cls_thresh
        self.reid_thresh = reid_thresh
        self.vis_thresh = vis_thresh
        self.iou_thresh = iou_thresh
        self.max_k = max_k
        self.num_attn_bins = num_attn_bins
        self.num_x_bins = num_x_bins
        self.random_seed = random_seed
        self._rank = {name: [] for name in METHODS}
        self._iou = {name: [] for name in METHODS}
        self._attn = []
        self._distance = []
        self.num_scenes = 0
        self.num_queries = 0

    def add_scene(self,
                  cls_score,
                  reid_score,
                  vis_score,
                  attn,
                  query_xy,
                  rp_cxcywh,
                  gt_cxcywh,
                  gt_valid,
                  gt_label,
                  scene_index=0):
        """Add one scene.

        Parameters
        ----------
        cls_score, reid_score : (Q,)
        vis_score : (Q, V)
        attn : (Q, V, R) attention probability of each RP
        query_xy : (Q, V, 2) query projection in image pixels
        rp_cxcywh : (R, V, 4)
        gt_cxcywh : (G, V, 4)
        gt_valid : (G, V) bool, True when the GT box exists in that view
        gt_label : (G,)
        """
        cls_score = np.asarray(cls_score, dtype=np.float64)
        reid_score = np.asarray(reid_score, dtype=np.float64)
        vis_score = np.asarray(vis_score, dtype=np.float64)
        attn = np.clip(np.asarray(attn, dtype=np.float64), 0.0, 1.0)
        query_xy = np.asarray(query_xy, dtype=np.float64)
        rp_cxcywh = np.asarray(rp_cxcywh, dtype=np.float64)
        gt_cxcywh = np.asarray(gt_cxcywh, dtype=np.float64)
        gt_valid = np.asarray(gt_valid, dtype=bool)
        gt_label = np.asarray(gt_label)

        keep = (cls_score > self.cls_thresh) & (reid_score > self.reid_thresh)
        if not np.any(keep):
            self.num_scenes += 1
            return
        cls_score = cls_score[keep]
        vis_score = vis_score[keep]
        attn = attn[keep]
        query_xy = query_xy[keep]
        num_query, num_views, _ = attn.shape
        self.num_scenes += 1
        self.num_queries += int(num_query)

        rp_id, rp_iou = _match_rps(rp_cxcywh, gt_cxcywh, gt_valid, self.iou_thresh)
        distance = _anchor_distance(query_xy, gt_cxcywh, gt_valid, gt_label)
        rng = np.random.default_rng(np.random.SeedSequence([self.random_seed, int(scene_index)]))
        orders = _ranks(attn, query_xy, rp_cxcywh, rng)
        valid = vis_score > self.vis_thresh

        ordered = {}
        ordered_iou = {}
        for name, order in orders.items():
            ordered[name] = np.take_along_axis(rp_id[None, :, :], order, axis=-1)
            ordered_iou[name] = np.take_along_axis(rp_iou[None, :, :], order, axis=-1)
        top1_attn = attn.max(axis=-1)

        for anchor in range(num_views):
            for other in range(num_views):
                if anchor == other:
                    continue
                selected = valid[:, anchor] & valid[:, other]
                if not np.any(selected):
                    continue
                self._attn.append(top1_attn[selected, anchor])
                self._distance.append(distance[selected, anchor])
                for name in METHODS:
                    rank = _pair_rank(ordered[name], anchor, other)[selected]
                    pair_iou = 0.5 * (
                        ordered_iou[name][selected, anchor, 0] + ordered_iou[name][selected, other, 0])
                    pair_iou = np.where(rank == 1, pair_iou, np.nan)
                    self._rank[name].append(rank)
                    self._iou[name].append(pair_iou)

    def _concat(self):
        if len(self._attn) == 0:
            empty_i = np.zeros((0,), dtype=np.int16)
            empty_f = np.zeros((0,), dtype=np.float64)
            return (
                {name: empty_i for name in METHODS},
                {name: empty_f for name in METHODS},
                empty_f,
                empty_f,
            )
        rank = {name: np.concatenate(self._rank[name]) for name in METHODS}
        iou = {name: np.concatenate(self._iou[name]) for name in METHODS}
        return rank, iou, np.concatenate(self._attn), np.concatenate(self._distance)

    def summarize(self):
        rank, iou, attn, distance = self._concat()
        summary = {
            'num_scenes': self.num_scenes,
            'num_queries': self.num_queries,
            'num_pairs': int(attn.shape[0]),
            'cls_thresh': self.cls_thresh,
            'reid_thresh': self.reid_thresh,
            'vis_thresh': self.vis_thresh,
            'iou_thresh': self.iou_thresh,
            'association': {},
            'topk_recall': [],
            'calibration': [],
            'failure': {'bins': [], 'infinity': {}},
        }
        for name in METHODS:
            correct = rank[name] == 1
            summary['association'][name] = {
                'label': METHOD_LABELS[name],
                'accuracy': _mean_or_none(correct.astype(np.float64)),
                'iou': _mean_or_none(iou[name][correct]) if np.any(correct) else None,
                'num_pairs': int(rank[name].shape[0]),
                'num_correct': int(correct.sum()),
            }
        for k in range(1, self.max_k + 1):
            row = {'k': k}
            for name in METHODS:
                hit = (rank[name] >= 1) & (rank[name] <= k)
                row[name] = _mean_or_none(hit.astype(np.float64))
            summary['topk_recall'].append(row)

        attn_edges = np.linspace(0.0, 1.0, self.num_attn_bins + 1)
        attn_bin = _digitize(attn, attn_edges)
        attention_rank = rank['attention']
        attention_iou = iou['attention']
        for bin_id in range(self.num_attn_bins):
            selected = attn_bin == bin_id
            correct = selected & (attention_rank == 1)
            summary['calibration'].append({
                'bin': bin_id,
                'low': float(attn_edges[bin_id]),
                'high': float(attn_edges[bin_id + 1]),
                'num_pairs': int(selected.sum()),
                'num_correct': int(correct.sum()),
                'accuracy': _mean_or_none(attention_rank[selected] == 1),
                'iou': _mean_or_none(attention_iou[correct]) if np.any(correct) else None,
            })

        finite = np.isfinite(distance)
        infinite = np.isinf(distance)
        undefined = ~(finite | infinite)
        summary['failure']['num_undefined'] = int(undefined.sum())
        summary['failure']['x_limit'] = FAILURE_X_MAX
        if np.any(finite):
            # Ten equal bins on [0, 1000]. Distances above 1000 stay out of the curve.
            x_edges = np.linspace(0.0, FAILURE_X_MAX, self.num_x_bins + 1)
            in_range = finite & (distance <= FAILURE_X_MAX)
            x_bin = _digitize(np.where(in_range, distance, 0.0), x_edges)
            for bin_id in range(self.num_x_bins):
                selected = in_range & (x_bin == bin_id)
                row = {
                    'bin': bin_id,
                    'low': float(x_edges[bin_id]),
                    'high': float(x_edges[bin_id + 1]),
                    'num_pairs': int(selected.sum()),
                }
                for name in METHODS:
                    row[name] = _mean_or_none(rank[name][selected] == 1)
                summary['failure']['bins'].append(row)
            summary['failure']['x_max'] = float(distance[finite].max())
            summary['failure']['num_above_limit'] = int((finite & (distance > FAILURE_X_MAX)).sum())
        else:
            summary['failure']['x_max'] = None
            summary['failure']['num_above_limit'] = 0
        for name in METHODS:
            selected = infinite
            summary['failure']['infinity'][name] = {
                'accuracy': _mean_or_none(rank[name][selected] == 1),
                'num_pairs': int(selected.sum()),
                'num_correct': int((selected & (rank[name] == 1)).sum()),
            }
        return summary

    def save(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        summary = self.summarize()
        with open(os.path.join(out_dir, 'summary.json'), 'w') as handle:
            json.dump(summary, handle, indent=2)
        _write_association_csv(os.path.join(out_dir, 'association_accuracy.csv'), summary)
        _write_topk_csv(os.path.join(out_dir, 'topk_recall.csv'), summary)
        _write_calibration_csv(os.path.join(out_dir, 'calibration.csv'), summary)
        _write_failure_csv(os.path.join(out_dir, 'failure_cases.csv'), summary)
        _plot_all(out_dir, summary)
        _print_summary(summary)
        return summary


def _digitize(values, edges):
    if len(edges) <= 2 and edges[0] == edges[-1]:
        return np.zeros(values.shape, dtype=np.int64)
    return np.digitize(values, edges[1:-1], right=False)


def _mean_or_none(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return None
    return float(values.mean())


def _write_association_csv(path, summary):
    with open(path, 'w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['method', 'association_accuracy', 'iou', 'num_pairs', 'num_correct'])
        for name in METHODS:
            row = summary['association'][name]
            writer.writerow([row['label'], _fmt(row['accuracy']), _fmt(row['iou']), row['num_pairs'], row['num_correct']])


def _write_topk_csv(path, summary):
    with open(path, 'w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['k'] + list(METHODS))
        for row in summary['topk_recall']:
            writer.writerow([row['k']] + [_fmt(row[name]) for name in METHODS])


def _write_calibration_csv(path, summary):
    with open(path, 'w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['bin', 'low', 'high', 'num_pairs', 'num_correct', 'association_accuracy', 'iou'])
        for row in summary['calibration']:
            writer.writerow([
                row['bin'], row['low'], row['high'], row['num_pairs'], row['num_correct'],
                _fmt(row['accuracy']), _fmt(row['iou']),
            ])


def _write_failure_csv(path, summary):
    with open(path, 'w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['bin', 'low', 'high', 'num_pairs'] + list(METHODS))
        for row in summary['failure']['bins']:
            writer.writerow(
                [row['bin'], row['low'], row['high'], row['num_pairs']] + [_fmt(row[name]) for name in METHODS])
        infinity = summary['failure']['infinity']
        writer.writerow(
            ['inf', 'inf', 'inf', infinity['attention']['num_pairs']] +
            [_fmt(infinity[name]['accuracy']) for name in METHODS])


def _plot_all(out_dir, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    _use_plot_style(plt)
    _plot_association(plt, os.path.join(out_dir, 'association_accuracy.png'), summary)
    _plot_topk(plt, os.path.join(out_dir, 'attention_topk_recall.png'), summary)
    _plot_calibration(plt, os.path.join(out_dir, 'attention_calibration.png'), summary)
    _plot_failure(plt, os.path.join(out_dir, 'attention_failure_cases.png'), summary)
    plt.close('all')


def _plot_association(plt, path, summary):
    _use_plot_style(plt)
    fig, ax = plt.subplots(figsize=(6.4, 2.2))
    ax.axis('off')
    cells = [['Method', 'Association accuracy', 'IoU']]
    for name in METHODS:
        row = summary['association'][name]
        cells.append([row['label'], _fmt(row['accuracy'], digits=3), _fmt(row['iou'], digits=3)])
    table = ax.table(cellText=cells, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)
    for cell in table.get_celld().values():
        cell.get_text().set_fontfamily('serif')
    fig.tight_layout()
    fig.savefig(path, dpi=160)


def _plot_topk(plt, path, summary):
    _use_plot_style(plt)
    fig, ax = plt.subplots(figsize=(6, 4))
    ks = [row['k'] for row in summary['topk_recall']]
    for name, color in zip(METHODS, LINE_COLORS):
        ax.plot(
            ks, [_or_nan(row[name]) for row in summary['topk_recall']],
            marker='o', color=color, label=METHOD_LABELS[name])
    ax.set_xlabel('K')
    ax.set_ylabel('Top-K matching recall')
    ax.set_xticks(ks)
    ax.set_ylim(0, 1)
    _legend(ax, loc='lower right')
    fig.tight_layout()
    fig.savefig(path, dpi=PLOT_DPI)


def _plot_calibration(plt, path, summary):
    _use_plot_style(plt)
    fig, ax = plt.subplots(figsize=(6.4, 4))
    centers, accuracy, iou, counts = [], [], [], []
    for row in summary['calibration']:
        centers.append(0.5 * (row['low'] + row['high']))
        counts.append(row['num_pairs'])
        if row['num_pairs'] == 0:
            accuracy.append(np.nan)
            iou.append(np.nan)
        else:
            accuracy.append(_or_nan(row['accuracy']))
            iou.append(_or_nan(row['iou']))
    total = float(sum(counts))
    proportions = [count / total if total else 0.0 for count in counts]
    count_ax = ax.twinx()
    bars = count_ax.bar(centers, proportions, width=0.08, color='#8c8c8c', zorder=0, label='Proportion of samples')
    count_ax.set_ylabel('Proportion of samples')
    count_ax.set_zorder(0)
    ax.set_zorder(1)
    ax.patch.set_visible(False)
    accuracy_line, = ax.plot(centers, accuracy, marker='o', color=LINE_COLORS[0], zorder=2, label='Association accuracy')
    iou_line, = ax.plot(centers, iou, marker='o', color=LINE_COLORS[1], zorder=2, label='IoU')
    ax.set_xlabel('Attention weight')
    ax.set_ylabel('Association accuracy / IoU')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    _legend(ax, [bars, accuracy_line, iou_line], ['Proportion of samples', 'Association accuracy', 'IoU'], loc='upper left')
    fig.tight_layout()
    fig.savefig(path, dpi=PLOT_DPI)


def _plot_failure(plt, path, summary):
    _use_plot_style(plt)
    fig, axes = plt.subplots(1, 2, figsize=(7.73, 4.8), gridspec_kw={'width_ratios': [4.8, 3.25]})
    bins = summary['failure']['bins']
    centers = [0.5 * (row['low'] + row['high']) for row in bins]
    for name, color in zip(METHODS, LINE_COLORS):
        axes[0].plot(centers, [_or_nan(row[name]) for row in bins], marker='o', color=color, label=METHOD_LABELS[name])
    axes[0].set_xlabel('Pixel distance to the same class', fontsize=20)
    axes[0].set_ylabel('Association accuracy', fontsize=20)
    axes[0].set_xlim(0, FAILURE_X_MAX)
    axes[0].set_xticks(np.arange(0, FAILURE_X_MAX + 1, 200))
    axes[0].set_ylim(0, 1)
    axes[0].tick_params(labelsize=16)
    _legend(axes[0], loc='center right', bbox_to_anchor=(0.98, 0.30), fontsize=16)
    infinity = summary['failure']['infinity']
    axes[1].bar(
        np.arange(len(METHODS)),
        [_or_nan(infinity[name]['accuracy']) for name in METHODS],
        width=0.3,
        color=list(LINE_COLORS))
    axes[1].set_xticks(np.arange(len(METHODS)))
    axes[1].tick_params(labelsize=16)
    axes[1].set_xticklabels(['Attention', 'Nearest', 'Random'], fontsize=16)
    axes[1].set_ylim(0, 1)
    axes[1].set_xlabel('No same class', fontsize=20)
    fig.tight_layout()
    fig.savefig(path, dpi=PLOT_DPI)


def _fmt(value, digits=6):
    if value is None:
        return ''
    return ('{:.%df}' % digits).format(value)


def _or_nan(value):
    return np.nan if value is None else value


def _print_summary(summary):
    print('pairs {}  queries {}  scenes {}'.format(
        summary['num_pairs'], summary['num_queries'], summary['num_scenes']))
    print('{:<20} {:>12} {:>8}'.format('Method', 'Accuracy', 'IoU'))
    for name in METHODS:
        row = summary['association'][name]
        print('{:<20} {:>12} {:>8}'.format(row['label'], _fmt(row['accuracy'], 4), _fmt(row['iou'], 4)))


def run_self_check():
    """Check pair counting, top-K, calibration bins, and the infinite-X group."""
    analyzer = AttentionReIDAnalyzer(max_k=5, num_attn_bins=10, num_x_bins=10, random_seed=0)
    gt = np.zeros((2, 3, 4), dtype=np.float64)
    gt[0, :, 2:] = 10
    gt[1, :, 0] = 100
    gt[1, :, 2:] = 10
    valid = np.ones((2, 3), dtype=bool)
    labels = np.array([0, 0])
    rp = np.zeros((3, 3, 4), dtype=np.float64)
    rp[0, :, 2:] = 10
    rp[1, :, 0] = 100
    rp[1, :, 2:] = 10
    rp[2, :, 0] = 500
    rp[2, :, 1] = 500
    rp[2, :, 2:] = 10
    attn = np.zeros((1, 3, 3), dtype=np.float64)
    attn[0, 0, 0] = 0.8
    attn[0, 0, 1] = 0.2
    attn[0, 1, 0] = 0.7
    attn[0, 1, 1] = 0.3
    attn[0, 2, 1] = 0.6
    attn[0, 2, 0] = 0.4
    query = np.zeros((1, 3, 2), dtype=np.float64)
    analyzer.add_scene(
        np.array([0.9]), np.array([0.9]), np.ones((1, 3)), attn, query, rp, gt, valid, labels, scene_index=0)

    # A second query is invisible in two views, so it contributes no pairs.
    analyzer.add_scene(
        np.array([0.9]), np.array([0.9]), np.array([[0.9, 0.1, 0.1]]),
        attn, query, rp, gt, valid, labels, scene_index=1)
    # A low classification score is dropped.
    analyzer.add_scene(
        np.array([0.05]), np.array([0.9]), np.ones((1, 3)), attn, query, rp, gt, valid, labels, scene_index=2)

    summary = analyzer.summarize()
    attention = summary['association']['attention']
    nearest = summary['association']['nearest']
    assert summary['num_pairs'] == 6, summary['num_pairs']
    failure_bins = summary['failure']['bins']
    assert len(failure_bins) == 10, len(failure_bins)
    assert abs(failure_bins[0]['low'] - 0.0) < 1e-8
    assert abs(failure_bins[-1]['high'] - 1000.0) < 1e-8
    assert failure_bins[1]['num_pairs'] == 6, failure_bins[1]
    assert failure_bins[0]['num_pairs'] == 0
    assert attention['num_correct'] == 2, attention
    assert abs(attention['accuracy'] - 2.0 / 6.0) < 1e-8
    assert abs(attention['iou'] - 1.0) < 1e-8
    assert nearest['num_correct'] == 6
    assert abs(summary['topk_recall'][0]['attention'] - attention['accuracy']) < 1e-8

    # The matching proposal is second in the other view, so recall starts at K=2.
    late = AttentionReIDAnalyzer(max_k=3, random_seed=0)
    late_attn = np.zeros((1, 2, 3), dtype=np.float64)
    late_attn[0, 0, 0] = 0.9
    late_attn[0, 1, 1] = 0.7
    late_attn[0, 1, 0] = 0.3
    two_view_gt = gt[:, :2]
    two_view_rp = rp[:, :2]
    late.add_scene(
        np.array([0.5]), np.array([0.5]), np.ones((1, 2)), late_attn,
        np.zeros((1, 2, 2)), two_view_rp, two_view_gt, np.ones((2, 2), dtype=bool), labels, scene_index=3)
    late_summary = late.summarize()
    pair_recall = [row['attention'] for row in late_summary['topk_recall']]
    # Both directions first meet the anchor id at rank 2.
    assert abs(pair_recall[0] - 0.0) < 1e-8, pair_recall
    assert abs(pair_recall[1] - 1.0) < 1e-8, pair_recall

    # No other box of the same class goes to the infinite bar, not the curve.
    unique = AttentionReIDAnalyzer(num_x_bins=10, random_seed=0)
    one_gt = gt[:1, :2]
    unique_attn = np.zeros((1, 2, 3), dtype=np.float64)
    unique_attn[0, :, 0] = 0.9
    unique_attn[0, :, 1] = 0.1
    unique.add_scene(
        np.array([0.5]), np.array([0.5]), np.ones((1, 2)), unique_attn,
        np.zeros((1, 2, 2)), two_view_rp, one_gt, np.ones((1, 2), dtype=bool), np.array([3]), scene_index=4)
    unique_summary = unique.summarize()
    assert unique_summary['failure']['bins'] == []
    assert unique_summary['failure']['infinity']['attention']['num_pairs'] == 2
    assert abs(unique_summary['failure']['infinity']['attention']['accuracy'] - 1.0) < 1e-8

    # Top-1 weight 0.05 and 0.95 land in the first and last calibration bins.
    bins = AttentionReIDAnalyzer(num_attn_bins=10, random_seed=0)
    low = np.zeros((1, 2, 2), dtype=np.float64)
    low[0, :, 0] = 0.01
    low[0, :, 1] = 0.05
    high = np.zeros((1, 2, 2), dtype=np.float64)
    high[0, :, 0] = 0.95
    high[0, :, 1] = 0.01
    same_gt = np.zeros((1, 2, 4))
    same_gt[..., 2:] = 10
    same_rp = np.zeros((2, 2, 4))
    same_rp[0, :, 2:] = 10
    same_rp[1, :, 0] = 50
    same_rp[1, :, 2:] = 10
    bins.add_scene(
        np.array([0.5]), np.array([0.5]), np.ones((1, 2)), low,
        np.zeros((1, 2, 2)), same_rp, same_gt, np.ones((1, 2), dtype=bool), np.array([1]), scene_index=5)
    bins.add_scene(
        np.array([0.5]), np.array([0.5]), np.ones((1, 2)), high,
        np.zeros((1, 2, 2)), same_rp, same_gt, np.ones((1, 2), dtype=bool), np.array([1]), scene_index=6)
    calibration = bins.summarize()['calibration']
    assert calibration[0]['num_pairs'] == 2, calibration[0]
    assert calibration[-1]['num_pairs'] == 2, calibration[-1]
    assert calibration[0]['iou'] is None
    assert abs(calibration[-1]['accuracy'] - 1.0) < 1e-8
    print('attention reid metric self-check passed')


if __name__ == '__main__':
    run_self_check()
