'''
# assign gt id to detection result to evaluate reid performances(ap, fpr-95) which are implemented in ASNet.

input 1. det.json
input 2. gt.json
output. det_with_gt_inst_id.json

for scene_id in scene_ids : 
    for cam_id in cam_ids :
        init graph
        pred_insts = frcnn_json.get_insts(scene_id, cam_id)
        gt_insts = gt_json.get_insts(scene_id, cam_id)
        new_id = max(gt_insts inst id) +1

        edges = []
        for pred_inst in pred_insts : 
            for gt_inst in pred_insts : 
                wgt = iou(pred_inst, gt_inst)
                if wgt : 
                    edge.append(['pred_inst_id', 'gt_inst_id', wgt])
        add edges to graph
        maching = matcing result of graph

        for pred_inst in pred_insts:
            if pred_inst_id in matching :
                inst_id = matching['pred_inst_id']
            else :
                inst_id = new_id 
                new_id += 1
            frcnn_with_gt_inst_id.add_inst(scene_id, cam_id, inst_id, subcls, pos)
'''

from tqdm import tqdm

from .utils import iou
from .bipartite_graph import Bipartite_graph
from .json_maker import json_maker

class GT_ID_ASSIGNER :
    def __init__(self, args):
        self.gt_path = args.gt_path
        self.src_path = args.src_path
        self.dst_path = args.dst_path
        self.test_emb = args.test_emb
        self.num_cam = args.num_cam

    def add_alias_matching(self, matching, pred_insts) :
        alias = dict()
        for pred_inst in pred_insts :
            target_id_in_G = 'pred_'+ pred_inst['inst_id']
            target_pos = pred_inst['pos']
            cur_alias = []
            for pred_inst in pred_insts :
                alias_id_in_G = 'pred_'+ pred_inst['inst_id']
                alias_pos = pred_inst['pos']
                if target_id_in_G != alias_id_in_G and target_pos == alias_pos :
                    cur_alias.append(alias_id_in_G)
            alias[target_id_in_G] = cur_alias
                            
        new_matching = matching.copy()
        for pred_id, gt_id in matching.items() : 
            for alias_id in alias[pred_id] :
                new_matching[alias_id] = gt_id

        return new_matching

    def main(self):
        src_json = json_maker([], self.src_path, 0)
        gt_json = json_maker([], self.gt_path, 0)
        dst_json = json_maker([], self.dst_path, 0)

        src_json.load()
        gt_json.load()

        dst_json.insert_intrinsics(gt_json.get_intrinsics())
        scene_ids = src_json.get_all_scenes()
        for scene_id in tqdm(scene_ids) :
            #G = Bipartite_graph()
            instance_summary = gt_json.get_instance_summary(scene_id) 
            instance_ids = [int(id) for id in instance_summary.keys()]
            new_id = max(instance_ids)+1

            dst_json.insert_scene(scene_id)
            scene = src_json.get_scene(scene_id)
            cam_ids = src_json.get_all_cams(scene)
            for cam_id in cam_ids :
                if int(cam_id) > self.num_cam : break
                G = Bipartite_graph()
                dst_json.insert_cam(scene_id, cam_id)

                src_cam = src_json.get_cam(scene_id, cam_id)
                pred_path, pred_insts = src_json.get_all_inst(src_cam) 

                gt_cam = gt_json.get_cam(scene_id, cam_id)
                gt_path, gt_insts = gt_json.get_all_inst(gt_cam)

                if(pred_path != gt_path):
                    #print('pred_path', pred_path, 'gt_path', gt_path)
                    exit(1)

                dst_json.insert_path(scene_id, cam_id, pred_path)

                extrinsics = gt_json.get_extrinsics(scene_id, cam_id)
                dst_json.insert_extrinsics(scene_id, cam_id, extrinsics)

                edges = []
                for pred_inst in pred_insts :
                    for gt_inst in gt_insts :
                        #pred_inst_pos = [p/resize_ratio for p in pred_inst['pos']]
                        #wgt = utility.iou(pred_inst_pos, gt_inst['pos'])
                        wgt = iou(pred_inst['pos'], gt_inst['pos'])
                        pred_id_in_G = 'pred_' + pred_inst['inst_id']
                        gt_id_in_G = 'gt_' + gt_inst['inst_id']
                        if wgt : edges.append((pred_id_in_G, gt_id_in_G, wgt))
            
                G.add_weighted_edges(edges)
                matching = G.match()
                matching = [sorted([k, v], reverse=True) for (k, v) in matching.items()]
                matching = {l[0]:l[1] for l in matching}
                matching = self.add_alias_matching(matching, pred_insts)

                for pred_inst in pred_insts :
                    pred_id = pred_inst['inst_id']
                    pred_id_in_G = 'pred_' + pred_id
                    subcls = pred_inst['subcls']
                    x1, y1, x2, y2 = pred_inst['pos']
                    prob = pred_inst['prob']
                    if pred_id_in_G in matching :
                        gt_id_in_G = matching[pred_id_in_G]
                        dst_inst_id = gt_id_in_G.split('_')[1]
                    else :
                        dst_inst_id = str(new_id)
                        new_id += 1
                    dst_json.insert_instance(scene_id, cam_id, pred_id, subcls, x1, y1, x2, y2, prob)
                    dst_json.insert_gt_id(scene_id, cam_id, pred_id, dst_inst_id)
                    if self.test_emb :
                        dst_json.insert_emb_dist(scene_id, cam_id, pred_id, pred_inst['emb_dist'])
                        dst_json.insert_is_valid(scene_id, cam_id, pred_id, pred_inst['is_valid'])

                    if not dst_json.is_inst_in_instance_summary(scene_id, pred_id) : 
                        dst_json.insert_instance_summary(scene_id, pred_id, subcls)

        dst_json.sort()
        dst_json.save()
