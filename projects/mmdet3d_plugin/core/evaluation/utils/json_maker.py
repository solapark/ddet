import json
import copy

class json_maker :
    def __init__(self, scenes, path, num_cam) :
        self.path = path
        self.num_cam = num_cam
        self.json = {}
        self.json['intrinsics'] = {}
        self.json['scenes'] = {}
        for scene in scenes :
            self.json['scenes'][scene] = {'instance_summary' : {}, 'cameras' : {}}
            for cam in range(1, self.num_cam+1) :
                pathname = image_base_path % (cam-1, scene)
                self.json['scenes'][scene]['cameras'][str(cam)] = {'pathname' : pathname, 'extrinsics' : {}, 'corners' : {}, 'instances' : {}}

    def load(self) :
        with open(self.path) as json_file:
            self.json = json.load(json_file)

    def sort(self) :
        self.json['scenes'] = dict(sorted(self.json['scenes'].items(), key=lambda item : item[0]))
        for scene in self.json['scenes']:
            self.json['scenes'][scene]['instance_summary'] = dict(sorted(self.json['scenes'][scene]['instance_summary'].items(), key=lambda item : int(item[0])))
            self.json['scenes'][scene]['cameras'] = dict(sorted(self.json['scenes'][scene]['cameras'].items(), key=lambda item : int(item[0])))
            for cam_str in self.json['scenes'][scene]['cameras'].keys() :
                self.json['scenes'][scene]['cameras'][cam_str]['instances'] = dict(sorted(self.json['scenes'][scene]['cameras'][cam_str]['instances'].items(), key=lambda item : int(item[0])))
                

    def is_scene_in_tree(self, scene) :
        if scene in self.json['scenes'] : return True
        else : return False

    def is_cam_in_scene(self, scene_num, cam_num) :
        if cam_num in self.json['scenes'][scene_num]['cameras'] : return True
        else : return False

    def is_inst_in_cam(self, scene, cam_idx, inst_id):
        if inst_id in self.json['scenes'][scene]['cameras'][cam_idx]['instances'] : return True
        else : return False

    def is_inst_in_instance_summary(self, scene, inst):
        if inst in self.json['scenes'][scene]['instance_summary'] : return True
        else : return False

    def is_prob_in_instance(self, scene_name, cam_idx, inst_id):
        if 'prob' in self.json['scenes'][scene_name]['cameras'][cam_idx]['instances'][inst_id] : return True
        else : return False

    def get_cls_in_instance_summary(self, scene, inst):
        return self.json['scenes'][scene]['instance_summary'][inst]

    def get_instance_summary(self, scene_num):
        return self.json['scenes'][scene_num]['instance_summary']

    def insert_intrinsics(self, intrinsics):
        self.json['intrinsics'] = intrinsics
 
    def get_intrinsics(self):
        return self.json['intrinsics']
 
    def insert_scene(self, scene_num, scene=None):
        if(scene):
            self.json['scenes'][scene_num] = scene
        else :
            self.json['scenes'][scene_num] = {'instance_summary' : {}, 'cameras' : {}}
        
    def insert_instance_summary(self, scene, inst, cls) :
        self.json['scenes'][scene]['instance_summary'][inst] = cls

    def insert_instance(self, scene, cam, inst, cls, x1, y1, x2, y2, prob=None) :
        x1, y1, x2, y2 = list(map(float, [x1, y1, x2, y2]))
        prob = float(prob) if prob else None
        self.json['scenes'][scene]['cameras'][cam]['instances'][inst] = {'subcls' : cls, 'pos': [x1, y1, x2, y2], 'prob':prob}

    def insert_emb_dist(self, scene, cam, inst, emb_dist) :
        self.json['scenes'][scene]['cameras'][cam]['instances'][inst]['emb_dist'] = emb_dist

    def insert_is_valid(self, scene, cam, inst, is_valid) :
        self.json['scenes'][scene]['cameras'][cam]['instances'][inst]['is_valid'] = int(is_valid)

    def insert_gt_id(self, scene, cam, inst, gt_id) :
        self.json['scenes'][scene]['cameras'][cam]['instances'][inst]['gt_id'] = gt_id

    def insert_pred_id(self, scene, cam, inst, pred_id) :
        self.json['scenes'][scene]['cameras'][cam]['instances'][inst]['pred_id'] = pred_id

    def change_prob_in_cam(self, scene, cam, inst, prob) :
        self.json['scenes'][scene]['cameras'][cam]['instances'][inst]['prob'] = prob

    def change_cls_in_cam(self, scene, cam, inst, cls) :
        self.json['scenes'][scene]['cameras'][cam]['instances'][inst]['subcls'] = cls

    def insert_cam(self, scene, cam) :
        self.json['scenes'][scene]['cameras'][cam] = {'pathname' : None, 'extrinsics':None, 'instances':{}}

    def insert_path(self, scene, cam, path) :
        self.json['scenes'][scene]['cameras'][cam]['pathname'] = path

    def insert_extrinsics(self, scene, cam, extrinsics) :
        self.json['scenes'][scene]['cameras'][cam]['extrinsics'] = extrinsics

    def get_extrinsics(self, scene, cam) :
        return self.json['scenes'][scene]['cameras'][cam]['extrinsics'] 

    def insert_corners(self, scene, cam, corners) :
        self.json['scenes'][scene]['cameras'][cam]['corners'] = corners

    def get_corners(self, scene, cam) :
        return self.json['scenes'][scene]['cameras'][cam]['corners'] 

    def get_scene(self, scene_num):
        return self.json['scenes'][scene_num]

    def get_all_scenes(self):
        return list(self.json['scenes'].keys())

    def get_all_cams(self, scene):
        return list(scene['cameras'].keys())

    def get_cam(self, scene_num, cam_num):
        return self.json['scenes'][scene_num]['cameras'][cam_num]

    def get_inst_cls(self, scene_name, cam_idx, inst_id):
        return self.json['scenes'][scene_name]['cameras'][cam_idx]['instances'][inst_id]['subcls']

    def get_inst_box(self, scene_name, cam_idx, inst_id):
        return self.json['scenes'][scene_name]['cameras'][cam_idx]['instances'][inst_id]['pos']

    def get_inst_prob(self, scene_name, cam_idx, inst_id):
        return self.json['scenes'][scene_name]['cameras'][cam_idx]['instances'][inst_id]['prob']

    def get_all_insts_in_cam(self, scene_id, cam_id):
        return self.json['scenes'][scene_id]['cameras'][cam_id]['instances']

    def get_all_inst(self, cam, thresh=0):
        path_name = cam['pathname']
        instances = cam['instances']
        inst_dict_list = []
        for inst_id, inst_info in instances.items():
            '''
            subcls = inst_info['subcls']
            pos = inst_info['pos']
            prob= inst_info['prob'] if 'prob' in inst_info else 1
            if prob < thresh :
                continue
            inst_dict = {'inst_id' : inst_id, 'subcls' : subcls, 'pos' : pos, 'prob':prob}
            '''

            inst_dict = copy.deepcopy(inst_info)
            if 'prob' not in inst_dict : 
                inst_dict['prob'] =  1

            if inst_dict['prob'] < thresh :
                continue

            inst_dict['inst_id'] = inst_id


            inst_dict_list.append(inst_dict)
        return path_name, inst_dict_list

    def get_number_of_insts_in_instance_summary(self, scene_num):
        return len(self.get_instance_summary(scene_num))

    def get_number_of_insts_in_cam(self, scene_num, cam_num):
        return len(self.json['scenes'][scene_num]['cameras'][cam_num]['instances'])

    def reset_instance_summary(self, scene_name):
        self.json['scenes'][scene_name]['instance_summary'] = {}

    def reset_instances(self, scene_name):
        for cam_idx in range(self.num_cam) :
            cam_idx = str(cam_idx+1)
            self.json['scenes'][scene_name]['cameras'][cam_idx]['instances'] = {}

    def print(self) :
        print(json.dumps(self.json, indent=4))

    def save(self) :
        with open(self.path, 'w') as fp:
            json.dump(self.json, fp, indent=4)

    def open(self) :
        with open(self.path) as fp:
            self.json = json.load(self.json)
 
