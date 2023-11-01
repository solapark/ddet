import os 
from .json_maker import json_maker 
from .dict2class import Dict2Class
from .gt_id_assigner import GT_ID_ASSIGNER

CLASS_DICT = {'water1': 0, 'water2': 1, 'pepsi': 2, 'coca1': 3, 'coca2': 4, 'coca3': 5, 'coca4': 6, 'tea1': 7, 'tea2': 8, 'yogurt': 9, 'ramen1': 10, 'ramen2': 11, 'ramen3': 12, 'ramen4': 13, 'ramen5': 14, 'ramen6': 15, 'ramen7': 16, 'juice1': 17, 'juice2': 18, 'can1': 19, 'can2': 20, 'can3': 21, 'can4': 22, 'can5': 23, 'can6': 24, 'can7': 25, 'can8': 26, 'can9': 27, 'ham1': 28, 'ham2': 29, 'pack1': 30, 'pack2': 31, 'pack3': 32, 'pack4': 33, 'pack5': 34, 'pack6': 35, 'snack1': 36, 'snack2': 37, 'snack3': 38, 'snack4': 39, 'snack5': 40, 'snack6': 41, 'snack7': 42, 'snack8': 43, 'snack9': 44, 'snack10': 45, 'snack11': 46, 'snack12': 47, 'snack13': 48, 'snack14': 49, 'snack15': 50, 'snack16': 51, 'snack17': 52, 'snack18': 53, 'snack19': 54, 'snack20': 55, 'snack21': 56, 'snack22': 57, 'snack23': 58, 'snack24': 59, 'green_apple': 60, 'red_apple': 61, 'tangerine': 62, 'lime': 63, 'lemon': 64, 'yellow_quince': 65, 'green_quince': 66, 'white_quince': 67, 'fruit1': 68, 'fruit2': 69, 'peach': 70, 'banana': 71, 'fruit3': 72, 'pineapple': 73, 'fruit4': 74, 'strawberry': 75, 'cherry': 76, 'red_pimento': 77, 'green_pimento': 78, 'carrot': 79, 'cabbage1': 80, 'cabbage2': 81, 'eggplant': 82, 'bread': 83, 'baguette': 84, 'sandwich': 85, 'hamburger': 86, 'hotdog': 87, 'donuts': 88, 'cake': 89, 'onion': 90, 'marshmallow': 91, 'mooncake': 92, 'shirimpsushi': 93, 'sushi1': 94, 'sushi2': 95, 'big_spoon': 96, 'small_spoon': 97, 'fork': 98, 'knife': 99, 'big_plate': 100, 'small_plate': 101, 'bowl': 102, 'white_ricebowl': 103, 'blue_ricebowl': 104, 'black_ricebowl': 105, 'green_ricebowl': 106, 'black_mug': 107, 'gray_mug': 108, 'pink_mug': 109, 'green_mug': 110, 'blue_mug': 111, 'blue_cup': 112, 'orange_cup': 113, 'yellow_cup': 114, 'big_wineglass': 115, 'small_wineglass': 116, 'glass1': 117, 'glass2': 118, 'glass3': 119}

class Json_writer :
    def __init__ (self, args) :
        self.jm = json_maker([], args.result_json_path, 0)
        self.jm.path = args.result_json_path
        self.class_mapping = args.class_mapping
        self.args = args

    def write(self, all_dets, image_paths):
        img_paths = list(image_paths[0].values())

        for cam_idx, filepath in enumerate(img_paths):
            filename_with_ext = os.path.basename(filepath)
            filename = filename_with_ext.split('.')[0]
            filename_split = filename.split('-')
            cam_num = str(int(filename_split[-1]))
            scene_num = '-'.join(filename_split[:-1])

            if not self.jm.is_scene_in_tree(scene_num) :
                self.jm.insert_scene(scene_num)

            if not self.jm.is_cam_in_scene(scene_num, cam_num):
                self.jm.insert_cam(scene_num, cam_num)
                self.jm.insert_path(scene_num, cam_num, filename_with_ext)

            dets = all_dets[cam_idx]
            for det in dets:
                x1, y1, x2, y2, reid_prob, prob, cls, inst_idx = det['x1'], det['y1'], det['x2'], det['y2'], det['reid_prob'], det['prob'], det['class'], det['inst_idx']

                x1, y1, x2, y2 = [round(p/self.args.resize_ratio) for p in [x1, y1, x2, y2]]
                cls_idx = self.class_mapping[cls]
                self.jm.insert_instance(scene_num, cam_num, inst_idx, cls_idx, x1, y1, x2, y2, prob)
                self.jm.insert_instance_summary(scene_num, inst_idx, cls_idx)
                self.jm.insert_reid_prob(scene_num, cam_num, inst_idx, reid_prob)

                if self.args.write_is_valid :
                    #self.jm.insert_is_valid(scene_num, cam_num, inst_idx, det['is_valid'])
                    self.jm.insert_is_valid(scene_num, cam_num, inst_idx, 1)

                if self.args.write_emb_dist :
                    self.jm.insert_emb_dist(scene_num, cam_num, inst_idx, det['emb_dist'])

    def close(self):
        self.jm.sort()
        self.jm.save()

class Json_saver :
    def __init__(self, img_dir, json_path, num_views, gt_path):
        self.img_dir = img_dir
        self.json_path = json_path
        json_wirter_args = {'result_json_path': json_path, 
                            'class_mapping' : CLASS_DICT,
                            'resize_ratio' : 1.,
                            'write_is_valid' : True,
                            'write_emb_dist' : False
                            }
        json_wirter_args = Dict2Class(json_wirter_args)
        self.json_writer = Json_writer(json_wirter_args)

        gt_id_assigner_args = {'gt_path' : gt_path,
                            'src_path' : json_path,
                            'dst_path' : json_path,
                            'test_emb' : False,
                            'num_cam' : num_views
                            }
        gt_id_assigner_args = Dict2Class(gt_id_assigner_args)
        self.gt_id_assigner = GT_ID_ASSIGNER(gt_id_assigner_args)


    def add_data(self, scene_id, data):
        image_paths = {str(i+1) : os.path.join(self.img_dir, '%s-%02d.jpg'%(scene_id, i+1)) for i in range(len(data))}
        self.json_writer.write(data, [image_paths])

    def close(self):
        self.json_writer.close()
        print('Results are saved in %s'%(self.json_path))
        #self.gt_id_assigner.main()
        #print('Gt ids are assigned.')


