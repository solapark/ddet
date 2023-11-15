from .gt_id_assigner import GT_ID_ASSIGNER
from .evaluator import Evaluator

class Reid_evaluator :
    def __init__(self, json_path, gt_path, num_cam, config_dir):
        self.json_path = json_path
        self.gt_id_assigner = GT_ID_ASSIGNER(gt_path, json_path, num_cam)
        self.evaluator = Evaluator(config_dir, json_path) 

    def main(self):
        print('reid json path...')
        print('evaluating reid performance...')
        #result = self.get_result()
        #1. assign gt_id
        self.gt_id_assigner.main()
        res = self.evaluator.main()
        print('evaluating reid performance done')
        return res
