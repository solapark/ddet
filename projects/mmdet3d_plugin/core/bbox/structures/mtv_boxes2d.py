# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch

class MtvBoxes2D(object):
    def __init__(self, gt_ids) :
        self.gt_ids = gt_ids

    def __call__(self) :
        return self.gt_ids

    def __len__(self) :
        return len(self.gt_ids)

    def __repr__(self):
        """str: Return a strings that describes the object."""
        return self.__class__.__name__ + '(\n    ' + str(self.gt_ids) + ')'
