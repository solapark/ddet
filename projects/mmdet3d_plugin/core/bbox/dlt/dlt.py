import torch
from projects.mmdet3d_plugin.core.bbox.builder import BBOX_DLT

def mv_DLT(Ps, pnts):
    # Ps: Tensor of shape (num_cams, 4, 4)
    # pnts: Tensor of shape (num_cams, 2)
    
    pnts = pnts.view(-1, 2, 1)
    first_row = pnts[:, 1] * Ps[:, 2] - Ps[:, 1]
    second_row = Ps[:, 0] - pnts[:, 0] * Ps[:, 2]

    A = torch.cat([first_row, second_row], dim=0).view(-1, 4)
    
    # Compute B = A^T * A
    B = A.t() @ A
    
    # Perform SVD
    U, s, Vh = torch.svd(B)
    
    return Vh[:, -1][0:3] / Vh[:, -1][3]

@BBOX_DLT.register_module()
class DLTv1 :
    def __init__(self, repeat, num_query, num_key, num_view) :
        self.repeat = repeat
        self.num_query = num_query
        self.num_key = num_key
        self.num_view = num_view

    def dlt(self, init_3Dquery, init_2Dquery, rp_cxcy, is_valids, Pmat, max_inds) :
        inst_3dp_list = []
        for i in range(self.num_query) :
            init_cam_idx = i//self.num_key
            cam_inds = [init_cam_idx]
            Ps = [Pmat[init_cam_idx]]
            pnts = [init_2Dquery[i]]
            for j in range(self.num_view) : 
                if not is_valids[i,j] : continue
                cam_inds.append(init_cam_idx)
                Ps.append(Pmat[j])
                pnts.append(rp_cxcy[i,j])

            if len(cam_inds) == 1 :
                inst_3dp = init_3Dquery[i]
            elif len(cam_inds) == 2 and init_cam_idx == cam_inds[1] :
                new_inds = max_inds[i, init_cam_idx]
                inst_3dp =  init_3Dquery[init_cam_idx*self.num_key + new_inds]
            else : 
                Ps = torch.stack(Ps, 0)
                pnts = torch.cat(pnts, 0)
                inst_3dp = mv_DLT(Ps, pnts) #(3,)

            inst_3dp_list.append(inst_3dp)
        return torch.stack(inst_3dp_list, 0)



