import torch
import torch.nn.functional as F
from projects.mmdet3d_plugin.core.bbox.builder import BBOX_DLT

def mv_DLT(Ps, pnts, weights=None):
    # Ps: Tensor of shape (num_cams, 4, 4)
    # pnts: Tensor of shape (num_cams, 2)
    
    pnts = pnts.view(-1, 2, 1)
    first_row = pnts[:, 1] * Ps[:, 2] - Ps[:, 1]
    second_row = Ps[:, 0] - pnts[:, 0] * Ps[:, 2]

    A = torch.cat([first_row, second_row], dim=0).view(-1, 4)
    
    if weights is not None:
        weights = weights.view(-1, 1).repeat(2, 1).view(-1, 1)  # Apply the weights to each pair of rows
        A = A * weights

    # Compute B = A^T * A
    B = A.t() @ A
    
    # Perform SVD
    U, s, Vh = torch.svd(B)
    
    return Vh[:, -1][0:3] / Vh[:, -1][3]

@BBOX_DLT.register_module()
class DLTv0 :
    def __init__(self, repeat, num_query, num_key, num_view, weights=None) :
        self.repeat = repeat
        self.num_query = num_query
        self.num_key = num_key
        self.num_view = num_view

    def dlt(self, init_3Dquery, init_2Dquery, rp_cxcy, is_valids, cls_scores, Pmat, max_inds) :
        inst_3dp_list = []
        for i in range(self.num_query) :
            init_cam_idx = i//self.num_key
            cam_inds = [init_cam_idx]
            Ps = [Pmat[init_cam_idx]]
            pnts = [init_2Dquery[i, init_cam_idx]]
            for j in range(self.num_view) : 
                if not is_valids[i,j] : continue
                cam_inds.append(j)
                Ps.append(Pmat[j])
                pnts.append(rp_cxcy[i,j])

            Ps = torch.stack(Ps, 0)
            pnts = torch.cat(pnts, 0)
            inst_3dp = mv_DLT(Ps, pnts) #(3,)
            inst_3dp_list.append(inst_3dp)
        return torch.stack(inst_3dp_list, 0)

@BBOX_DLT.register_module()
class DLTv1 :
    def __init__(self, repeat, num_query, num_key, num_view, weights=None) :
        self.repeat = repeat
        self.num_query = num_query
        self.num_key = num_key
        self.num_view = num_view

    def dlt(self, init_3Dquery, init_2Dquery, rp_cxcy, is_valids, cls_scores, Pmat, max_inds) :
        inst_3dp_list = []
        for i in range(self.num_query) :
            init_cam_idx = i//self.num_key
            cam_inds = [init_cam_idx]
            Ps = [Pmat[init_cam_idx]]
            pnts = [init_2Dquery[i, init_cam_idx]]
            for j in range(self.num_view) : 
                if not is_valids[i,j] : continue
                cam_inds.append(j)
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

@BBOX_DLT.register_module()
class DLTv2 :
    def __init__(self, repeat, num_query, num_key, num_view, weights=[1.0, 1.0]) :
        self.repeat = repeat
        self.num_query = num_query
        self.num_key = num_key
        self.num_view = num_view
        self.weights = weights

    def dlt(self, init_3Dquery, init_2Dquery, rp_cxcy, is_valids, cls_scores, Pmat, max_inds) :
        inst_3dp_list = []
        for i in range(self.num_query) :
            cam_inds = []
            Ps = []
            pnts = []
            weights = []
            for j in range(self.num_view) : 
                cam_inds.append(j)
                Ps.append(Pmat[j])
                pnts.append(init_2Dquery[i,j])
                weights.append(self.weights[0])

                if not is_valids[i,j] : continue
                cam_inds.append(j)
                Ps.append(Pmat[j])
                pnts.append(rp_cxcy[i,j])
                weights.append(self.weights[1])

            Ps = torch.stack(Ps, 0)
            pnts = torch.cat(pnts, 0)
            weights = torch.tensor(weights, device=pnts.device)
            inst_3dp = mv_DLT(Ps, pnts, weights) #(3,)

            inst_3dp_list.append(inst_3dp)
        return torch.stack(inst_3dp_list, 0)@BBOX_DLT.register_module()

@BBOX_DLT.register_module()
class DLTv3 :
    def __init__(self, repeat, num_query, num_key, num_view, weights=[1.0, 1.0], thresh=.9) :
        self.repeat = repeat
        self.num_query = num_query
        self.num_key = num_key
        self.num_view = num_view
        self.weights = weights
        self.thresh = thresh

    def dlt(self, init_3Dquery, init_2Dquery, rp_cxcy, is_valids, cls_scores, Pmat, max_inds) :
        inst_3dp_list = []
        soft_cls_scores, labels = F.softmax(cls_scores, dim=-1).max(-1) #(900,3), #(900,3)
        max_view_soft_cls_scores, max_view = soft_cls_scores.max(-1) #(900,), #(900,)

        for i in range(self.num_query) :
            cam_inds = []
            Ps = []
            pnts = []
            weights = []
            for j in range(self.num_view) : 
                cam_inds.append(j)
                Ps.append(Pmat[j])
                pnts.append(init_2Dquery[i,j])
                weights.append(self.weights[0])

            scores = soft_cls_scores[i]
            scores[labels[i]!=labels[i, max_view[i]]]=0  #(3,)
            is_low = scores<self.thresh
            scores[is_low] = 0.
            if is_low.sum() == self.num_view : 
                scores[max_view[i]] = 1.
            scores = scores/scores.sum()
            for j in range(self.num_view) : 
                if(scores[j]==0.0) : continue
                cam_inds.append(j)
                Ps.append(Pmat[j])
                pnts.append(rp_cxcy[i,j])
                weights.append(self.weights[1]*scores[j])

            Ps = torch.stack(Ps, 0)
            pnts = torch.cat(pnts, 0)
            weights = torch.tensor(weights, device=pnts.device)
            inst_3dp = mv_DLT(Ps, pnts, weights) #(3,)

            inst_3dp_list.append(inst_3dp)
        return torch.stack(inst_3dp_list, 0)

@BBOX_DLT.register_module()
class DLTv4 :
    def __init__(self, repeat, num_query, num_key, num_view, weights=[1.0, 2.0], thresh=.9) :
        self.repeat = repeat
        self.num_query = num_query
        self.num_key = num_key
        self.num_view = num_view
        self.weights = weights
        self.thresh = thresh

    def dlt(self, init_3Dquery, init_2Dquery, rp_cxcy, is_valids, cls_scores, Pmat, max_inds) :
        inst_3dp_list = []
        soft_cls_scores, labels = F.softmax(cls_scores, dim=-1).max(-1) #(900,3), #(900,3)
        max_view_soft_cls_scores, max_view = soft_cls_scores.max(-1) #(900,), #(900,)

        for i in range(self.num_query) :
            cam_inds = []
            Ps = []
            pnts = []
            weights = []
            for j in range(self.num_view) : 
                cam_inds.append(j)
                Ps.append(Pmat[j])
                pnts.append(init_2Dquery[i,j])
                weights.append(self.weights[0])

            if max_view_soft_cls_scores[i] > self.thresh :
                max_j = max_view[i]
                pnts[max_j]=rp_cxcy[i,max_j]
                weights[max_j]=self.weights[1]

            Ps = torch.stack(Ps, 0)
            pnts = torch.cat(pnts, 0)
            weights = torch.tensor(weights, device=pnts.device)
            inst_3dp = mv_DLT(Ps, pnts, weights) #(3,)

            inst_3dp_list.append(inst_3dp)
        return torch.stack(inst_3dp_list, 0)

@BBOX_DLT.register_module()
class DLTv5 :
    def __init__(self, repeat, num_query, num_key, num_view, weights=[1.0, 2.0], thresh=.9, acc_thresh=.5) :
        self.repeat = repeat
        self.num_query = num_query
        self.num_key = num_key
        self.num_view = num_view
        self.weights = weights
        self.thresh = thresh
        self.acc_thresh = acc_thresh

    def dlt(self, init_3Dquery, init_2Dquery, rp_cxcy, rp_cxcy_cam, is_valids, cls_scores, Pmat, max_inds, img_metas) :
        max_cls_scores, labels = cls_scores.max(-1) #(900,3), #(900,3)
        max_view_cls_scores, max_view = max_cls_scores.max(-1) #(900,), #(900,)
        idx = torch.arange(self.num_query, device=cls_scores.device)
        max_view_label = labels[idx, max_view]
        final_cls_scores = cls_scores.transpose(-1, -2) #(900, 120, 3)
        final_cls_scores = [final_cls_scores[i, max_view_label[i]].sigmoid() for i in range(len(final_cls_scores))] #(900, 3)
        final_cls_scores = torch.stack(final_cls_scores) #(900, 3)

        acc = 1. - self.calc_epipolar_errors(max_view, rp_cxcy_cam, img_metas) #(900, 3) 0~1
        inst_3dp_list = []
        for i in range(self.num_query) :
            cam_inds = []
            Ps = []
            pnts = []
            weights = []
            for j in range(self.num_view) : 
                if final_cls_scores[i,j] > self.thresh and acc[i,j] > self.acc_thresh:
                    cam_inds.append(j)
                    Ps.append(Pmat[j])
                    pnts.append(rp_cxcy[i,j])
                    weights.append(acc[i,j])

            if len(cam_inds) >= 2 :
                Ps = torch.stack(Ps, 0)
                pnts = torch.cat(pnts, 0)
                weights = torch.tensor(weights, device=pnts.device)
                inst_3dp = mv_DLT(Ps, pnts, weights) #(3,)
            else :
                inst_3dp = init_3Dquery[i] #(3,)

            inst_3dp_list.append(inst_3dp)
        return torch.stack(inst_3dp_list, 0)

    def check_epipolar_constraint(self, cam_points1, cam_points2, F):
        """
        Check if the epipolar constraint is satisfied between two sets of camera points.
        """
        M, _ = cam_points1.shape  # Points, 3

        # Reshape for matrix multiplication
        cam_points1_homog = torch.cat([cam_points1, torch.ones_like(cam_points1[:, :1])], dim=-1)  # (M, 4)
        cam_points2_homog = torch.cat([cam_points2, torch.ones_like(cam_points2[:, :1])], dim=-1)  # (M, 4)
        
        # Perform the epipolar constraint check
        errors = []
        for i in range(M) : 
            error = torch.abs(torch.matmul(cam_points2_homog[i:i+1], torch.matmul(F[i], cam_points1_homog[i:i+1].t()))).squeeze()
            errors.append(error)
        errors = cam_points1.new_tensor(errors)
            
        #error = torch.abs(torch.matmul(cam_points2_homog, torch.matmul(F, cam_points1_homog.t()))).diagonal()  # (M, )
        
        return errors

    def calc_epipolar_errors(self, max_view, cam_points, img_metas): 
        """
        Calculate epipolar errors for all views relative to the max_view.
        
        cam_points: Tensor of shape (V, M, 3)
        img_metas: List of dicts containing 'intrinsics' and 'dec_extrinsics'
        
        Returns:
        all_errors: Tensor of shape (M, V)
        """
        extrinsics = cam_points.new_tensor([img_meta['dec_extrinsics'] for img_meta in img_metas])[0]
        intrinsics = cam_points.new_tensor([img_meta['intrinsics'] for img_meta in img_metas])[0]

        M, V, _ = cam_points.shape
        F_matrices = torch.zeros((V, V, 3, 3), device=cam_points.device)
        
        for i in range(V):
            for j in range(V): 
                K1_inv = torch.inverse(intrinsics[i, :3, :3])
                K2_inv = torch.inverse(intrinsics[j, :3, :3])
                R = extrinsics[j, :3, :3] @ extrinsics[i, :3, :3].t()
                T = extrinsics[j, :3, 3] - R @ extrinsics[i, :3, 3]

                T_skew = torch.tensor([[0, -T[2], T[1]],
                                       [T[2], 0, -T[0]],
                                       [-T[1], T[0], 0]], device=T.device)

                E = T_skew @ R
                F = K2_inv.t() @ E @ K1_inv
                F_matrices[i, j] = F
                F_matrices[j, i] = F.t()  # F_ji is the transpose of F_ij

        idx = torch.arange(M, device=cam_points.device)
        max_view_pnts = cam_points[idx, max_view]
        all_errors = []
        for i in range(V) :  
            errors = self.check_epipolar_constraint(max_view_pnts, cam_points[:, i], F_matrices[max_view, i])
            all_errors.append(errors)
        all_errors = torch.stack(all_errors, 1)  # (M, V)
        all_errors[idx, max_view] = 0
        all_errors = all_errors/(all_errors.max(-1, keepdim=True)[0]+1e-6)
        return all_errors

