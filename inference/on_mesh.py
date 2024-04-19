import trimesh
import torch
import numpy as np
import models.Net as mod
from utils.model_sl import load_model
from utils.hungarian import hungarian
from utils.evaluation_metric import calcorrespondpc
from utils.config import cfg_from_file
from utils.config import cfg
from tqdm import tqdm
from scipy.spatial.transform import Rotation
from models.correspondSlover import SVDslover, RANSACSVDslover

def caliters_perm(model, P1_gt_copy, P2_gt_copy, 
                  A1_gt, A2_gt, n1_gt, n2_gt, 
                  estimate_iters, use_ransac:bool=False):
    lap_solver1 = hungarian
    s_perm_indexs = []
    for estimate_iter in tqdm(range(estimate_iters)):
        s_prem_i, Inlier_src_pre, Inlier_ref_pre = model(P1_gt_copy, P2_gt_copy,
                                                         A1_gt, A2_gt, n1_gt, n2_gt)
        s_perm_i_mat = lap_solver1(s_prem_i, n1_gt, n2_gt, Inlier_src_pre, Inlier_ref_pre)
        P2_gt_copy1, s_perm_i_mat_index = calcorrespondpc(s_perm_i_mat, P2_gt_copy)
        s_perm_indexs.append(s_perm_i_mat_index)
        if use_ransac:
            R_pre, T_pre, s_perm_i_mat = RANSACSVDslover(P1_gt_copy[:,:,:3], 
                                                         P2_gt_copy1[:,:,:3], 
                                                         s_perm_i_mat)
        else:
            R_pre, T_pre = SVDslover(P1_gt_copy[:,:,:3], P2_gt_copy1[:,:,:3], 
                                     s_perm_i_mat)
        
        P1_gt_copy[:,:,:3] = torch.bmm(P1_gt_copy[:,:,:3], 
                                       R_pre.transpose(2, 1).contiguous()) + T_pre[:, None, :]
        P1_gt_copy[:,:,3:6] = P1_gt_copy[:,:,3:6] @ R_pre.transpose(-1, -2)
    return s_perm_i_mat

def infer_on_mesh(x_mesh, y_mesh, model, 
                  x_n: int = 2048,
                  y_n: int = 2048,
                  eval_cycle:bool=True, 
                  estimate_iters:int=1,
                  use_ransac:bool=False):
    lap_solver = hungarian
    with torch.set_grad_enabled(False):
        n1_gt, n2_gt = torch.tensor([x_n]).cuda(), torch.tensor([y_n]).cuda()
        x_pc, x_idx = x_mesh.sample(n1_gt, return_index=True)
        P1_gt = torch.as_tensor(np.concatenate([x_pc.copy(), 
                                                x_mesh.face_normals[x_idx]].copy(), 
                                                axis=-1)).cuda().unsqueeze(0).type(torch.float32)
        y_pc, y_idx = y_mesh.sample(n2_gt, return_index=True)
        P2_gt = torch.as_tensor(np.concatenate([y_pc.copy(), 
                                                y_mesh.face_normals[y_idx]].copy(), 
                                                axis=-1)).cuda().unsqueeze(0).type(torch.float32)
        # fully connected graph
        A1_gt, A2_gt = torch.eye(n1_gt).cuda(), torch.eye(n2_gt).cuda()


        # Model inference
        if estimate_iters > 1:
            # duplicate tensors
            P1_gt_copy = P1_gt.clone()
            P2_gt_copy = P2_gt.clone()
            P1_gt_copy_inv = P1_gt.clone()
            P2_gt_copy_inv = P2_gt.clone()
            s_perm_mat = caliters_perm(model, P1_gt_copy, P2_gt_copy, A1_gt, A2_gt, n1_gt, n2_gt, estimate_iters, use_ransac)
            if eval_cycle:
                s_perm_mat_inv = caliters_perm(model, P2_gt_copy_inv, P1_gt_copy_inv, A2_gt, A1_gt, n2_gt, n1_gt, estimate_iters)
                s_perm_mat = s_perm_mat * s_perm_mat_inv.permute(0, 2, 1)
        else:
            s_prem_tensor, Inlier_src_pre, Inlier_ref_pre_tensor = model(P1_gt, P2_gt, A1_gt, A2_gt, n1_gt, n2_gt)

            s_perm_mat = lap_solver(s_prem_tensor, n1_gt, n2_gt, Inlier_src_pre, Inlier_ref_pre_tensor)
            if eval_cycle:
                s_prem_tensor_inv, Inlier_src_pre_inv, Inlier_ref_pre_tensor_inv = model(P2_gt, P1_gt, A2_gt, A1_gt, n2_gt, n1_gt)
                s_perm_mat_inv = lap_solver(s_prem_tensor_inv, n2_gt, n1_gt, Inlier_src_pre_inv, Inlier_ref_pre_tensor_inv)
                s_perm_mat = s_perm_mat * s_perm_mat_inv.permute(0, 2, 1)


        # Calculate R and T
        pre_P2_gt, _ = calcorrespondpc(s_perm_mat, P2_gt[..., :3])
        R_pre, T_pre = SVDslover(P1_gt[..., :3].clone(), pre_P2_gt, s_perm_mat)

    R_pre, T_pre = R_pre.cpu().numpy(), T_pre.cpu().numpy()
    return R_pre, T_pre


if __name__ == '__main__':
    # load config
    cfg_path = 'experiments/test_RGM_Seen_Crop_modelnet40_transformer.yaml'
    cfg_from_file(cfg_path)

    # load model
    model = mod.Net()
    model_path = "output/RGM_DGCNN_ModelNet40Seen_NoPreW['xyz', 'gxyz']_attentiontransformer_crop/params/params_best.pt"
    load_model(model, model_path)
    model.eval()
    model = model.to('cuda:0' if torch.cuda.is_available() else 'cpu')

    # random rotation
    x_mesh_gt = trimesh.load('~/ws/registration_data_sample/bitescan.ply')
    y_mesh = trimesh.load('~/ws/registration_data_sample/lower_gt.ply')
    r_gt = Rotation.from_euler('xyz', np.random.randint(1, 45, 3), degrees=True)
    x_mesh = x_mesh_gt.copy()
    x_mesh.vertices = (r_gt.as_matrix() @ x_mesh.vertices.T).T
    print(f'GT rotated by {r_gt.magnitude() * 180 / np.pi:.2f} degrees')

    R, T = infer_on_mesh(x_mesh, x_mesh_gt, 
                         model,
                         x_n=2048,
                         y_n=2048,
                         eval_cycle=True, 
                         estimate_iters=100, 
                         use_ransac=False)

    x_mesh_copy = x_mesh.copy()
    y_mesh_copy = y_mesh.copy()
    x_mesh_copy.vertices = (R[0] @ x_mesh_copy.vertices.T).T + T[0]
    x_mesh_copy.visual.face_colors = [1.,0.,0.,1.]
