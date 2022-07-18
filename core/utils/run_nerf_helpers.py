import torch
import numpy as np
from copy import deepcopy

from .skeleton_utils import *


# Misc
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)
def load_ckpt_from_path(ray_caster, optimizer, ckpt_path,
                        finetune=False):
    ckpt = torch.load(ckpt_path)

    global_step = ckpt["global_step"]
    ray_caster.load_state_dict(ckpt)
    if optimizer is not None and not finetune:
        print("load optimizer from ckpt")
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    return global_step, ray_caster, optimizer, ckpt

def filter_state_dict(current_state_dict, state_dict):

    filtered_state_dict = {}
    for local_key in current_state_dict:
        local_val = current_state_dict[local_key]
        try:
            loaded_val = state_dict[local_key]
        except:
            if 'pe_fn' in local_key:
                print('!!!WARNING: temporary fix for A-NeRF pe coef')
                if not 'tau' in local_key:
                    loaded_val = current_state_dict[local_key]
                else:
                    # TODO: assume hard cutoff around threshold.
                    loaded_val = torch.tensor(1000.)
                print(f'{local_key}: {loaded_val}')
        if local_val.shape != loaded_val.shape:
            print(f'!!!WARNING!!!: size mismatch for {local_key}: current model is {local_val.shape} '+
                  f'while the size in ckpt is {loaded_val.shape}')
            print(f'!!!WARNING!!!: Automatically omit loading {local_key}. If this is not intented, stop the program now!')
            if 'framecodes' in local_key:
                print('!!!WARNING: framecode shape different, load mean values!!!')
                mean_val = loaded_val.mean(dim=0, keepdims=True).repeat(local_val.shape[0], 1)
                filtered_state_dict[local_key] = mean_val
        else:
            filtered_state_dict[local_key] = loaded_val

    return filtered_state_dict

# TODO: naming not accurate
def decay_optimizer_lrate(lrate, lrate_decay, decay_rate, optimizer,
                          global_step=None, decay_unit=1000):

    #decay_steps = lrate_decay * decay_unit
    decay_steps = lrate_decay
    optim_step = optimizer.state[optimizer.param_groups[0]['params'][0]]['step'] // decay_unit
    #new_lrate = lrate * (decay_rate ** (global_step / decay_steps))
    new_lrate = lrate * (decay_rate ** (optim_step / decay_steps))
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lrate
    return new_lrate, None

def imgs_to_grid(imgs, nrols=4, ncols=3):
    N, H, W, C = imgs.shape
    n_grids = int(np.ceil(N / (nrols * ncols)))
    grids = np.zeros((n_grids, nrols * H, ncols * W, C), dtype=np.uint8)

    for i, img in enumerate(imgs):
        grid_loc = i // (nrols * ncols)
        col = i % ncols
        row = (i // ncols) % nrols
        grids[grid_loc, row*H:(row+1)*H, col*W:(col+1)*W, :] = img
    return grids

def get_bone_dist(pts, kps, skel_type=SMPLSkeleton):

    joint_trees = torch.tensor(skel_type.joint_trees)
    nonroot_id = torch.tensor(skel_type.nonroot_id)
    root_id = torch.tensor(skel_type.root_id)

    N_rays, N_sampels = pts.shape[:2]
    if pts.dim() == 3:
        pts = pts[:, :, None, :].expand(-1, -1, len(nonroot_id), -1)

    kps_parent = kps[:, joint_trees, :]
    kps_parent = kps_parent[:, None, nonroot_id, :]
    kps_nonroot = kps[:, None, nonroot_id, :]

    dist_to_bone, proj_pts, proj = get_dist_pts_to_lineseg(pts, kps_nonroot, kps_parent)
    return dist_to_bone, proj_pts, proj

def get_dist_pts_to_lineseg(pts, p0, p1):
    """
    pts: query pts
    p0: the 1st endpoint of the lineseg
    p1: the 2nd endpoint of the lineseg
    """

    seg = p1 - p0
    seg_len = torch.norm(seg, dim=-1, p=2)

    vec = pts - p0

    # determine if the pts is in-between p0 and p1.
    # if so, the dist is dist(pts, seg)
    # otherwise it should be dist(p0, pts) or dist(p1, pts)
    dist_p0 = torch.norm(vec, dim=-1, p=2)
    dist_p1 = torch.norm(pts - p1, dim=-1, p=2)

    # unroll it here to save some computes..
    # dist_line = get_dist_pts_to_line(pts, p0, p1)
    cross = torch.cross(vec, seg.expand(*vec.shape), dim=-1)
    dist_line = torch.norm(cross, dim=-1, p=2) / torch.norm(seg, dim=-1, p=2)

    # we can check if it's in-between by projecting vec to seg and check the length/dir
    proj = (vec * seg).sum(-1) / seg_len
    proj_pts = proj[..., None] * (seg / seg_len[..., None]) + p0

    dist = torch.where(
                proj < 0, # case1
                dist_p0,
                torch.where(
                    proj > 1, # case2
                    dist_p1,
                    dist_line # case3
                )
    )

    return dist, proj_pts, proj
