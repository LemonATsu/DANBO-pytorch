import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .utils.skeleton_utils import rot6d_to_rotmat, axisang_to_rot, axisang_to_quat, calculate_angle, \
                                  SMPLSkeleton, get_children_joints, axisang_to_rot6d
from copy import deepcopy
from .cutoff_embedder import get_embedder # TODO: rename this

def get_pts_embedder(args, data_attrs):
    '''
    Get an embedder that encodes all the data for you (except P.E.)
    '''
    embed_dims = {}
    skel_type = data_attrs['skel_type']

    pts_tr_fn = get_pts_tr_fn(args)
    ray_tr_fn = get_ray_tr_fn(args)
    kp_input_fn, input_dims, cutoff_dims = get_kp_input_fn(args, skel_type)
    bone_input_fn, bone_dims = get_bone_input_fn(args, skel_type)
    view_input_fn, view_dims = get_view_input_fn(args, skel_type)

    embed_dims['input_dims'] = input_dims
    embed_dims['cutoff_dims'] = cutoff_dims
    embed_dims['bone_dims'] = bone_dims
    embed_dims['view_dims'] = view_dims

    graph_input_fn = None
    if args.nerf_type in ['graph', 'danbo', 'danbo3d']:
        graph_input_fn, graph_dims = get_graph_input_fn(args, skel_type)
        embed_dims['graph_dims'] = graph_dims

    print(f'PPE: {pts_tr_fn.encoder_name}, KPE: {kp_input_fn.encoder_name},' +
          f'BPE: {bone_input_fn.encoder_name}, VPE: {view_input_fn.encoder_name}')
    
    pts_embedder = SamplePointsEmbedder(pts_tr_fn, ray_tr_fn, 
                                        kp_input_fn=kp_input_fn,
                                        bone_input_fn=bone_input_fn, 
                                        view_input_fn=view_input_fn,
                                        graph_input_fn=graph_input_fn,
                                        skel_type=skel_type)
    
    return pts_embedder, embed_dims

def get_pe_embedder(args, data_attrs, embed_dims):

    network_chs = {}
    network_pe_fns = {}

    input_dims = embed_dims['input_dims']
    cutoff_dims = embed_dims['cutoff_dims']
    bone_dims = embed_dims['bone_dims']
    view_dims = embed_dims['view_dims']
    graph_dims = embed_dims.get('graph_dims', None)

    skel_type = data_attrs["skel_type"]

    cutoff_kwargs = {
        "cutoff": args.use_cutoff,
        "normalize_cutoff": args.normalize_cutoff,
        "cutoff_dist": args.cutoff_mm * args.ext_scale,
        "cutoff_inputs": args.cutoff_inputs,
        "opt_cutoff": args.opt_cutoff,
        "cutoff_dim": cutoff_dims,
        "dist_inputs":  not(input_dims == cutoff_dims),
    }

    dist_cutoff_kwargs = deepcopy(cutoff_kwargs)
    dist_cutoff_kwargs['cut_to_cutoff'] = args.cut_to_dist
    dist_cutoff_kwargs['shift_inputs'] = args.cutoff_shift
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed,
                                      input_dims=input_dims,
                                      skel_type=skel_type,
                                      freq_schedule=args.freq_schedule,
                                      init_alpha=args.init_freq,
                                      cutoff_kwargs=dist_cutoff_kwargs)
    network_chs['input_ch'] = input_ch
    network_pe_fns['pe_fn'] = embed_fn


    # PE for bones (joint angles) 
    input_ch_bones = bone_dims
    if args.cutoff_bones:
        bone_cutoff_kwargs = deepcopy(cutoff_kwargs)
        bone_cutoff_kwargs["dist_inputs"] = True
    else:
        bone_cutoff_kwargs = {"cutoff": False}

    embedbones_fn, input_ch_bones = get_embedder(args.multires_bones, args.i_embed,
                                                input_dims=bone_dims,
                                                skel_type=skel_type,
                                                freq_schedule=args.freq_schedule,
                                                init_alpha=args.init_freq,
                                                cutoff_kwargs=bone_cutoff_kwargs)
    network_chs['input_ch_bones'] = input_ch_bones
    network_pe_fns['bones_pe_fn'] = embedbones_fn

    # PE for view direction
    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        if args.cutoff_viewdir:
            view_cutoff_kwargs = deepcopy(cutoff_kwargs)
            view_cutoff_kwargs["dist_inputs"] = True
        else:
            view_cutoff_kwargs = {"cutoff": False}
        view_cutoff_kwargs["cutoff_dim"] = len(skel_type.joint_trees)
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed,
                                                    input_dims=view_dims,
                                                    skel_type=skel_type,
                                                    freq_schedule=args.freq_schedule,
                                                    init_alpha=args.init_freq,
                                                    cutoff_kwargs=view_cutoff_kwargs)
    network_chs['input_ch_views'] = input_ch_views
    network_pe_fns['dirs_pe_fn'] = embeddirs_fn 

    # PE for Graph
    input_ch_graph, input_ch_voxel = 0, 0
    embedgraph_fn, embedvoxel_fn = None, None
    if args.nerf_type in ['graph', 'danbo', 'danbo3d']:
        graph_dims = embed_dims['graph_dims']
        embedgraph_fn, input_ch_graph = get_embedder(args.multires_graph, args.i_embed,
                                                     input_dims=graph_dims,
                                                     freq_schedule=args.freq_schedule,
                                                     init_alpha=args.init_freq,
                                                     skel_type=skel_type)

        # TODO: full of hacks, fix them!
        voxel_input_dims = args.voxel_feat
        if args.gnn_backbone.endswith('cat'):
            voxel_input_dims *= 3
            if args.gnn_concat:
                voxel_input_dims = voxel_input_dims * len(skel_type.joint_names) 

        if args.gnn_backbone.startswith('CoordCat'):
            voxel_input_dims = args.voxel_feat + 3
        if args.input_coords:
            voxel_input_dims = 3
        if args.cat_coords:
            voxel_input_dims += 3
        if args.cat_all:
            voxel_input_dims = voxel_input_dims * len(skel_type.joint_trees)
        if args.nerf_type in ['danbo3d']:
            voxel_input_dims -= 1

        pe_input_dims = voxel_input_dims

        embedvoxel_fn, input_ch_voxel = get_embedder(args.multires_voxel, args.i_embed,
                                                     input_dims=pe_input_dims,
                                                     freq_schedule=args.freq_schedule,
                                                     init_alpha=args.init_freq,
                                                     skel_type=skel_type)
        network_chs['input_ch_graph'] = input_ch_graph
        network_chs['input_ch_voxel'] = input_ch_voxel

        network_pe_fns['graph_pe_fn'] = embedgraph_fn
        network_pe_fns['voxel_pe_fn'] = embedvoxel_fn


    output_ch = 5 if args.N_importance > 0 else 4
    network_chs['output_ch'] = output_ch

    return network_pe_fns, network_chs

 

def get_pts_tr_fn(args):

    if args.pts_tr_type == 'local':
        tr_fn = WorldToLocalEncoder()
    elif args.pts_tr_type == 'bone':
        tr_fn = WorldToBoneEncoder()
    elif args.pts_tr_type == 'bone_er':
        tr_fn = WorldToBoneEncoder(local_root=True)
    else:
        raise NotImplementedError(f'Point transformation {args.pts_tr_type} is undefined.')
    return tr_fn

def get_ray_tr_fn(args):

    if args.ray_tr_type == 'local':
        tr_fn = transform_batch_rays
    elif args.ray_tr_type == 'root_local':
        tr_fn = RootLocalEncoder()
    elif args.ray_tr_type == 'world':
        tr_fn = lambda rays_o, rays_d, *args, **kwargs: rays_d
    else:
        raise NotImplementedError(f'Ray transformation {args.ray_tr_type} is undefined.')
    return tr_fn


def get_kp_input_fn(args, skel_type):

    kp_input_fn = None
    cutoff_dims = len(skel_type.joint_names)
    N_joints = len(skel_type.joint_names)

    if args.kp_dist_type == 'reldist':
        kp_input_fn = RelDistEncoder(N_joints, skel_type=skel_type)
    elif args.kp_dist_type == 'cat':
        kp_input_fn = KPCatEncoder(N_joints, skel_type=skel_type)
    elif args.kp_dist_type == 'relpos':
        kp_input_fn = RelPosEncoder(N_joints, skel_type=skel_type)
    elif args.kp_dist_type == 'querypts':
        kp_input_fn = IdentityEncoder(1, 3)
        cutoff_dims = 3
    elif args.kp_dist_type == 'bonedist':
        kp_input_fn = BoneDistEncoder(N_joints, skel_type=skel_type)
    elif args.kp_dist_type == 'bonedistchild':
        kp_input_fn = BoneDistChildEncoder(N_joints, skel_type=skel_type)
    else:
        raise NotImplementedError(f'{args.kp_dist_type} is not implemented.')
    input_dims = kp_input_fn.dims

    return kp_input_fn, input_dims, cutoff_dims

def get_view_input_fn(args, skel_type):

    N_joints = len(skel_type.joint_names)

    if args.view_type == "relray":
        view_input_fn = VecNormEncoder(N_joints, skel_type=skel_type)
    elif args.view_type == "rayangle":
        view_input_fn = RayAngEncoder(N_joints, skel_type=skel_type)
    elif args.view_type == "world":
        view_input_fn = IdentityExpandEncoder(N_joints, 3, skel_type=skel_type)
    elif args.view_type == 'identity':
        view_input_fn = IdentityEncoder(N_joints, 3, skel_type=skel_type)
    else:
        raise NotImplementedError(f'{args.view_type} is not implemented.')
    view_dims = view_input_fn.dims

    return view_input_fn, view_dims

def get_bone_input_fn(args, skel_type):

    bone_input_fn = None
    N_joints = len(skel_type.joint_names)

    if args.bone_type.startswith('proj') and args.pts_tr_type == 'bone':
        print(f'bone_type=={args.bone_type} is unnecessary when pts_tr_type==bone.' + \
              f'Use bone_type==dir instead')

    if args.bone_type == 'reldir':
        bone_input_fn = VecNormEncoder(N_joints, skel_type=skel_type)
    elif args.bone_type == 'axisang':
        bone_input_fn = IdentityExpandEncoder(N_joints, 3, skel_type=skel_type)
    elif args.bone_type == 'Nope':
        bone_input_fn = EmptyEncoder(N_joints, skel_type=skel_type)
    else:
        raise NotImplementedError(f'{args.bone_type} bone function is not implemented')
    bone_dims = bone_input_fn.dims

    return bone_input_fn, bone_dims

def get_graph_input_fn(args, skel_type):

    N_joints = len(skel_type.joint_names)
    graph_input_fn = None

    if args.nerf_type.startswith('graphrot'):
        print(f'forcefully rewrite the graph_input_type for {args.nerf_type}')
        graph_input_fn = PartRotateEncoder(N_joints, 3, part_dims=args.part_dims, skel_type=skel_type)
        dims = graph_input_fn.dims
    elif args.graph_input_type == 'quat':
        graph_input_fn = lambda bones, *args, **kwargs: axisang_to_quat(bones)
        dims = 4
    elif args.graph_input_type == 'rot6d':
        graph_input_fn = AxisAngtoRot6DEncoder(N_joints, 3, skel_type)
        dims = graph_input_fn.dims
    elif args.graph_input_type == 'reldir':
        graph_input_fn = RelBoneDirEncoder(N_joints, 3, skel_type=skel_type)
        dims = 3
    elif args.graph_input_type == 'jbonedir':
        graph_input_fn = JointBoneDirEncoder(N_dims=3)
        dims = graph_input_fn.dims
    elif args.graph_input_type == 'enc':
        assert args.multires_graph == 0
        graph_input_fn = lambda bones, *args, **kwargs: axisang_to_rot(bones).flatten(start_dim=-2)
        dims = 9

    return graph_input_fn, dims




# SKT (skeleton transformation-related)
def transform_batch_pts(pts, skt):

    N_rays, N_samples = pts.shape[:2]
    NJ = skt.shape[-3]

    if skt.shape[0] < pts.shape[0]:
        skt = skt.expand(pts.shape[0], *skt.shape[1:])

    # make it from (N_rays, N_samples, 4) to (N_rays, NJ, 4, N_samples)
    pts = torch.cat([pts, torch.ones(*pts.shape[:-1], 1)], dim=-1)
    pts = pts.view(N_rays, -1, N_samples, 4).expand(-1, NJ, -1, -1).transpose(3, 2).contiguous()
    # MM: (N_rays, NJ, 4, 4) x (N_rays, NJ, 4, N_samples) -> (N_rays, NJ, 4, N_samples)
    # permute back to (N_rays, N_samples, NJ, 4)
    mm = (skt @ pts).permute(0, 3, 1, 2).contiguous()

    return mm[..., :-1] # don't need the homogeneous part

def transform_batch_rays(rays_o, rays_d, skt):

    # apply only the rotational part
    N_rays, N_samples = rays_d.shape[:2]
    NJ = skt.shape[-3]
    rot = skt[..., :3, :3]

    if rot.shape[0] < rays_d.shape[0]:
        rot = rot.expand(rays_d.shape[0], *rot.shape[1:])
    rays_d = rays_d.view(N_rays, -1, N_samples, 3).expand(-1, NJ, -1, -1).transpose(3, 2).contiguous()
    mm = (rot @ rays_d).permute(0, 3, 1, 2).contiguous()

    return mm

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

    dist = torch.where(
                proj < 0, # case1
                dist_p0,
                torch.where(
                    proj > 1, # case2
                    dist_p1,
                    dist_line # case3
                )
    )

    return dist

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

    dist_to_bone = get_dist_pts_to_lineseg(pts, kps_nonroot, kps_parent)
    return dist_to_bone

def get_bone_dist_child(pts, kps, children, mask, skel_type=SMPLSkeleton, eps=1e-8):
    '''
    mask: joints with no child (or multiple children)    
    '''

    joint_trees = torch.tensor(skel_type.joint_trees)
    children = children

    N_rays, N_sampels = pts.shape[:2]
    if pts.dim() == 3:
        pts = pts[:, :, None, :].expand(-1, -1, len(joint_trees), -1)

    # perturb for case without child or (multiple children)
    kps_child = kps[:, children, :] + (1. - mask.reshape(1, len(joint_trees), 1)) * 1e-5

    dist_to_bone = get_dist_pts_to_lineseg(pts, kps_child[:, None], kps[:, None])
    return dist_to_bone

class SamplePointsEmbedder(nn.Module):

    def __init__(self, 
                 pts_tr_fn, 
                 ray_tr_fn, 
                 kp_input_fn=None,
                 bone_input_fn=None,
                 view_input_fn=None,
                 graph_input_fn=None,
                 skel_type=SMPLSkeleton):
        super().__init__()

        self.pts_tr_fn = pts_tr_fn
        self.ray_tr_fn = ray_tr_fn

        self.kp_input_fn = kp_input_fn
        self.bone_input_fn = bone_input_fn
        self.view_input_fn = view_input_fn
        self.graph_input_fn = graph_input_fn

        self.skel_type = skel_type
    
    def forward(self, *args, fwd_type='pts', **kwargs):
        if fwd_type == 'pts':
            return self.encode_pts(*args, **kwargs)
        elif fwd_type == 'view':
            return self.encode_views(*args, **kwargs)
        elif fwd_type == 'graph_inputs':
            return self.encode_graph_inputs(*args, **kwargs)
        else:
            raise NotImplementedError(f'Encoding for {fwd_type} is not implemented!')
    
    def encode_pts(self, pts, kps, skts, bones, 
                   align_transforms=None, rest_pose=None):
        '''
        align_transforms: transformation to make pts_t aligned with certain axis
        '''
        
        if pts.shape[0] > kps.shape[0]:
            # expand kps to match the number of rays
            assert kps.shape[0] == 1
            kps = kps.expand(pts.shape[0], *kps.shape[1:])

        pts_t = self.pts_tr_fn(pts, skts, bones=bones, kps=kps)

        pts_rp = None
        if rest_pose is not None:
            rest_pose = rest_pose.reshape(-1, 1, *rest_pose.shape[-2:])
            pts_rp = pts_t + rest_pose

        if align_transforms is not None:
            pts_t = (align_transforms[..., :3, :3] @ pts_t[..., None]).squeeze(-1) \
                        + align_transforms[..., :3, -1]

        # pts-related relative encoding
        v = self.kp_input_fn(pts, pts_t, kps)
        r = self.bone_input_fn(pts_t, bones=bones)

        return {'v': v, 'r': r, 'pts_rp': pts_rp, 'pts_t': pts_t}
    
    def encode_views(self, rays_o, rays_d, skts, refs=None):
        '''
        refs: reference tensors for expanding d
        '''
        rays_t = self.ray_tr_fn(rays_o, rays_d, skts)
        d = self.view_input_fn(rays_t, refs=refs)
        return {'d': d}
    
    def encode_graph_inputs(self, kps, bones, skts, N_uniques=1, 
                            **kwargs):
        '''
        N_uniques: number of unique poses in the kps/bones/skts
        '''
        skip = bones.shape[0] // N_uniques

        unique_kps = kps[::skip]
        unique_bones = bones[::skip]
        unique_skts = skts[::skip]

        w = self.graph_input_fn(unique_bones, **kwargs)

        return {'w': w}


class BaseEncoder(nn.Module):

    def __init__(self, N_joints=24, N_dims=None, skel_type=SMPLSkeleton):
        super().__init__()
        self.N_joints = N_joints
        self.N_dims = N_dims if N_dims is not None else 1
        self.skel_type = skel_type

    @property
    def dims(self):
        return self.N_joints * self.N_dims

    @property
    def encoder_name(self):
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        raise NotImplementedError

class IdentityEncoder(BaseEncoder):

    @property
    def encoder_name(self):
        return 'Identity'

    @property
    def dims(self):
        return self.N_dims

    def forward(self, inputs, refs, *args, **kwargs):
        shape = inputs.shape
        if len(shape) < 4:
            # no sample dimension, expand it
            inputs = inputs[:, None]
            inputs = inputs.expand(shape[0], refs.shape[1], *shape[1:])
            inputs = inputs.flatten(start_dim=2)
        return inputs

class EmptyEncoder(BaseEncoder):

    @property
    def encoder_name(self):
        return 'Empty'

    @property
    def dims(self):
        return 0

    def forward(self, inputs, *args, **kwargs):
        return inputs[..., :0]

class IdentityExpandEncoder(BaseEncoder):

    @property
    def encoder_name(self):
        return 'IdentityExpand'

    def forward(self, inputs, refs, *args, **kwargs):
        N_rays, N_samples = refs.shape[:2]
        return inputs[:, None].view(N_rays, 1, -1).expand(-1, N_samples, -1)

class WorldToLocalEncoder(BaseEncoder):

    @property
    def encoder_name(self):
        return 'W2LEncoder'

    def forward(self, pts, skts, *args, **kwargs):
        return transform_batch_pts(pts, skts)

class RelBoneDirEncoder(BaseEncoder):

    @property
    def encoder_name(self):
        return 'RBDEncoder'
    
    def forward(self, bones, kps, skts, transforms=None, **kwargs):
        '''
        transforms: bone-align transformation
        '''
        N_graphs, N_joints = bones.shape[:2]
        parents = self.skel_type.joint_trees
        parent_loc = kps[..., parents, :]
        # apply transformation (3, 3) x (3, 1) + (3, 1)
        rel_parent_loc = skts[..., :3, :3] @ parent_loc[..., None] + skts[..., :3, -1:] 
        if transforms is not None:
            transforms = transforms.reshape(N_graphs, N_joints, 4, 4)
            # apply transformation (3, 3) x (3, 1) + (3, 1)
            rel_parent_loc = transforms[..., :3, :3] @ rel_parent_loc + transforms[..., :3, -1:]

        # remove the expanded dimension
        rel_parent_loc = F.normalize(rel_parent_loc[..., 0], dim=-1)
        return rel_parent_loc

class RootLocalEncoder(BaseEncoder):

    @property
    def encoder_name(self):
        return 'RLEncoder'
    
    def forward(self, rays_o, rays_d, skts):
        local_rays = (skts[..., :1, :3, :3] @ rays_d[..., None])[..., 0]
        return local_rays

class WorldToBoneEncoder(WorldToLocalEncoder):

    def __init__(self, *args, local_root=False, **kwargs):
        self.local_root = local_root
        super(WorldToBoneEncoder, self).__init__(*args, **kwargs)

    @property
    def encoder_name(self):
        return 'W2BEncoder'

    def forward(self, pts, skts, rots=None, bones=None,
                coords=None, ref=None, *args, **kwargs):

        # get points in local coordinate (still not aligned with the bone)
        pts_l = super(WorldToBoneEncoder, self).forward(pts, skts)
        N_B, N_J = bones.shape[:2]

        if rots is None:
            N_B, N_J = bones.shape[:2]
            if bones.shape[-1] == 6:
                rots = rot6d_to_rotmat(bones)
            else:
                rots = axisang_to_rot(bones.reshape(-1, 3))
        else:
            N_B, N_J = rots.shape[:2]
        rots = rots.reshape(N_B, 1, N_J, 3, 3)

        # to (N_B, N_samples, N_joints, 3, 3)
        coords = coords.reshape(-1, 1, N_J, 3, 3)

        # project to the coordinate system aligned with the bones
        pts_b = (coords @ rots @ pts_l[..., None])[..., 0]

        if self.local_root:
            # assume root is at dim 0!
            pts_b = torch.cat([pts_l[..., :1, :], pts_b[..., 1:, :]], dim=-2)

        return pts_b

class JointCenteredEncoder(BaseEncoder):

    @property
    def encoder_name(self):
        return 'JCEncoder'

    def forward(self, pts, skts, rots=None, bones=None,
                coords=None, ref=None, kps=None, *args, **kwargs):
        return pts[..., None, :] - kps[:, None]

# KP-position encoders
class RelDistEncoder(BaseEncoder):

    def __init__(self, N_joints=24, N_dims=1, skel_type=SMPLSkeleton):
        super().__init__(N_joints, N_dims, skel_type=skel_type)

    @property
    def encoder_name(self):
        return 'RelDist'

    def forward(self, pts, pts_t, kps, *args, **kwargs):
        '''
        Args:
          pts (N_rays, N_pts, 3): 3d queries in world space.
          pts_t (N_rays, N_pts, N_joints, 3): 3d queries in local space (joints at (0, 0, 0)).
          kps (N_rays, N_joints, 3): 3d human keypoints.

        Returns:
          v (N_rays, N_pts, N_joints): relative distance encoding in the paper.
        '''
        if pts_t is not None:
            return torch.norm(pts_t, dim=-1, p=2)
        return torch.norm(pts[:, :, None] - kps[:, None], dim=-1, p=2)


class RelPosEncoder(BaseEncoder):

    def __init__(self, N_joints=24, N_dims=3, skel_type=SMPLSkeleton):
        super().__init__(N_joints, N_dims, skel_type=skel_type)

    @property
    def dims(self):
        return self.N_joints * 3

    @property
    def encoder_name(self):
        return 'RelPos'

    def forward(self, pts, pts_t, kps, *args, **kwargs):
        '''Return relative postion in 3D
        '''
        if pts_t is not None:
            return pts_t.flatten(start_dim=-2)
        return (pts[:, :, None] - kps[:, None]).flatten(start_dim=-2)

class KPCatEncoder(BaseEncoder):

    def __init__(self, N_joints=24, N_dims=3, skel_type=SMPLSkeleton):
        super().__init__(N_joints, N_dims, skel_type=skel_type)

    @property
    def dims(self):
        return self.N_joints * self.N_dims + self.N_dims

    @property
    def encoder_name(self):
        return 'KPCat'

    def forward(self, pts, pts_t, kps, *args, **kwargs):
        '''
        Args:
          pts (N_rays, N_pts, 3): 3d queries in world space
          pts_t (N_rays, N_pts, N_joints, 3): 3d queries in local space (joints at (0, 0, 0))
          kps (N_rays, N_joints, 3): 3d human keypoints

        Returns:
            cat (N_rays, N_pts, N_joints * 3 + 3): relative distance encoding in the paper.
        '''
        # to shape (N_rays, N_pts, N_joints * 3)
        kps = kps[:, None].expand(*pts.shape[:2], kps.shape[-2:]).flatten(start_dim=-2)
        return torch.cat([pts, kps], dim=-1)

class BoneDistEncoder(BaseEncoder):

    def __init__(self, N_joints=24, N_dims=1, skel_type=SMPLSkeleton):
        super().__init__(N_joints, N_dims, skel_type=skel_type)

    @property
    def encoder_name(self):
        # intentionally not put 'Dist' here so that we can still dist. to joints as cutoff
        return 'BoneD'

    def forward(self, pts, pts_t, kps, *args, **kwargs):
        '''
        Args:
          pts (N_rays, N_pts, 3): 3d queries in world space.
          pts_t (N_rays, N_pts, N_joints, 3): 3d queries in local space (joints at (0, 0, 0)).
          kps (N_rays, N_joints, 3): 3d human keypoints.

        Returns:
          v (N_rays, N_pts, N_joints): distance to bone (defined by the joint and its parent).
        '''
        # there's no bone for too, so just calculate relative distance
        root = pts_t[..., :1, :].norm(p=2, dim=-1)
        v = get_bone_dist(pts, kps, self.skel_type)
        return torch.cat([root, v],  dim=-1)

class BoneDistChildEncoder(BaseEncoder):

    def __init__(self, N_joints=24, N_dims=1, skel_type=SMPLSkeleton):
        super().__init__(N_joints, N_dims, skel_type=skel_type)
        self.init_bones()

    @property
    def encoder_name(self):
        # intentionally not put 'Dist' here so that we can still dist. to joints as cutoff
        return 'BoneDistChild'
    
    def init_bones(self):
        # some joint does not have a child
        children = get_children_joints(self.skel_type)
        self.bone_masks = torch.ones(len(self.skel_type.joint_trees))

        self.children = torch.arange(len(children))
        for parent, child_idxs in enumerate(children):
            if len(child_idxs) < 1 or len(child_idxs) > 1:
                self.bone_masks[parent] = 0.
                continue
            self.children[parent] = child_idxs[0]

    def forward(self, pts, pts_t, kps, *args, **kwargs):
        '''
        Args:
          pts (N_rays, N_pts, 3): 3d queries in world space.
          pts_t (N_rays, N_pts, N_joints, 3): 3d queries in local space (joints at (0, 0, 0)).
          kps (N_rays, N_joints, 3): 3d human keypoints.

        Returns:
          v (N_rays, N_pts, N_joints): distance to bone (defined by the joint and its parent).
        '''
        mask = self.bone_masks.to(kps.device)
        # there's no bone for too, so just calculate relative distance
        joint_dist = pts_t.norm(p=2, dim=-1)
        bone_dist = get_bone_dist_child(pts, kps, self.children.to(kps.device), 
                                        mask, self.skel_type)
        if torch.isnan(bone_dist).any():
            bone_dist = torch.nan_to_num(bone_dist, 0.0)
            import pdb; pdb.set_trace()
            print
        mask = mask.reshape(1, 1, -1)
        v = bone_dist * mask + (1 - mask) * joint_dist 
        return v


# View/Bone encoding
class VecNormEncoder(BaseEncoder):

    def __init__(self, N_joints=24, N_dims=3, skel_type=SMPLSkeleton):
        super().__init__(N_joints, N_dims, skel_type=skel_type)

    @property
    def encoder_name(self):
        return 'VecNorm'

    def forward(self, vecs, refs=None, *args, **kwargs):
        '''
        Args:
          vecs (N_rays, *, ...): vector to normalize.
          refs (N_rays, N_pts, ...): reference tensor for shape expansion.
        Returns:
          (N_rays, N_pts, *): normalized vector with expanded shape (if needed).
        '''
        n = F.normalize(vecs, dim=-1, p=2).flatten(start_dim=2)
        # expand to match N samples
        if refs is not None:
            n = n.expand(*refs.shape[:2], -1)
        return n

class JointBoneDirEncoder(BaseEncoder):

    def __init__(self, N_joints=24, N_dims=3, skel_type=SMPLSkeleton):
        super().__init__(N_joints, N_dims, skel_type=skel_type)
        self.parent_idxs = skel_type.joint_trees
        self.child_idxs = np.arange(len(self.parent_idxs))
        self.child_mask = torch.ones(len(self.parent_idxs))
        self.end_effectors = torch.zeros(len(self.parent_idxs))
        self.end_effectors[skel_type.end_effectors] = 1
        children = [[] for i in range(len(self.parent_idxs))]
        for i, parent in enumerate(self.parent_idxs):
            children[parent].append(i)
        for i, c in enumerate(children):
            if len(c) > 1 or len(c) == 0: # exist multiple children
                self.child_mask[i] = 0
                continue
            self.child_idxs[i] = c[0]

    @property
    def encoder_name(self):
        return 'JointBoneDir'

    @property
    def dims(self):
        return self.N_dims * 2

    def forward(self, bones, kps, skts, *args, **kwargs):
        # assume shape be (N_B, N_joint, 3)
        kps = kps[:, None]
        kp_parent = kps[:, :, self.parent_idxs]
        kp_child = kps[:, :, self.child_idxs]
        vec_parent = F.normalize(transform_batch_pts(kp_parent, skts), dim=-1, p=2).squeeze(1)
        vec_child = F.normalize(transform_batch_pts(kp_child, skts), dim=-1, p=2).squeeze(1)
        # mask out multi-children joints and end effectors
        mask = self.child_mask.reshape(1, -1, 1).to(vec_child.device)

        # use their rotation (in normalized direction) as "child rotation"
        # which is equivalent to the first column of rotation
        # i.e., R @ [1, 0, 0]^T = first column
        rot = axisang_to_rot(bones)[..., :3, 0]
        vec_child = vec_child * mask + (1 - mask) * rot
        return torch.cat([vec_parent, vec_child], dim=-1)

class RayAngEncoder(BaseEncoder):

    def __init__(self, N_joints=24, N_dims=1, skel_type=SMPLSkeleton):
        super().__init__(N_joints, N_dims, skel_type=skel_type)

    @property
    def encoder_name(self):
        return 'RayAng'

    def forward(self, rays_t, pts_t, *args, **kwargs):
        '''
        Args:
          rays_t (N_rays, 1, N_joints, 3): rays direction in local space (joints at (0, 0, 0))
          pts_t (N_rays, N_pts, N_joints, 3): 3d queries in local space (joints at (0, 0, 0))
        Returns:
          d (N_rays, N_pts, N_joints*3): normalized ray direction in local space
        '''
        return calculate_angle(pts_t, rays_t)

class AxisAngtoRot6DEncoder(BaseEncoder):

    def __init__(self, N_joints=24, N_dims=3, skel_type=SMPLSkeleton):
        super().__init__(N_joints, N_dims, skel_type=skel_type)

    @property
    def encoder_name(self):
        return 'Rot6D'
    
    @property
    def dims(self):
        return 6
    
    def forward(self, bones, *args, **kwargs):
        if bones.shape[-1] == 6:
            # already in 6D format.
            # this happens for the case of pose optim with rot6d
            return bones
        return axisang_to_rot6d(bones)

class PartRotateEncoder(BaseEncoder):
    def __init__(self, N_joints=24, N_dims=3, part_dims=10, skel_type=SMPLSkeleton):
        self.part_dims = part_dims
        super().__init__(N_joints, N_dims, skel_type=skel_type)

    @property
    def encoder_name(self):
        return 'PartRot'
    
    @property
    def dims(self):
        return 3 * self.part_dims
    
    def forward(self, bones, *args, part_feat=None, align_transforms=None,**kwargs):
        if bones.shape[-1] == 6:
            # already in 6D format.
            # this happens for the case of pose optim with rot6d
            rotmat = rot6d_to_rotmat(bones)
        elif bones.shape[-1] == 3:
            rotmat = axisang_to_rot(bones)
        else:
            raise NotImplementedError(f'unknown rotation format with shape {bones.shape}')
        return rotmat @ part_feat[None]


