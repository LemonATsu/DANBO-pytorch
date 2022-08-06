import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad

import numpy as np
from copy import deepcopy

from .networks import *
from .encoders import *
from .cutoff_embedder import get_embedder
from .utils.ray_utils import *
from .utils.run_nerf_helpers import *
from .utils.skeleton_utils import *

def create_raycaster(args, data_attrs, device=None):
    """Instantiate NeRF's MLP model.
    """
    skel_type = data_attrs["skel_type"]
    near, far = data_attrs["near"], data_attrs["far"]
    n_framecodes = data_attrs["n_views"] if args.n_framecodes is None else args.n_framecodes

    if args.vol_cal_scale:
        data_attrs['skel_profile'] = get_skel_profile_from_rest_pose(data_attrs['rest_pose'], skel_type=skel_type)

    pts_tr_fn = get_pts_tr_fn(args)
    ray_tr_fn = get_ray_tr_fn(args)
    kp_input_fn, input_dims, cutoff_dims = get_kp_input_fn(args, skel_type)
    bone_input_fn, bone_dims = get_bone_input_fn(args, skel_type)
    view_input_fn, view_dims = get_view_input_fn(args, skel_type)
    print(f'PPE: {pts_tr_fn.encoder_name}, KPE: {kp_input_fn.encoder_name},' +
          f'BPE: {bone_input_fn.encoder_name}, VPE: {view_input_fn.encoder_name}')

    cutoff_kwargs = {
        "cutoff": args.use_cutoff,
        "normalize_cutoff": args.normalize_cutoff,
        "cutoff_dist": args.cutoff_mm * args.ext_scale,
        "cutoff_inputs": args.cutoff_inputs,
        "opt_cutoff": args.opt_cutoff,
        "cutoff_dim": cutoff_dims,
        "dist_inputs":  not(input_dims == cutoff_dims),
    }

    # function to encode input (RGB/view to PE)
    embedpts_fn, input_ch_pts = None, None # only for mnerf
    dist_cutoff_kwargs = deepcopy(cutoff_kwargs)
    dist_cutoff_kwargs['cut_to_cutoff'] = args.cut_to_dist
    dist_cutoff_kwargs['shift_inputs'] = args.cutoff_shift
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed,
                                      input_dims=input_dims,
                                      skel_type=skel_type,
                                      freq_schedule=args.freq_schedule,
                                      init_alpha=args.init_freq,
                                      cutoff_kwargs=dist_cutoff_kwargs)

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

    input_ch_views = 0
    embeddirs_fn = None
    # no cutoff for view dir
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


    # TODO: not the best way to have this
    caster_nerf_kwargs, caster_class_kwargs, caster_preproc_kwargs = {}, {}, {}
    input_ch_graph, input_ch_voxel = 0, 0
    embedgraph_fn, embedvoxel_fn = None, None
    if 'graph' in args.nerf_type:
        graph_input_fn, graph_dims = get_graph_input_fn(args, skel_type)
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
        if args.gnn_backbone.startswith('MVP'):
            voxel_input_dims -= 1

        pe_input_dims = voxel_input_dims

        embedvoxel_fn, input_ch_voxel = get_embedder(args.multires_voxel, args.i_embed,
                                                     input_dims=pe_input_dims,
                                                     freq_schedule=args.freq_schedule,
                                                     init_alpha=args.init_freq,
                                                     skel_type=skel_type)
        caster_nerf_kwargs['input_ch_graph'] = input_ch_graph
        caster_nerf_kwargs['input_ch_voxel'] = input_ch_voxel

        caster_nerf_kwargs['graph_pe_fn'] = embedgraph_fn
        caster_nerf_kwargs['voxel_pe_fn'] = embedvoxel_fn
        caster_nerf_kwargs['mask_vol_prob'] = args.mask_vol_prob
        caster_nerf_kwargs['agg_type'] = args.agg_type


    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]

    nerf_kwargs = {'D': args.netdepth, 'W': args.netwidth,
                   'input_ch': input_ch,
                   'input_ch_bones': input_ch_bones,
                   'input_ch_views': input_ch_views,
                   'output_ch': output_ch, 'skips': skips,
                   'use_viewdirs': args.use_viewdirs,
                   'use_framecode': args.opt_framecode,
                   'framecode_ch': args.framecode_size,
                   'n_framecodes': n_framecodes,
                   'skel_type': skel_type,
                   'density_scale': args.density_scale,
                   'pts_embedder': get_pts_embedder(args, data_attrs), # here, temporarily 
                   'pe_fn': embed_fn,
                   'bones_pe_fn': embedbones_fn,
                   'dirs_pe_fn': embeddirs_fn,
                   'view_W': args.netwidth_view,
                   **caster_nerf_kwargs}

    model, model_fine, caster_class = create_nerf(args, nerf_kwargs, data_attrs)

    # create ray caster
    if caster_class is None:
        caster_class = RayCaster
    elif caster_class.startswith('graph'):
        caster_class = GraphCaster
        caster_class_kwargs['use_volume_near_far'] = args.use_volume_near_far
    else:
        raise NotImplementedError(f'caster class {caster_class} is not implemented.')
    ray_caster = caster_class(model,
                              network_fine=model_fine,
                              rest_poses=data_attrs['rest_pose'],
                              single_net=args.single_net, 
                              align_bones=args.align_bones,
                              skel_type=skel_type,
                              **caster_class_kwargs)
    print(ray_caster)
    ray_caster.state_dict()
    # add all learnable grad vars
    grad_vars = get_grad_vars(args, ray_caster)

    # Create optimizer
    if args.weight_decay is None:
        optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))
    else:
        optimizer = torch.optim.AdamW(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999),
                                      weight_decay=args.weight_decay)

    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path!='None':
        ckpts = [args.ft_path]
    else:
        import os
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f and 'pose' not in f]

    print('Found ckpts', ckpts)
    loaded_ckpt = None
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        start, ray_caster, optimizer, loaded_ckpt = load_ckpt_from_path(ray_caster,
                                                                        optimizer,
                                                                        ckpt_path,
                                                                        args.finetune or args.finetune_light)
        if args.finetune or args.finetune_light:
            start = 0
            print(f"set global step to {start}")

    ##########################
    preproc_kwargs = {
        'pts_tr_fn': pts_tr_fn,
        'ray_tr_fn': ray_tr_fn,
        'kp_input_fn': kp_input_fn,
        'view_input_fn': view_input_fn,
        'bone_input_fn': bone_input_fn,
        'density_scale': args.density_scale,
        'density_fn': get_density_fn(args),
        **caster_preproc_kwargs,
    }
    # copy preproc kwargs and disable peturbation  for test-time rendering
    preproc_kwargs_test = {k: preproc_kwargs[k] for k in preproc_kwargs}

    render_kwargs_train = {
        'ray_caster': nn.DataParallel(ray_caster) if not args.debug else nn.DataParallel(ray_caster, device_ids=[0]),
        'perturb' : args.perturb,
        'N_importance' : args.N_importance,
        'N_samples' : args.N_samples,
        'use_viewdirs' : args.use_viewdirs,
        'raw_noise_std' : args.raw_noise_std,
        'ray_noise_std': args.ray_noise_std,
        'ext_scale': args.ext_scale,
        'preproc_kwargs': preproc_kwargs,
        'lindisp': args.lindisp,
        'nerf_type': args.nerf_type,
    }

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    # set properties that should be turned off during test here
    # TODO: we have to reuse some inputs during test time (e.g., kp has batch_size == 1)
    # which doesn't work for DataParallel. Any smart way to fix this?
    render_kwargs_test['ray_caster'] = ray_caster
    render_kwargs_test['preproc_kwargs'] = preproc_kwargs_test
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.
    render_kwargs_test['ray_noise_std'] = 0.
    print(f"#parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # reset gradient
    optimizer.zero_grad()

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer, loaded_ckpt

def get_grad_vars(args, ray_caster):

    network, network_fine = ray_caster.get_networks()

    grad_vars = []

    def freeze_weights(pts_linears, layer):
        '''
        only works for standard NeRF now
        '''
        for i, l in enumerate(pts_linears):
            if i >= layer:
                break
            for p in l.parameters():
                p.requires_grad = False

    def get_vars(x, add=True):
        if not add or x is None:
            return []

        trainable_vars = []
        for p in x.parameters():
            if p.requires_grad:
                trainable_vars.append(p)
        return trainable_vars

    get_vars_old = lambda x, add=True: list(x.parameters()) if (x is not None) and add else []

    if args.finetune and args.fix_layer > 0:
        freeze_weights(network.pts_linears, args.fix_layer)
        freeze_weights(network_fine.pts_linears, args.fix_layer)
    
    if args.finetune_light:
        for n, p in network.named_parameters():
            if not ('framecodes' in n):
                p.requires_grad = False
            else:
                print(f'{n} has gradient.')


    grad_vars += get_vars(network)
    grad_vars += get_vars(network_fine, not args.single_net)

    return grad_vars

def get_density_fn(args):
    if args.density_type == 'relu':
        return F.relu
    elif args.density_type == 'softplus':
        shift = args.softplus_shift
        softplus = lambda x: F.softplus(x - shift, beta=1) #(1 + (x - shift).exp()).log()
        return softplus
    else:
        raise NotImplementedError(f'density activation {args.density_type} is undefined')

    pass


class RayCaster(nn.Module):

    def __init__(self, network,
                 network_fine=None, 
                 single_net=False,
                 rest_poses=None, 
                 align_bones=None, 
                 skel_type=None, 
                 **kwargs):
        super().__init__()

        self.network = network
        self.network_fine = network_fine
        self.rest_poses = rest_poses
        self.skel_type = skel_type
        self.align_bones = align_bones

        if self.align_bones is not None:
            self.init_bone_align_transforms()

        self.single_net = single_net

    @torch.no_grad()
    def forward_eval(self, *args, **kwargs):
        return self.render_rays(*args, **kwargs)



    def forward(self, *args, fwd_type='', **kwargs):
        if fwd_type == 'density':
            return self.render_pts_density(*args, **kwargs)
        elif fwd_type == 'density_color':
            return self.render_pts_density(*args, **kwargs, color=True)
        elif fwd_type == 'mesh':
            return self.render_mesh_density(*args, **kwargs)

        if not self.training:
            return self.forward_eval(*args, **kwargs)
        return self.render_rays(*args, **kwargs)

    def render_rays(self,
                    ray_batch,
                    N_samples,
                    kp_batch,
                    skts=None,
                    cyls=None,
                    bones=None,
                    cams=None,
                    subject_idxs=None,
                    retraw=False,
                    lindisp=False,
                    perturb=0.,
                    N_importance=0,
                    network_fine=None,
                    raw_noise_std=0.,
                    ray_noise_std=0.,
                    verbose=False,
                    ext_scale=0.001,
                    pytest=False,
                    N_uniques=1,
                    render_confd=False,
                    render_entropy=False,
                    preproc_kwargs={},
                    netchunk=1024*64,
                    nerf_type="nerf"):
        """Volumetric rendering.
        Args:
          ray_batch: array of shape [batch_size, ...]. All information necessary
            for sampling along a ray, including: ray origin, ray direction, min
            dist, max dist, and unit-magnitude viewing direction.
          N_samples: int. Number of different times to sample along each ray.
          kp_batch: array of keypoints for calculating input points.
          kp_input_fn: function that calculate the input using keypoints for the network
          cyls: cylinder parameters for dynamic near/far plane sampling
          retraw: bool. If True, include model's raw, unprocessed predictions.
          lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
          perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
            random points in time.
          N_importance: int. Number of additional times to sample along each ray.
            These samples are only passed to network_fine.
          network_fine: "fine" network with same spec as network_fn.
          raw_noise_std: ...
          N_uniques: int. number of unique poses
          verbose: bool. If True, print more debugging info.
        Returns:
          rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
          disp_map: [num_rays]. Disparity map. 1 / depth.
          acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
          raw: [num_rays, num_samples, 4]. Raw predictions from model.
          rgb0: See rgb_map. Output for coarse model.
          disp0: See disp_map. Output for coarse model.
          acc0: See acc_map. Output for coarse model.
          z_std: [num_rays]. Standard deviation of distances along ray for each
            sample.
        """
        # Step 1: prep ray data
        # Note: last dimension for ray direction needs to be normalized
        N_rays = ray_batch.shape[0]
        rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each
        viewdirs = ray_batch[:,-3:] if ray_batch.shape[-1] > 8 else None
        bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])
        near, far = bounds[...,0], bounds[...,1] # [-1,1]

        # Step 2: Sample 'coarse' sample from the ray segment within the bounding cylinder
        #near, far =  get_near_far_in_cylinder(rays_o, rays_d, cyls, near=near, far=far)
        near, far = self.get_near_far(rays_o, rays_d, cyls, near=near, far=far, skts=skts)
        pts, z_vals = self.sample_pts(rays_o, rays_d, near, far, N_rays, N_samples,
                                      perturb, lindisp, pytest=pytest, ray_noise_std=ray_noise_std)

        """
        # prepare local coordinate system (not really used here)
        joint_coords = self.get_subject_joint_coords(subject_idxs, pts.device)

        # Step 3: encode
        encoded = self.encode_inputs(pts, [rays_o[:, None, :], rays_d[:, None, :]], kp_batch,
                                     skts, bones, cam_idxs=cams, subject_idxs=subject_idxs,
                                     joint_coords=joint_coords, network=self.network,
                                     N_uniques=N_uniques, **preproc_kwargs)
        """

        nerf_inputs = self.get_nerf_inputs(
                                    pts, [rays_o[:, None, :], rays_d[:, None, :]], 
                                    kp_batch, skts, bones, cam_idxs=cams, 
                                    subject_idxs=subject_idxs,
                                    N_uniques=N_uniques,
                            )

        # Step 4: forwarding in NeRF and get coarse outputs
        raw, encoded = self.network(nerf_inputs, netchunk=netchunk)
        ret_dict = self.network.raw2outputs(raw, z_vals, rays_d, 
                                            raw_noise_std=raw_noise_std, pytest=pytest,
                                            encoded=encoded, B=preproc_kwargs['density_scale'],
                                            act_fn=preproc_kwargs['density_fn'])

        # Step 6: generate fine outputs
        ret_dict0, encoded0 = None, None
        if N_importance > 0:
            # preserve coarse output
            ret_dict0 = ret_dict

            # get the importance samples (z_samples), as well as sorted_idx
            pts_is, z_vals, z_samples, sorted_idxs = self.sample_pts_is(rays_o, rays_d, z_vals, ret_dict0['weights'],
                                                                        N_importance, det=(perturb==0.), pytest=pytest,
                                                                        is_only=self.single_net, ray_noise_std=ray_noise_std)

            if not self.single_net:
                # fine network needs both coarse and importance samples
                N_total_samples = N_importance + N_samples
                pts_is = self._merge_encodings({'pts': pts}, {'pts': pts_is}, sorted_idxs,
                                                N_rays, N_total_samples)['pts']
                encoded0 = encoded

            nerf_inputs_is = self.get_nerf_inputs(
                                    pts_is, [rays_o[:, None, :], rays_d[:, None, :]], 
                                    kp_batch, skts, bones, cam_idxs=cams, 
                                    subject_idxs=subject_idxs,
                                    N_uniques=N_uniques,
                          )
            raw_is, encoded_is = self.network_fine(nerf_inputs_is, netchunk=netchunk)
            if self.single_net:
                N_total_samples = N_importance + N_samples
                encoded_is = self._merge_encodings(encoded, encoded_is, sorted_idxs,
                                                   N_rays, N_total_samples)
                raw = self._merge_encodings({'raw': raw}, {'raw': raw_is}, sorted_idxs,
                                             N_rays, N_total_samples)['raw']
            else:
                encoded = encoded_is
                raw = raw_is
            ret_dict = self.network_fine.raw2outputs(raw, z_vals, rays_d, raw_noise_std=raw_noise_std, pytest=pytest,
                                                     encoded=encoded_is, B=preproc_kwargs['density_scale'],
                                                     act_fn=preproc_kwargs['density_fn'])

        return self._collect_outputs(ret_dict, ret_dict0, encoded_is, encoded0)

    def get_nerf_inputs(self, pts, rays, kps, skts, bones, 
                        cam_idxs=None, subject_idxs=None, 
                        N_uniques=1):
        
        # TODO: internalize this to NeRF model?
        if self.align_bones is not None:
            if subject_idxs is None:
                subject_idxs = torch.zeros(len(pts)).long()
            transforms = self.transforms[subject_idxs, None].to(pts.device)

        # TODO: internalize this to NeRF model?
        if subject_idxs is not None:
            rest_poses = self.rest_poses
            if len(rest_poses.shape) < 3:
                rest_poses = rest_poses[None]
            rest_pose = torch.tensor(rest_poses)[subject_idxs]
        else:
            rest_pose = torch.tensor(self.rest_poses)
        rest_pose = rest_pose.reshape(-1, 1, *rest_pose.shape[-2:])

        inputs = {
            'pts': pts,
            'kps': kps,
            'skts': skts,
            'bones': bones,
            'rest_pose': rest_pose,
            'align_transforms': transforms,
            'N_uniques': N_uniques,
        }

        # don't necessarily need these all the time
        if rays is not None:
            inputs['rays_o'] = rays[0]
            inputs['rays_d'] = rays[1]
            inputs['cam_idxs'] = cam_idxs

        return inputs
    

    def get_near_far(self, rays_o, rays_d, cyls, near=0., far=100., **kwargs):
        return get_near_far_in_cylinder(rays_o, rays_d, cyls, near=near, far=far)
        
    @torch.no_grad()
    def render_mesh_density(self, kps, skts, bones, subject_idxs=None, radius=1.0, res=64,
                            render_kwargs=None, netchunk=1024*64, v=None):

        # generate 3d cubes (voxels)
        t = np.linspace(-radius, radius, res+1)
        grid_pts = np.stack(np.meshgrid(t, t, t), axis=-1).astype(np.float32)
        sh = grid_pts.shape
        grid_pts = torch.tensor(grid_pts.reshape(-1, 3)) + kps[0, 0]
        grid_pts = grid_pts.reshape(-1, 1, 3)

        # create density forward function (for batchify computation)
        raw_density = self.render_pts_density(grid_pts, kps, skts, 
                                              bones, netchunk)[..., :1]

        # swap x-y to match trimesh
        return raw_density.reshape(*sh[:-1]).transpose(1, 0)

    def render_pts_density(self, pts, kps, skts, bones, netchunk=1024*64, network=None):

        assert kps.shape[0] == 1, 'Assuming only one poses are provided, got {kps.shape[0]} instead'

        # batchified computation
        density = []
        for i in range(0, pts.shape[0], netchunk):
            sub_pts = pts[i:i+netchunk]
            nerf_inputs = self.get_nerf_inputs(sub_pts, None, kps, skts, bones, N_uniques=1)
            sub_outputs = self.network.forward_pts(nerf_inputs, with_feature=False)
            density.append(sub_outputs)

        # create density forward function (for batchify computation)
        density = torch.cat(density, dim=0)
        return density

    def sample_pts(self, rays_o, rays_d, near, far, N_rays, N_samples,
                   perturb, lindisp, pytest=False, ray_noise_std=0.):

        z_vals = sample_from_lineseg(near, far, N_rays, N_samples,
                                     perturb, lindisp, pytest=pytest)

        # range of points should be bounded within 2pi.
        # so the lowest frequency component (sin(2^0 * p)) don't wrap around within the bound
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]

        if ray_noise_std > 0.:
            pts = pts + torch.randn_like(pts) * ray_noise_std

        return pts, z_vals

    def sample_pts_is(self, rays_o, rays_d, z_vals, weights, N_importance,
                      det=True, pytest=False, is_only=False, ray_noise_std=0.):

        z_vals, z_samples, sorted_idxs = isample_from_lineseg(z_vals, weights, N_importance, det=det,
                                                              pytest=pytest, is_only=is_only)

        pts_is = rays_o[...,None,:] + rays_d[...,None,:] * z_samples[...,:,None] # [N_rays, N_samples + N_importance, 3]

        if ray_noise_std> 0.:
            pts_is = pts_is + torch.randn_like(pts_is) * ray_noise_std


        return pts_is, z_vals, z_samples, sorted_idxs

    def _merge_encodings(self, encoded, encoded_is, sorted_idxs,
                         N_rays, N_total_samples, inplace=True):
        """
        merge coarse and fine encodings.
        encoded: dictionary of coarse encodings
        encoded_is: dictionary of fine encodings
        sorted_idxs: define how the [encoded, encoded_is] are sorted
        """
        gather_idxs = torch.arange(N_rays * (N_total_samples)).view(N_rays, -1)
        gather_idxs = torch.gather(gather_idxs, 1, sorted_idxs)
        if not inplace:
            merged = {}
        else:
            merged = encoded
        for k in encoded.keys():
            #if not k.startswith(('pts', 'blend')) and k not in ['g', 'graph_feat', 'bone_logit']:
            merged[k] = merge_samples(encoded[k], encoded_is[k], gather_idxs, N_total_samples)

        # need special treatment here to preserve the computation graph.
        # (otherwise we can just re-encode everything again, but that takes extra computes)
        if 'pts' in encoded and encoded['pts'] is not None:
            if not inplace:
                merged['pts'] = encoded['pts']

            merged['pts_is'] = encoded_is['pts']
            merged['gather_idxs'] = gather_idxs

            merged['pts_sorted'] = merge_samples(encoded['pts'], encoded_is['pts'],
                                                 gather_idxs, N_total_samples)

        return merged

    def _collect_outputs(self, ret, ret0=None, encoded=None, encoded0=None):
        ''' collect outputs into a dictionary for loss computation/rendering
        ret: outputs from fine networki (or coarse network if we don't have a fine one).
        ret0: outputs from coarse network.
        '''
        collected = {'rgb_map': ret['rgb_map'], 'disp_map': ret['disp_map'],
                     'acc_map': ret['acc_map'], 'alpha': ret['alpha'], 
                     'T_i': ret['weights']}
        if ret0 is not None:
            collected['rgb0'] = ret0['rgb_map']
            collected['disp0'] = ret0['disp_map']
            collected['acc0'] = ret0['acc_map']
            collected['alpha0'] = ret0['alpha']

        if 'confd' in ret and self.training:
            collected['confd'] = ret['confd']
            if ret0 is not None:
                collected['confd0'] = ret0['confd']

        if 'j_dists' in ret and self.training:
            collected['j_dists'] = ret['j_dists']
            if ret0 is not None:
                collected['j_dists0'] = ret0['j_dists']

        if 'gradient' in encoded and self.training:
            collected['gradient'] = encoded['gradient']
        
            if encoded0 is not None:
                collected['gradient0'] = encoded0['gradient']

        return collected

    def init_bone_align_transforms(self):
        from core.utils.skeleton_utils import get_axis_aligned_rotation 
        skel_type = self.skel_type
        joint_names = skel_type.joint_names
        joint_trees = skel_type.joint_trees
        rest_poses = self.rest_poses.reshape(-1, len(joint_trees), 3)
        children = [[] for j in joint_trees]

        # search for children
        for i, parent in enumerate(joint_trees):
            children[parent].append(i)
        
        N_rest_poses = len(self.rest_poses) if len(self.rest_poses.shape) == 3 else 1
        
        transforms = torch.eye(4).reshape(1, 1, 4, 4).repeat(N_rest_poses, len(joint_trees), 1, 1)
        child_idxs = []
        for parent_idx, c in enumerate(children):
            # has no child or has multiple child:
            # no needs to align
            if len(c) < 1 or len(c) > 1:
                child_idxs.append(parent_idx)
                continue
            child_idx = c[0]
            # from parent to child
            dir_vecs = rest_poses[:, child_idx] - rest_poses[:, parent_idx]

            if self.align_bones == 'align':
                rots = []
                # the underlying function is not vectorized
                for dir_vec in dir_vecs:
                    rots.append(get_axis_aligned_rotation(dir_vec))
                rots = np.stack(rots)

                # translation to center of the bone (shift it along z-axis)
                trans = -0.5 * np.linalg.norm(dir_vecs, axis=-1)[..., None] * np.array([[0., 0., 1.]], dtype=np.float32)
                rots[..., :3, -1] = trans
            else:
                rots = np.eye(4).reshape(1, 4, 4).repeat(len(self.rest_poses), 0)
                trans = -0.5 * (dir_vecs).astype(np.float32)
                rots[:, :3, -1] = trans
            transforms[:, parent_idx] = torch.tensor(rots.copy())
            child_idxs.append(child_idx)
        self.transforms = transforms
        self.child_idxs = np.array(child_idxs)
        
    def update_embed_fns(self, global_step, args):

        self.network.update_embed_fns(global_step, args)
        if not (self.network is self.network_fine):
            self.network_fine.update_embed_fns(global_step, args)


    # TODO: check if we really need this state_dict/load_state_dict ..
    def state_dict(self):
        state_dict = {}
        modules = self.__dict__['_modules']
        for k in modules:
            m = modules[k]
            # rules for backward compatibility ...
            if k.endswith("_fine"):
                state_dict[f"{k}_state_dict"] = m.state_dict()
            elif k.endswith("_fn"):
                state_dict[f"{k.split('_fn')[0]}_state_dict"] = m.state_dict()
            elif k == "network":
                state_dict["network_fn_state_dict"] = m.state_dict()
            else:
                state_dict[f"{k}_state_dict"] = m.state_dict()
        return state_dict

    def load_state_dict(self, ckpt, strict=True):

        modules = self.__dict__['_modules']
        for k in modules:
            if k.endswith("_fine"):
                target_key = f"{k}_state_dict"
            elif k.endswith("_fn"):
                target_key = f"{k.split('_fn')[0]}_state_dict"
            elif k == "network":
                target_key = "network_fn_state_dict"
            else:
                target_key = f"{k}_state_dict"
            try:
                modules[k].load_state_dict(ckpt[target_key], strict=strict)
            except (KeyError, RuntimeError):
                if k.startswith('network'):
                    print(f'Error occur when loading state dict for network. Try loading with strict=False now')
                    filtered_sd = filter_state_dict(modules[k].state_dict(), ckpt[target_key])
                    modules[k].load_state_dict(filtered_sd, strict=False)
                else:
                    print(f'Error occurr when loading state dict for {target_key}. The entity is not in the state dict?')

    def get_networks(self):
        return self.network, self.network_fine

class GraphCaster(RayCaster):

    def __init__(self, *args, use_volume_near_far=False, **kwargs):
        super(GraphCaster, self).__init__(*args, **kwargs)
        self.use_volume_near_far = use_volume_near_far
    
    @torch.no_grad()
    def get_near_far(self, rays_o, rays_d, cyls, near=0., far=100., skts=None):

        near, far = get_near_far_in_cylinder(rays_o, rays_d, cyls, near=near, far=far)
        if not self.use_volume_near_far:
            return near, far
        
        assert skts is not None
        assert self.align_bones is not None, 'volume_near_far only works when self.align_bones="align"'
        B, J = skts.shape[:2]

        # transform both rays origin and direction to the per-joint coordinate
        rays_ot = (skts[..., :3, :3] @ rays_o.reshape(B, 1, 3, 1) + skts[..., :3, -1:]).reshape(B, J, 3)
        rays_dt = (skts[..., :3, :3] @ rays_d.reshape(B, 1, 3, 1)).reshape(B, J, 3)

        if self.align_bones is not None:
            align_transforms = self.transforms[:1].to(rays_o.device)
            rays_ot = ((align_transforms[..., :3, :3] @ rays_ot[..., None]) + \
                        align_transforms[..., :3, -1:]).reshape(B, J, 3)
            rays_dt = (align_transforms[..., :3, :3] @ rays_dt[..., None]).reshape(B, J, 3)
        
        # scale the rays by the learned volume scale and find the intersections with the volumes
        axis_scale = self.network.graph_net.get_axis_scale().reshape(1, J, 3).abs()
        p_valid, v_valid, p_intervals = get_ray_box_intersections(
                                            rays_ot / axis_scale, 
                                            rays_dt / axis_scale, 
                                            bound_range=1.1, # intentionally makes the bound a bit larger
                                        )
        # now undo the scale so we can calculate the near far in the original space
        axis_scale = axis_scale.expand(B, J, 3)
        p_intervals = p_intervals * axis_scale[v_valid][..., None, :]

        norm_rays = rays_dt[v_valid].norm(dim=-1)
        # find the step size (near / far)
        # t * norm_ray + ray_o = p -> t =  (p - ray_o) / norm_rays
        # -> distance is the norm 
        steps = (p_intervals - rays_ot[v_valid][..., None, :]).norm(dim=-1) / norm_rays[..., None]

        # extract near/far for each volume
        v_near = 100000 * torch.ones(B, J)
        v_far = -100000 * torch.ones(B, J)

        # find the near/far
        v_near[v_valid] = steps.min(dim=-1).values
        v_far[v_valid] = steps.max(dim=-1).values

        # pick the closest/farthest points as the near/far planes
        v_near = v_near.min(dim=-1).values
        v_far = v_far.max(dim=-1).values

        # merge the values back to the cylinder near far
        ray_valid = (v_valid.sum(-1) > 0)
        new_near = near.clone()
        new_far = far.clone()

        new_near[ray_valid, 0] = v_near[ray_valid]
        new_far[ray_valid, 0] = v_far[ray_valid]

        return new_near, new_far


    def _collect_outputs(self, ret, ret0=None, encoded=None, encoded0=None):
        collected = super(GraphCaster, self)._collect_outputs(ret, ret0, encoded)
        if 'part_invalid' in encoded and self.training:
            collected['part_invalid'] = encoded['part_invalid']
        if 'confd' in encoded and self.training:
            collected['confd'] = encoded['confd']
        return collected

    def _get_density_fwd_fn(self, kps, skts, bones, network):

        if network is None:
            if self.network_fine is not None:
                network = self.network_fine
            else:
                network = self.network
        fwd_pts = network.forward_pts

        def fwd_fn(pts, v=None):
            # prepare inputs for density network
            # TODO: could be buggy if aways set N_uniques=1?
            encoded_graphs = self.encode_graphs(kps, bones, skts, network,
                                                N_uniques=1, graph_input_fn=graph_input_fn)
            encoded = self.encode_inputs(encoded_graphs, pts, None, 
                                         kps, skts, bones, 
                                         network=network, 
                                         kp_input_fn=kp_input_fn,
                                         bone_input_fn=bone_input_fn,
                                         pts_tr_fn=pts_tr_fn)
            pts_out = fwd_pts(encoded['blend_feat'], 
                              subject_idxs=encoded['subject_idxs'])

            return pts_out

        return fwd_fn

def merge_samples(x, x_is, gather_idxs, N_total_samples):
    """
    merge coarse and fine samples.
    x: coarse samples of shape (N_rays, N_coarse, -1)
    x_is: importance samples of shape (N_rays, N_fine, -1)
    gather_idx: define how the [x, x_is] are sorted
    """
    if x is None or x.shape[-1] == 0:
        return None
    N_rays = x.shape[0]
    x_is = torch.cat([x, x_is], dim=1)
    sh = x_is.shape
    feat_size = np.prod(sh[2:])
    x_is = x_is.view(-1, feat_size)[gather_idxs, :]
    x_is = x_is.view(N_rays, N_total_samples, *sh[2:])

    return x_is

def batchify(fn, chunk):
    if chunk is None:
        return fn
    def ret(inputs, **kwargs):
        # be careful about the concatenation dim
        return torch.cat([fn(inputs[i:i+chunk], **{k: kwargs[k][i:i+chunk] for k in kwargs})
                          for i in range(0, inputs.shape[0], chunk)], 0)
    return ret

