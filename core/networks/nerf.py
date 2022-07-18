from .embedding import Optcodes
from .misc import get_confidence_rgb, get_entropy_rgb

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import itertools

# Standard NeRF
class NeRF(nn.Module):
    def __init__(self, 
                 D=8, W=256, input_ch=3, input_ch_bones=0, input_ch_views=3,
                 output_ch=4, skips=[4], use_viewdirs=False, use_framecode=False,
                 framecode_ch=16, n_framecodes=0, 
                 pts_embedder=None, 
                 pe_fn=None, 
                 bones_pe_fn=None,
                 dirs_pe_fn=None,
                 skel_type=None, 
                 view_W=None,
                 density_scale=1.0):
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.view_W = W // 2 if view_W is None else view_W

        self.input_ch = input_ch
        self.input_ch_bones = input_ch_bones
        self.input_ch_views = input_ch_views

        self.skips = skips
        self.use_viewdirs = use_viewdirs

        self.use_framecode = use_framecode
        self.framecode_ch = framecode_ch
        self.n_framecodes = n_framecodes

        self.cam_ch = 1 if self.use_framecode else 0

        self.N_joints = 24
        self.output_ch = output_ch
        self.skel_type = skel_type
        self.density_scale = density_scale

        self.act_fn = nn.ReLU(inplace=True)

        # embedding and PE
        self.pts_embedder = pts_embedder
        self.pe_fn = pe_fn
        self.bones_pe_fn = bones_pe_fn
        self.dirs_pe_fn = dirs_pe_fn

        self.init_density_net()
        self.init_radiance_net()

        ## Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)

    @property
    def dnet_input(self):
        # input size of density network
        return self.input_ch + self.input_ch_bones

    @property
    def vnet_input(self):
        # input size of radiance (view) network
        view_ch_offset = 0 if not self.use_framecode else self.framecode_ch
        size = self.input_ch_views + view_ch_offset + (self.view_W * 2)
        return size

    def init_density_net(self):

        W, D = self.W, self.D

        layers = [nn.Linear(self.dnet_input, W)]

        for i in range(D-1):
            if i not in self.skips:
                layers += [nn.Linear(W, W)]
            else:
                layers += [nn.Linear(W + self.dnet_input, W)]

        self.pts_linears = nn.ModuleList(layers)

        if self.use_viewdirs:
            self.alpha_linear = nn.Linear(W, 1)

    def init_radiance_net(self):


        W, view_W = self.W, self.view_W

        # Note: legacy code, don't really need nn.ModuleList
        self.views_linears = nn.ModuleList([nn.Linear(self.vnet_input, view_W)])

        if self.use_viewdirs:
            self.feature_linear = nn.Linear(W, view_W * 2)
            self.rgb_linear = nn.Linear(view_W, 3)
        else:
            self.output_linear = nn.Linear(W, self.output_ch)

        if self.use_framecode:
            self.framecodes = Optcodes(self.n_framecodes, self.framecode_ch)

    def forward(self, inputs, netchunk=1024*64):

        # Step 1: encode all pts feature
        density_inputs, encoded_pts = self.encode_pts(inputs)

        # Step 2: encode all ray feature
        view_inputs, encoded_views = self.encode_views(inputs, 
                                                       refs=encoded_pts['pts_t'],
                                                       encoded_pts=encoded_pts)

        # Step 3: batchify forward
        shape = inputs['pts'].shape[:2] #(N_rays, N_samples)
        # returns a tuple of raw outputs and encoded feature (if needed)
        outputs = self.inference_batchify(density_inputs, view_inputs, shape, chunk=netchunk)

        return outputs, self.collect_encoded(encoded_pts, encoded_views)
    
    def inference_density_batchify(self, density_inputs, shape, chunk=1024*64):
        outputs_flat = torch.cat([self.inference_density_and_feature(density_inputs[i:i+chunk])
                                  for i in range(0, density_inputs.shape[0], chunk)], -2)
        outputs = outputs_flat.reshape(*shape, outputs_flat.shape[-1])
        return outputs
    
    def inference_rgb_batchify(self, view_inputs, density_feature, shape, chunk=1024*64):
        outputs_flat = torch.cat([self.inference_rgb(view_inputs[i:i+chunk], density_feature[i:i+chunk])
                                  for i in range(0, view_inputs.shape[0], chunk)], -2)
        outputs = outputs_flat.reshape(*shape, outputs_flat.shape[-1])
        return outputs
    
    def inference_density_and_feature(self, density_inputs, with_feature=True):
        '''
        inference density outputs (can be in the form of raw logit)
        '''
        h = self.forward_density(density_inputs)
        # density_preds are raw density ([-inf, inf])
        density_preds = self.alpha_linear(h)
        if with_feature:
            return torch.cat([density_preds, h], dim=-1)
        return density_preds
    
    def inference_rgb(self, view_inputs, density_feature):
        return self.forward_view(view_inputs, density_feature)
    
    def forward_pts(self, inputs, with_feature=False):

        density_inputs, encoded_pts = self.encode_pts(inputs)
        density_preds = self.inference_density_and_feature(density_inputs, with_feature=False)
        return density_preds

    def collect_encoded(self, encoded_pts, encoded_views):
        '''
        Collect encodings for analysis and loss calculation.
        Collect none by default. Overwrite this function as you need.
        '''
        ret = {}
        return ret
    
    def inference_batchify(self, density_inputs, view_inputs, shape, chunk=1024*64):
        '''
        density_inputs: (N_rays * N_samples, ...)
        view_inputs: (N_rays * N_samples, ...)
        shape: tuples of (N_rays, N_samples) to unflatten outputs!
        netchunk: break the batch input smaller sub-batch to avoid OOM
        '''
        outputs_flat = torch.cat([self.inference(density_inputs[i:i+chunk], view_inputs[i:i+chunk])
                                  for i in range(0, density_inputs.shape[0], chunk)], -2)
        outputs = outputs_flat.reshape(*shape, outputs_flat.shape[-1])
        return outputs

    def inference(self, density_inputs, view_inputs):

        h = self.forward_density(density_inputs)

        if self.use_viewdirs:
            # predict density and radiance separately
            alpha = self.alpha_linear(h)
            rgb = self.forward_view(view_inputs, h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs

    def forward_density(self, density_inputs):
        h = density_inputs

        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = self.act_fn(h)
            if i in self.skips:
                h = torch.cat([density_inputs, h], -1)
        return h

    def forward_view(self, view_inputs, density_feature):
        # produce features for color/radiance
        feature = self.feature_linear(density_feature)
        h = torch.cat([feature, view_inputs], -1)

        for i, l in enumerate(self.views_linears):
            h = self.views_linears[i](h)
            h = self.act_fn(h)

        return self.rgb_linear(h)

    """
    def forward_pts(self, inputs, with_feature=False, **kwargs):

        density_inputs, encoded_pts = self.encode_pts(inputs)
        h = self.forward_density(density_inputs)
        alpha = self.alpha_linear(h)
        if with_feature:
            alpha = torch.cat([alpha, h], dim=-1)
        return alpha
    """

    def encode_pts(self, inputs):

        pts = inputs['pts']
        kps = inputs['kps']
        skts = inputs['skts']
        bones = inputs['bones']
        align_transforms = inputs['align_transforms']
        rest_pose = inputs['rest_pose']

        encoded = self.pts_embedder.encode_pts(pts, kps, skts, bones, 
                                               align_transforms, 
                                               rest_pose)
        v, r = encoded['v'], encoded['r']

        # check if you have j_dists computed for cutoff PE
        if 'Dist' in self.pts_embedder.kp_input_fn.encoder_name:
            j_dists = v
        else:
            j_dists = torch.norm(pts[:, :, None] - kps[:, None], dim=-1, p=2)
        encoded['j_dists'] = j_dists

        # apply positional encoding (PE)
        # pe_fn returns a tuple: (encoded outputs, cutoff weights)
        v_pe = self.pe_fn(v, dists=j_dists)[0]
        r_pe = self.bones_pe_fn(r, dists=j_dists)[0]

        density_inputs = torch.cat([v_pe, r_pe], dim=-1).flatten(end_dim=1)

        return density_inputs, encoded
    
    def encode_views(self, inputs, refs, encoded_pts):
        '''
        refs: reference tensor for expanding rays
        encoded_pts: point encoding that could be useful for encoding view
        '''
        rays_o = inputs['rays_o']
        rays_d = inputs['rays_d']
        skts = inputs['skts']
        j_dists = encoded_pts.get('j_dists', None)

        encoded = self.pts_embedder.encode_views(rays_o, rays_d, skts, refs=refs)

        # apply positional encoding (PE)
        d_pe = self.dirs_pe_fn(encoded['d'], dists=j_dists)[0]

        view_inputs = d_pe
        if self.use_framecode:
            N_rays, N_samples = refs.shape[:2]
            # expand from (N_rays, ...) to (N_rays, N_samples, ...)
            cam_idxs = inputs['cam_idxs']
            cam_idxs = cam_idxs.reshape(N_rays, 1, -1).expand(-1, N_samples, -1)
            framecodes = self.framecodes(cam_idxs.reshape(N_rays * N_samples, -1))
            framecodes = framecodes.reshape(N_rays, N_samples, -1)
            view_inputs = torch.cat([view_inputs, framecodes], dim=-1)

        view_inputs = view_inputs.flatten(end_dim=1)

        return view_inputs, encoded

    def raw2outputs(self, raw, z_vals, rays_d, raw_noise_std=0, pytest=False,
                    B=0.01, rgb_act=torch.sigmoid, act_fn=F.relu, rgb_eps=0.001,
                    alpha_w=None, render_confd=False, render_entropy=False, **kwargs):
        """Transforms model's predictions to semantically meaningful values.
        Args:
            raw: [num_rays, num_samples along ray, 4]. Prediction from model.
            z_vals: [num_rays, num_samples along ray]. Integration time.
            rays_d: [num_rays, 3]. Direction of each ray.
        Returns:
            rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
            disp_map: [num_rays]. Disparity map. Inverse of depth map.
            acc_map: [num_rays]. Sum of weights along each ray.
            weights: [num_rays, num_samples]. Weights assigned to each sampled color.
            depth_map: [num_rays]. Estimated distance to object.
        """

        raw2alpha = lambda raw, dists, noise, act_fn=act_fn: 1.-torch.exp(-(act_fn(raw/B + noise))*dists)

        dists = z_vals[...,1:] - z_vals[...,:-1]
        dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]

        # Multiply each distance by the norm of its corresponding direction ray
        # to convert to real world distance (accounts for non-unit directions).
        dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

        if render_confd:
            assert raw.shape[-1] > 4, 'Needs to have confidence/prob logit when render_confd=True'
            rgb = get_confidence_rgb(raw[..., 4:], kwargs['encoded']) 
        elif render_entropy:
            assert raw.shape[-1] > 4, 'Needs to have confidence/prob logit when render_confd=True'
            rgb = get_entropy_rgb(raw[..., 4:], kwargs['encoded']) 
        else:
            rgb = rgb_act(raw[...,:3]) * (1 + 2 * rgb_eps) - rgb_eps # [N_rays, N_samples, 3]
        noise = 0.
        if raw_noise_std > 0.:
            noise = torch.randn(raw[...,3].shape) * raw_noise_std * B

            # Overwrite randomly sampled data if pytest
            if pytest:
                np.random.seed(0)
                noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
                noise = torch.Tensor(noise)

        alpha = raw2alpha(raw[...,3], dists, noise)  # [N_rays, N_samples]
        if alpha_w is not None:
            alpha = alpha_w * alpha

        # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
        # sum_{i=1 to N samples} prob_of_already_hit_particles * alpha_for_i * color_for_i
        # C(r) = sum [T_i * (1 - exp(-sigma_i * delta_i)) * c_i] = sum [T_i * alpha_i * c_i]
        # alpha_i = 1 - exp(-sigma_i * delta_i)
        # T_i = exp(sum_{j=1 to i-1} -sigma_j * delta_j) = torch.cumprod(1 - alpha_i)
        # standard NeRF
        weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
        rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

        depth_map = torch.sum(weights * z_vals, -1)
        disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / (torch.sum(weights, -1)  + 1e-10))

        invalid_mask = torch.ones_like(disp_map)
        invalid_mask[torch.isclose(weights.sum(-1), torch.tensor(0.))] = 0.
        disp_map = disp_map * invalid_mask

        acc_map = torch.minimum(torch.sum(weights, -1), torch.tensor(1.))

        return {'rgb_map': rgb_map, 'disp_map': disp_map, 'acc_map': acc_map,
                'weights': weights, 'alpha': alpha}

    def update_embed_fns(self, global_step, args):
        cutoff_step = args.cutoff_step
        cutoff_rate = args.cutoff_rate

        freq_schedule_step = args.freq_schedule_step
        freq_target = args.multires-1

        if self.pe_fn is not None:
            self.pe_fn.update_threshold(global_step, cutoff_step, cutoff_rate,
                                        freq_schedule_step, freq_target)

        if self.dirs_pe_fn is not None:
            self.dirs_pe_fn.update_threshold(global_step, cutoff_step, cutoff_rate,
                                             freq_schedule_step, freq_target)

        if self.bones_pe_fn is not None:
            self.bones_pe_fn.update_threshold(global_step, cutoff_step, cutoff_rate,
                                                freq_schedule_step, freq_target)

