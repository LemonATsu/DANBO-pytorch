from .nerf import NeRF
from .gnn_backbone import *
from .misc import ParallelLinear, factorize_grid_sample

import torch
import torch.nn as nn
import torch.nn.functional as F

class DANBO(NeRF):

    def __init__(self, *args, node_W=128, input_ch_graph=44, input_ch_voxel=44, voxel_feat=4, voxel_res=4,
                 gcn_D=4, gcn_fc_D=0, gcn_sep_bias=False,
                 graph_pe_fn=None,
                 voxel_pe_fn=None,
                 backbone='PNBGNN',
                 agg_W=16,
                 agg_D=3,
                 rest_pose=None,
                 mask_root=False,
                 align_corners=False,
                 agg_backbone=None,
                 adj_self_one=False,
                 gnn_concat=False,
                 aggregate_dim=None,
                 detach_agg_grad=False,
                 init_adj_w=0.05,
                 attenuate_feat=False,
                 attenuate_invalid=False,
                 opt_scale=False,
                 base_scale=0.5,
                 gnn_n_basis=32,
                 no_adj=False,
                 skel_profile=None,
                 mask_vol_prob=False,
                 use_posecode=False,
                 agg_type='sigmoid',
                 **kwargs):

        self.node_W = node_W
        self.agg_W = agg_W
        self.agg_D = agg_D
        self.skel_type = kwargs['skel_type']
        self.rest_pose = rest_pose
        self.aggregate_dim = aggregate_dim
        self.detach_agg_grad = detach_agg_grad
        self.init_adj_w = init_adj_w

        self.input_ch_graph = input_ch_graph
        self.input_ch_voxel = input_ch_voxel

        self.voxel_feat = voxel_feat
        self.voxel_res = voxel_res
        self.backbone = backbone

        self.align_corners = align_corners
        self.mask_root = mask_root
        self.gcn_D = gcn_D
        self.gcn_fc_D = gcn_fc_D
        self.gcn_sep_bias = gcn_sep_bias
        self.agg_backbone = agg_backbone
        self.adj_self_one = adj_self_one
        self.gnn_concat = gnn_concat

        self.no_adj = no_adj

        # optimize scale for volumes
        self.opt_scale = opt_scale
        self.base_scale = base_scale
        self.skel_profile = skel_profile

        # volume attenuation
        self.attenuate_feat = attenuate_feat
        self.attenuate_invalid = attenuate_invalid
        self.mask_vol_prob = mask_vol_prob
        self.agg_type = agg_type

        self.volume_shape = [len(self.skel_type.joint_trees), self.voxel_feat] + \
                                3 * [self.voxel_res]

        # they will be blended into a single one
        # TODO: ugly hack
        if kwargs['input_ch_views'] % 24 == 0:
            kwargs['input_ch_views'] = kwargs['input_ch_views'] // len(self.skel_type.joint_trees)

        # additional add-on
        self.use_posecode = use_posecode
        self.gnn_n_basis = gnn_n_basis


        super(DANBO, self).__init__(*args, **kwargs)
        self.graph_pe_fn = graph_pe_fn
        self.voxel_pe_fn = voxel_pe_fn

        self.init_graph_net()
        self.init_agg_net()


    @property
    def pts_input_ch(self):
        if self.use_posecode:
            return self.input_ch_voxel + self.voxel_feat * 3 * 4
        return self.input_ch_voxel

    def init_graph_net(self):
        # init a graph network that yields body features
        input_ch_graph = self.input_ch_graph

        self.graph_net = get_volume_gnn_backbone(input_ch_graph, skel_type=self.skel_type,
                                                 gcn_D=self.gcn_D, node_W=self.node_W,
                                                 gcn_fc_D=self.gcn_fc_D,
                                                 gcn_sep_bias=self.gcn_sep_bias,
                                                 voxel_res=self.voxel_res,
                                                 voxel_feat=self.voxel_feat,
                                                 skip_gcn=False, backbone=self.backbone,
                                                 rest_pose=self.rest_pose,
                                                 adj_self_one=self.adj_self_one,
                                                 mask_root=self.mask_root,
                                                 opt_scale=self.opt_scale,
                                                 base_scale=self.base_scale,
                                                 skel_profile=self.skel_profile,
                                                 aggregate_dim=self.aggregate_dim,
                                                 attenuate_feat=self.attenuate_feat,
                                                 attenuate_invalid=self.attenuate_invalid,
                                                 align_corners=self.align_corners,
                                                 no_adj=self.no_adj,
                                                 n_basis=self.gnn_n_basis,
                                                 init_adj_w=self.init_adj_w)

        if self.use_posecode:
            # TODO: this is hard-coded for now
            self.target_joints = np.array([15, 22, 23])
            # assume number of pose == number of frame for simplicity
            self.posecodes = Optcodes(self.n_framecodes, 3 * self.voxel_feat * 4)

    def init_density_net(self):

        self.joint_trees = joint_trees = self.skel_type.joint_trees
        W, D = self.W, self.D

        pts_input_ch = self.pts_input_ch

        layers = [nn.Linear(pts_input_ch, W)]
        for i in range(D-1):
            if i not in self.skips:
                layers += [nn.Linear(W, W)]
            else:
                layers += [nn.Linear(W + pts_input_ch, W)]
        self.pts_linears = nn.ModuleList(layers)

        self.alpha_linear = nn.Linear(W, 1)

    def init_agg_net(self):
        self.joint_trees = joint_trees = self.skel_type.joint_trees
        W, D = self.W, self.D

        prob_out = len(self.joint_trees)
        if self.agg_backbone.startswith('vox'):
            # TODO: lots of hack here ... fix it
            #prob_input = self.dnet_input // len(self.joint_trees) + self.voxel_feat
            if self.backbone.endswith('cat') or self.backbone.startswith('MIFC'):
                prob_input = self.voxel_feat * 3
            else:
                prob_input = self.voxel_feat
            if self.backbone.startswith('PoseCat'):
                prob_input += 3

            N_joints = len(joint_trees)
            if self.agg_backbone == 'vox':
                layers = [ParallelLinear(N_joints, prob_input, self.agg_W), nn.ReLU(inplace=True)]
                for i in range(self.agg_D - 2):
                    layers += [ParallelLinear(N_joints, self.agg_W, self.agg_W), nn.ReLU(inplace=True)]
                layers += [ParallelLinear(N_joints, self.agg_W, 1)]
                self.prob_linears = nn.Sequential(*layers)
            else:
                backbone = '_'.join(self.agg_backbone.split('_')[1:])
                node_W = self.agg_W
                self.prob_linears = get_gnn_backbone(prob_input, skel_type=self.skel_type,
                                                     gcn_D=self.agg_D, node_W=node_W,
                                                     gcn_sep_bias=self.gcn_sep_bias,
                                                     output_ch=1,
                                                     skip_gcn=False, backbone=backbone,
                                                     rest_pose=self.rest_pose,
                                                     init_adj_w=self.init_adj_w,
                                                     no_adj=self.no_adj,
                                                     )
        elif self.agg_backbone.startswith('vol'):
            self.prob_linears = lambda x: x[..., :1]

    # ========= DANBO-specifc forward =========
    def forward_graph(self, x, *args, **kwargs):
        graph_feat = self.graph_net(x)
        if not self.backbone.startswith(('FBGNN', 'FCGNN', 'FGNN', 'CoordCat', 'PoseCat')):
            graph_feat = graph_feat.reshape(-1, *self.volume_shape)
        return graph_feat

    def extract_graph_feat(self, graph_feat, pts_t):
        part_feat, invalid = self.graph_net.sample_from_volume(graph_feat, pts_t,
                                                               need_hessian=False)
        return part_feat, invalid

    def forward_blend_batchify(self, part_feat, valid, chunk=1024*64, **kwargs):
        if self.agg_backbone.endswith('VAN'):
            chunk = part_feat.shape[0]
        return torch.cat([self.forward_blend(part_feat[i:i+chunk], valid[i:i+chunk])
                          for i in range(0, part_feat.shape[0], chunk)], -2)

    def forward_blend(self, part_feat, valid):
        if self.agg_backbone == 'mlp':
            return self.prob_linears(part_feat)

        N_B = part_feat.shape[0]

        agg = self.prob_linears(part_feat, valid=valid)
        if isinstance(agg, tuple):
            agg = agg[1]
        return agg.reshape(N_B, -1)
    # ========= DANBO-specifc forward =========

    def encode_pts(self, inputs, netchunk=1024*64):
        '''
        similar to A-NeRF case, but only do things on graphs
        '''
        pts = inputs['pts']
        kps = inputs['kps']
        skts = inputs['skts']
        bones = inputs['bones']
        align_transforms = inputs['align_transforms']
        rest_pose = inputs['rest_pose']
        N_uniques = inputs['N_uniques']

        N_rays, N_samples = pts.shape[:2]
        N_joints = kps.shape[-2]

        encoded = self.pts_embedder.encode_pts(pts, kps, skts, bones,
                                               align_transforms,
                                               rest_pose)
        pts_t = encoded['pts_t']
        encoded_graph = self.pts_embedder.encode_graph_inputs(kps, bones,
                                                              skts, N_uniques)

        # this is the part that does not need batchify
        w = self.graph_pe_fn(encoded_graph['w'])[0]
        v = self.forward_graph(w)

        """
        for visualization
        import os
        import glob
        saved = sorted(glob.glob('gfeat/*.npy'))
        if len(saved) == 0:
            np.save('gfeat/000', v.cpu().numpy())
        else:
            ckpt = np.load(saved[-1])
            if not np.isclose(ckpt, v.cpu().numpy()).all():
                name = int(saved[-1].split('/')[-1].split('.npy')[0])
                np.save(f'gfeat/{name+1:03d}', v.cpu().numpy())

        for visualization
        """

        h, invalid = self.extract_graph_feat(v, pts_t)

        """
        # TODO: bad practice - short-circuit this function
        if self.cat_all:
            part_feat = part_feat * invalid[..., None]
            blend_feat = part_feat.reshape(N_rays * N_samples, -1)
            blend_view = encoded['d'].reshape(N_rays * N_samples, -1)
            blend_feat = self.embedvoxel_fn(blend_feat)[0]
            blend_view = self.embeddirs_fn(blend_view)[0]
            encoded['blend_feat'] = blend_feat
            encoded['blend_view'] = blend_view
            return encoded
        """

        # predict blend weights and blend features
        h = h.reshape(N_rays * N_samples, N_joints, -1)
        valid = (1 - invalid).reshape(N_rays * N_samples, N_joints)
        a = self.forward_blend_batchify(h, valid, netchunk)

        """
        if not self.cat_coords:
            a = self.forward_blend_batchify(h, valid, netchunk)
            blend_logit = network.forward_blend_batchify(h, valid, netchunk)
        else:
            cat_feat = h
            part_feat = part_feat[..., :-3] # the last 3 dim are coords
            blend_logit = network.forward_blend_batchify(cat_feat, valid, netchunk)
        """
        p = self.get_agg(a, invalid)

        # if we want to do mc sampling
        """
        if self.mc_sampling and (importance or self.mc_all):
            blend_prob = self.get_mc_sample(blend_prob, encoded, self.mc_eps)
        """

        # to (N_rays * N_samples, N_joints, N_features)
        h = h.reshape(N_rays * N_samples, N_joints, -1)
        h = (h * p[..., None]).sum(-2)

        density_inputs = self.voxel_pe_fn(h)[0]

        if self.use_posecode:
            cam_idxs = inputs['cam_idxs'].reshape(N_rays, 1, -1).expand(N_rays, N_samples, -1)
            posecodes = self.posecodes(cam_idxs.reshape(N_rays * N_samples, -1))
            posecodes = posecodes.reshape(N_rays * N_samples, 3, -1) * valid[..., self.target_joints, None]
            posecodes = posecodes.reshape(N_rays * N_samples, -1)
            density_inputs = torch.cat([density_inputs, posecodes], dim=-1)

        """
        blend_feat, blend_view, blend_pts, _ = network.blend_inputs(
                                                      part_feat,
                                                      d,
                                                      blend_prob,
                                                      pts_rp=pts_rp.flatten(end_dim=1),
                                                  )
        """

        # if we want to use different input to NeRF network
        """
        if self.input_coords:
            blend_feat = blend_pts
        if self.cat_coords:
            blend_feat = torch.cat([blend_feat, blend_pts], dim=-1)

        blend_feat = self.embedvoxel_fn(blend_feat)[0]
        if blend_view is not None:
            blend_view = self.embeddirs_fn(blend_view)[0]

        encoded['blend_feat'] = blend_feat
        encoded['blend_view'] = blend_view
        encoded['blend_logit'] = blend_logit
        """
        encoded['confd'] = a.reshape(N_rays, N_samples, N_joints)
        encoded['part_invalid'] = invalid
        encoded['agg_p'] = p

        return  density_inputs, encoded

    def collect_encoded(self, encoded_pts, encoded_views):
        ret = super(DANBO, self).collect_encoded(encoded_pts, encoded_views)
        if 'confd' in encoded_pts:
            ret['confd'] = encoded_pts['confd']
        ret['part_invalid'] = encoded_pts['part_invalid']
        return ret

    def blend_inputs(self, part_feat, part_view, blend_prob, per_joint_codes=None, pts_rp=None, **kwargs):
        '''
        of size (N_B * N_samples, *)
        assume blend_prob already normalized to become a distribution
        '''
        N_B = part_feat.shape[0] # N_B * N_samples
        N_joints = len(self.skel_type.joint_names)


        if self.gnn_concat:
            # no blending for now
            blend_feat = (blend_prob[..., None] * part_feat).flatten(start_dim=1)
        else:
            blend_feat = (blend_prob[..., None] * part_feat).sum(-2)

        # assume last dim is 3
        blend_view = None
        if part_view is not None:
            part_view = part_view.reshape(N_B, -1, 3)
            # only blend when we are actually using reldir
            if part_view.shape[1] > 1:
                blend_view = (blend_prob[..., None] * part_view).sum(-2)
                blend_view = F.normalize(blend_view, dim=-1)
            else:
                blend_view = part_view.reshape(N_B, 3)

        blend_codes = None

        blend_pts = None
        if pts_rp is not None:
            blend_pts = (blend_prob[..., None] * pts_rp).sum(-2)

        return blend_feat, blend_view, blend_pts, blend_codes

    def get_adjw(self):
        adjw_list = self.graph_net.get_adjw()
        if hasattr(self.prob_linears, 'get_adjw'):
            adjw_list += self.prob_linears.get_adjw()
        return adjw_list

    def softmax(self, logit, invalid, eps=1e-7, temp=1.0):
        if not self.mask_vol_prob:
            return F.softmax(logit / temp, dim=-1)

        logit = logit / temp

        invalid = invalid.flatten(end_dim=-2)
        # find the valid part
        valid = 1 - invalid
        # for stability: doesn't change the output as the term will be canceled out
        max_logit = logit.max(dim=-1, keepdim=True)[0]

        # only keep the valid part!
        nominator = torch.exp(logit - max_logit) * valid
        denominator = torch.sum(nominator + eps, dim=-1, keepdim=True)

        return nominator / denominator.clamp(min=eps)

    def sigmoid(self, logit, invalid,
                mask_invalid=True, clamp=True,
                eps=1e-7, sigmoid_eps=0.001):

        p = torch.sigmoid(logit) * (1 + 2 * sigmoid_eps) - sigmoid_eps
        if mask_invalid:
            invalid = invalid.flatten(end_dim=-2)
            valid = 1 - invalid
            p = p * valid
        return p

    def relu_prob(self, logit, invalid, eps=1e-7, noise_std=1.0):

        if self.training:
            # add some noises for exploration
            noise = torch.randn_like(logit) * noise_std
            logit = logit + noise
        invalid = invalid.flatten(end_dim=-2)
        valid = 1 - invalid
        logit = F.relu(logit * valid)
        # note: this is non-differentiable selection.
        p = (logit > 0).float()
        return p


    def get_agg(self, logit, invalid, eps=1e-7):
        if self.agg_type == 'softmax':
            return self.softmax(logit, invalid, eps=eps)
        if self.agg_type == 'sigmoid':
            return self.sigmoid(logit, invalid, eps=eps)
        if self.agg_type == 'relu':
            return self.relu_prob(logit, invalid, eps=eps)
        if self.agg_type == 'sum':
            return torch.ones_like(logit)
        raise NotImplementedError(f'agg_type {self.agg_type} is not implemented.')
