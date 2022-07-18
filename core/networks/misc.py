import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List

import numpy as np
import math
import time

class PerNodeValidMLP(nn.Module):
    '''
    with out-of-bound handling, only for voxel aggregation network
    '''
    
    def __init__(self, input_ch=32, output_ch=1, W=32, n_nodes=24, D=2, input_relu=True):
        '''
        input_relu: apply activation on the input
        '''
        super().__init__()
        self.input_ch = input_ch
        self.output_ch = output_ch
        self.W = W
        self.D = D
        self.n_nodes = n_nodes

        networks = []

        for n in range(self.n_nodes):
            network = []
            if input_relu:
                network = [nn.ReLU(inplace=True)]
            if D == 1:
                network += [nn.Linear(input_ch, output_ch)]
            else:
                network += [nn.Linear(input_ch, W), nn.ReLU(inplace=True)]
                for d in range(D-2):
                    network += [nn.Linear(W, W), nn.ReLU(inplace=True)]
                network += [nn.Linear(W, output_ch)]
            networks.append(nn.Sequential(*network))
        self.networks = nn.ModuleList(networks)

    def forward(self, h, valid):
        N_samples = h.shape[0]
        outputs = torch.zeros(N_samples, self.n_nodes, self.output_ch)

        #valid_idx = torch.where(valid[:, 0] > 0)[0]
        for n in range(self.n_nodes): # run forward for each volume
            valid_idx = torch.where(valid[:, n] > 0)[0]
            a = self.networks[n](h[valid_idx, n])
            outputs[valid_idx, n] = a
        
        return outputs

class GroupedPNMLP(nn.Module):

    def __init__(self, input_ch=32, output_ch=1, W=32, n_nodes=24, D=2, 
                 n_groups=6, input_relu=True):
        '''
        input_relu: apply activation on the input
        '''
        super().__init__()
        self.input_ch = input_ch
        self.output_ch = output_ch
        self.W = W
        self.D = D
        self.n_nodes = n_nodes
        self.n_groups = n_groups
        self.n_nodes_per_group = self.n_nodes // self.n_groups
        n_nodes_per_group = self.n_nodes_per_group

        networks = []

        for n in range(self.n_groups):
            network = []
            if input_relu:
                network = [nn.ReLU(inplace=True)]
            if D == 1:
                network += [ParallelLinear(n_nodes_per_group, input_ch, output_ch)]
            else:
                network += [ParallelLinear(n_nodes_per_group, input_ch, W), nn.ReLU(inplace=True)]
                for d in range(D-2):
                    network += [ParallelLinear(n_nodes_per_group, W, W), nn.ReLU(inplace=True)]
                network += [ParallelLinear(n_nodes_per_group, W, output_ch)]
            networks.append(nn.Sequential(*network))
        self.networks = nn.ModuleList(networks)
        # TODO: hard-coded for now
        self.register_buffer('grouping',
            torch.tensor([
                [0, 3, 6, 9], # hip to spine
                [1, 4, 7, 10], # left legs
                [2, 5, 8, 11], # right legs
                [12, 13, 14, 15], # head, neck and collars
                [16, 18, 20, 22], # left hands
                [17, 19, 21, 23], # right hands
            ]).long())
        
        mapping = []
        grouping = self.grouping.cpu().numpy().reshape(-1)
        for i in range(self.n_nodes):
            for j, num in enumerate(grouping):
                if i == num:
                    mapping.append(j)
                    break

    def forward(self, h, valid):
        N_samples = h.shape[0]
        outputs = torch.zeros(N_samples, self.n_groups, self.n_nodes_per_group, self.output_ch)
        sorted_outputs = torch.zeros(N_samples, self.n_nodes, self.output_ch)
        import pdb; pdb.set_trace()
        print

        #valid_idx = torch.where(valid[:, 0] > 0)[0]
        for n in range(self.n_groups): # run forward for each volume
            group_idxs = self.grouping[n]
            group_valid = valid[:, group_idxs].sum(-1)
            valid_idxs = torch.where(group_valid > 0)[0]
            selected_h = h[valid_idxs]
            a = self.networks[n](selected_h[:, group_idxs])
            outputs[valid_idxs, n] = a
        
        scatter_idxs = self.grouping.reshape(1, -1, 1).expand(N_samples, -1, 1)
        outputs = outputs.reshape(N_samples, -1, self.output_ch)
        sorted_outputs.scatter_(1, scatter_idxs, outputs)
        
        return sorted_outputs



class ParallelLinear(nn.Module):

    def __init__(self, n_parallel, in_feat, out_feat, share=False, bias=True):
        super().__init__()
        self.n_parallel = n_parallel
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.share = share

        if not self.share:
            self.register_parameter('weight',
                                    nn.Parameter(torch.randn(n_parallel, in_feat, out_feat),
                                                 requires_grad=True)
                                   )
            if bias:
                self.register_parameter('bias',
                                        nn.Parameter(torch.randn(1, n_parallel, out_feat),
                                                     requires_grad=True)
                                       )
        else:
            self.register_parameter('weight', nn.Parameter(torch.randn(1, in_feat, out_feat),
                                                           requires_grad=True))
            if bias:
                self.register_parameter('bias', nn.Parameter(torch.randn(1, 1, out_feat), requires_grad=True))
        if not hasattr(self, 'bias'):
            self.bias = None
        #self.bias = nn.Parameter(torch.Tensor(n_parallel, 1, out_feat))
        self.reset_parameters()
        """
        self.conv = nn.Conv1d(in_feat * n_parallel, out_feat * n_parallel,
                              kernel_size=1, groups=n_parallel, bias=bias)
        """

    def reset_parameters(self):

        for n in range(self.n_parallel):
            # transpose because the weight order is different from nn.Linear
            nn.init.kaiming_uniform_(self.weight[n].T.data, a=math.sqrt(5))

        if self.bias is not None:
            #fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[0].T)
            #bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            #nn.init.uniform_(self.bias, -bound, bound)
            nn.init.constant_(self.bias.data, 0.)

    def forward(self, x):
        weight, bias = self.weight, self.bias
        if self.share:
            weight = weight.expand(self.n_parallel, -1, -1)
            if bias is not None:
                bias = bias.expand(-1, self.n_parallel, -1)
        out = torch.einsum("bkl,klj->bkj", x, weight.to(x.device))
        if bias is not None:
            out = out + bias.to(x.device)
        return out

    def extra_repr(self):
        return "n_parallel={}, in_features={}, out_features={}, bias={}".format(
            self.n_parallel, self.in_feat, self.out_feat, self.bias is not None
        )

class HyperParallelLinear(nn.Module):

    def __init__(self, n_parallel, in_feat, out_feat):
        super().__init__()
        self.n_parallel = n_parallel
        self.in_feat = in_feat
        self.out_feat = out_feat
    
    def forward(self, x, weight, bias=None):
        weight = weight.reshape(self.n_parallel, self.in_feat, self.out_feat)
        out = torch.einsum("bkl,klj->bkj", x, weight.to(x.device))
        if bias is not None:
            bias = bias.reshape(1, self.n_parallel, self.out_feat)
            out = out + bias.to(x.device)
        return out

class SymmetryLinear(nn.Module):

    def __init__(self, skel_type, in_feat, out_feat, share=False, bias=True):
        super().__init__()
        self.skel_type = skel_type
        self.in_feat = in_feat
        self.out_feat = out_feat

        # build symmetry map
        joint_names = skel_type.joint_names
        part_names = {}
        part_idxs = []
        part_indicators = []
        count = 0
        for i, name in enumerate(joint_names):
            is_left = name.startswith('left')
            is_right = name.startswith('right')
            is_unique = not (is_left or is_right)

            if is_unique:
                part_names[name] = count
                part_idxs.append(part_names[name])
                count += 1
                part_indicators.append(0)
            else:
                part_name = name.split('_')[-1]
                if not part_name in part_names:
                    part_names[part_name] = count
                    count += 1
                part_idxs.append(part_names[part_name])
                part_indicators.append(1 if is_left else -1)
        self.part_names = part_names
        self.part_idxs = part_idxs
        self.n_uniques = len(self.part_names)

        # in_feat + 1 for indicator
        self.register_parameter('weight',
                                nn.Parameter(torch.randn(self.n_uniques, in_feat+1, out_feat),
                                             requires_grad=True)
                                )
        if bias:
            self.register_parameter('bias',
                                    nn.Parameter(torch.randn(1, self.n_uniques, out_feat),
                                                 requires_grad=True)
                                   )
        else:
            self.bias = None

        self.register_buffer('part_indicators',
                             torch.tensor(part_indicators).reshape(1, -1, 1),
                            )
        self.reset_parameters()


    def forward(self, x):
        ind = self.part_indicators.expand(x.shape[0], -1, -1)
        weight = self.weight[self.part_idxs]
        out = torch.einsum("bkl,klj->bkj", torch.cat([ind, x], dim=-1), weight.to(x.device))

        if self.bias is not None:
            out = out + self.bias[:, self.part_idxs].to(out.device)
        return out

    def reset_parameters(self):

        for n in range(self.n_uniques):
            # transpose because the weight order is different from nn.Linear
            nn.init.kaiming_uniform_(self.weight[n].T.data, a=math.sqrt(5))

        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[0].T)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def extra_repr(self):
        return "n_uniques={}, in_features={}, out_features={}, bias={}".format(
                self.n_uniques, self.in_feat+1, self.out_feat, self.bias is not None
         )

class ParallelFactorizeCNN(nn.Module):
    
    def __init__(self, N_joints, in_channel, voxel_feat=5):
        super(ParallelFactorizeCNN, self).__init__()

        self.N_joints = N_joints
        self.in_channel = in_channel
        self.voxel_res = 10
        self.voxel_feat = voxel_feat
        self.out_channel = voxel_feat * 3
       
        self.init_cnn()
        
    def init_cnn(self):
        groups = self.N_joints
        in_channel, out_channel = self.in_channel, self.out_channel
        # only works for voxel_res == 10 for now
        conv1 = nn.ConvTranspose1d(in_channel * groups, 96 * groups, kernel_size=3, stride=1, padding=0, groups=groups)
        conv2 = nn.ConvTranspose1d(96 * groups, 64 * groups, kernel_size=3, stride=2, padding=0, groups=groups)
        conv3 = nn.ConvTranspose1d(64 * groups, 64 * groups, kernel_size=2, stride=1, padding=0, groups=groups)
        conv4 = nn.ConvTranspose1d(64 * groups, out_channel * groups, kernel_size=3, stride=1, padding=0, groups=groups)
        
        self.net = nn.Sequential(
            conv1,
            nn.LeakyReLU(0.1, inplace=True),
            conv2,
            nn.LeakyReLU(0.1, inplace=True),
            conv3,
            nn.LeakyReLU(0.1, inplace=True),
            conv4,
        )


    def forward(self, x):
        N_B, N_J, N_feat = x.shape
        x = x.reshape(N_B, N_J * N_feat, -1)
        out = self.net(x).reshape(N_B, N_J, -1, 3, self.voxel_res)
        out = out.transpose(-1, -2)
        return out.flatten(start_dim=-3)

    def extra_repr(self):
        return "n_parallel={}, in_features={}, out_features={}x{}x{}".format(
            self.N_joints, self.in_channel, self.voxel_feat, self.voxel_res, 3
        )


def factorize_grid_sample(features, coords, align_corners=False, mode='bilinear', padding_mode='zeros', 
                          training=False, need_hessian=False):
    '''
    Factorized grid sample: only gives the same outcomes as the original one under certain circumstances.
    '''
    bnd = 1.0 if align_corners else 2.0 / 3.0
    # cols are meant to make grid_samples axis-independent
    cols = torch.linspace(-bnd, bnd, 3)
    coords = coords[..., None]
    cols = cols.reshape(1, 1, 3, 1).expand(*coords.shape[:2], -1, -1)
    coords = torch.cat([cols, coords], dim=-1)
    # the default grid_sample in pytorch does not have 2nd order derivative
    if training and need_hessian:
        sample_feature = grid_sample_diff(features, coords, padding_mode=padding_mode, align_corners=align_corners)
        #sample_feature = F.grid_sample(features, coords, mode=mode,
        #                               align_corners=align_corners, padding_mode=padding_mode)
    else:
        sample_feature = F.grid_sample(features, coords, mode=mode,
                                       align_corners=align_corners, padding_mode=padding_mode)
    #return sample_feature
    return sample_feature

def grid_sample_diff(image, optical, padding_mode='zero', align_corners=False, eps=1e-7, clamp_x=True):
    N, C, IH, IW = image.shape
    _, H, W, _ = optical.shape

    ix = optical[..., 0]
    iy = optical[..., 1]

    if align_corners:
        ix = ((ix + 1.) / 2.) * (IW-1)
        iy = ((iy + 1.) / 2.) * (IH-1)
    else:
        ix = ((ix + 1.) * IW - 1) / 2.
        iy = ((iy + 1.) * IH - 1) / 2.
    

    with torch.no_grad():
        iy_nw = torch.floor(iy)
        iy_ne = iy_nw
        iy_sw = iy_nw + 1
        iy_se = iy_nw + 1

        if clamp_x:
            # this is a special case: our x is used as an indicator,
            # so it should always be integer value
            ix = ix.round()
            ix_nw = ix
            ix_ne = ix_nw + 1 
            ix_sw = ix_nw
            ix_se = ix_nw + 1
        else:
            ix_nw = torch.floor(ix)
            ix_ne = ix_nw + 1
            ix_sw = ix_nw
            ix_se = ix_nw + 1

    nw = (ix_se - ix)    * (iy_se - iy)
    ne = (ix    - ix_sw) * (iy_sw - iy)
    sw = (ix_ne - ix)    * (iy    - iy_ne)
    se = (ix    - ix_nw) * (iy    - iy_nw)
    
    valid_nw = torch.ones_like(iy_nw)
    valid_ne = torch.ones_like(iy_ne)
    valid_sw = torch.ones_like(iy_sw)
    valid_se = torch.ones_like(iy_se)
    if padding_mode == 'zeros':

        bnd_z = 0 - eps
        bnd_W = IW - 1 + eps
        bnd_H = IH - 1 + eps

        valid_nw[ix_nw < bnd_z] = 0.
        valid_nw[ix_nw > bnd_W] = 0.
        valid_nw[iy_nw < bnd_z] = 0.
        valid_nw[iy_nw > bnd_H] = 0.

        valid_ne[ix_ne < bnd_z] = 0.
        valid_ne[ix_ne > bnd_W] = 0.
        valid_ne[iy_ne < bnd_z] = 0.
        valid_ne[iy_ne > bnd_H] = 0.

        valid_sw[ix_sw < bnd_z] = 0.
        valid_sw[ix_sw > bnd_W] = 0.
        valid_sw[iy_sw < bnd_z] = 0.
        valid_sw[iy_sw > bnd_H] = 0.

        valid_se[ix_se < bnd_z] = 0.
        valid_se[ix_se > bnd_W] = 0.
        valid_se[iy_se < bnd_z] = 0.
        valid_se[iy_se > bnd_H] = 0.

        valid_nw = valid_nw.view(N, -1, H * W)
        valid_ne = valid_ne.view(N, -1, H * W)
        valid_sw = valid_sw.view(N, -1, H * W)
        valid_se = valid_se.view(N, -1, H * W)

    with torch.no_grad():
        torch.clamp(ix_nw, 0, IW-1, out=ix_nw)
        torch.clamp(iy_nw, 0, IH-1, out=iy_nw)

        torch.clamp(ix_ne, 0, IW-1, out=ix_ne)
        torch.clamp(iy_ne, 0, IH-1, out=iy_ne)

        torch.clamp(ix_sw, 0, IW-1, out=ix_sw)
        torch.clamp(iy_sw, 0, IH-1, out=iy_sw)

        torch.clamp(ix_se, 0, IW-1, out=ix_se)
        torch.clamp(iy_se, 0, IH-1, out=iy_se)



    image = image.view(N, C, IH * IW)

    nw_val = torch.gather(image, 2, (iy_nw * IW + ix_nw).long().view(N, 1, H * W).repeat(1, C, 1))
    ne_val = torch.gather(image, 2, (iy_ne * IW + ix_ne).long().view(N, 1, H * W).repeat(1, C, 1))
    sw_val = torch.gather(image, 2, (iy_sw * IW + ix_sw).long().view(N, 1, H * W).repeat(1, C, 1))
    se_val = torch.gather(image, 2, (iy_se * IW + ix_se).long().view(N, 1, H * W).repeat(1, C, 1))
    if padding_mode == 'zeros':
        nw_val = nw_val * valid_nw
        ne_val = ne_val * valid_ne
        sw_val = sw_val * valid_sw
        se_val = se_val * valid_se

    out_val = (nw_val.view(N, C, H, W) * nw.view(N, 1, H, W) + 
               ne_val.view(N, C, H, W) * ne.view(N, 1, H, W) +
               sw_val.view(N, C, H, W) * sw.view(N, 1, H, W) +
               se_val.view(N, C, H, W) * se.view(N, 1, H, W))

    return out_val 

def unrolled_propagate(adj, w):
    '''
    Unrolled adjacency propagation (to save memory and maybe computation)
    '''
    o0 = adj[0, 0, 0] * w[:, 0] + adj[0, 0, 1] * w[:, 1] + adj[0, 0, 2] * w[:, 2] + adj[0, 0, 3] * w[:, 3]
    o1 = adj[0, 1, 1] * w[:, 1] + adj[0, 1, 0] * w[:, 0] + adj[0, 1, 4] * w[:, 4]
    o2 = adj[0, 2, 2] * w[:, 2] + adj[0, 2, 0] * w[:, 0] + adj[0, 2, 5] * w[:, 5]
    o3 = adj[0, 3, 3] * w[:, 3] + adj[0, 3, 0] * w[:, 0] + adj[0, 3, 6] * w[:, 6]
    o4 = adj[0, 4, 4] * w[:, 4] + adj[0, 4, 1] * w[:, 1] + adj[0, 4, 7] * w[:, 7]
    o5 = adj[0, 5, 5] * w[:, 5] + adj[0, 5, 2] * w[:, 2] + adj[0, 5, 8] * w[:, 8]
    o6 = adj[0, 6, 6] * w[:, 6] + adj[0, 6, 3] * w[:, 3] + adj[0, 6, 9] * w[:, 9]
    o7 = adj[0, 7, 7] * w[:, 7] + adj[0, 7, 4] * w[:, 4] + adj[0, 7, 10] * w[:, 10]
    o8 = adj[0, 8, 8] * w[:, 8] + adj[0, 8, 5] * w[:, 5] + adj[0, 8, 11] * w[:, 11]
    o9 = adj[0, 9, 9] * w[:, 9] + adj[0, 9, 6] * w[:, 6] + adj[0, 9, 12] * w[:, 12] + adj[0, 9, 13] * w[:, 13] + adj[0, 9, 14] * w[:, 14]
    o10 = adj[0, 10, 10] * w[:, 10] + adj[0, 10, 7] * w[:, 7]
    o11 = adj[0, 11, 11] * w[:, 11] + adj[0, 11, 8] * w[:, 8]
    o12 = adj[0, 12, 12] * w[:, 12] + adj[0, 12, 9] * w[:, 9] + adj[0, 12, 15] * w[:, 15]
    o13 = adj[0, 13, 13] * w[:, 13] + adj[0, 13, 9] * w[:, 9] + adj[0, 13, 16] * w[:, 16]
    o14 = adj[0, 14, 14] * w[:, 14] + adj[0, 14, 9] * w[:, 9] + adj[0, 14, 17] * w[:, 17]
    o15 = adj[0, 15, 15] * w[:, 15] + adj[0, 15, 12] * w[:, 12]
    o16 = adj[0, 16, 16] * w[:, 16] + adj[0, 16, 13] * w[:, 13] + adj[0, 16, 18] * w[:, 18]
    o17 = adj[0, 17, 17] * w[:, 17] + adj[0, 17, 14] * w[:, 14] + adj[0, 17, 19] * w[:, 19]
    o18 = adj[0, 18, 18] * w[:, 18] + adj[0, 18, 16] * w[:, 16] + adj[0, 18, 20] * w[:, 20]
    o19 = adj[0, 19, 19] * w[:, 19] + adj[0, 19, 17] * w[:, 17] + adj[0, 19, 21] * w[:, 21]
    o20 = adj[0, 20, 20] * w[:, 20] + adj[0, 20, 18] * w[:, 18] + adj[0, 20, 22] * w[:, 22]
    o21 = adj[0, 21, 21] * w[:, 21] + adj[0, 21, 19] * w[:, 19] + adj[0, 21, 23] * w[:, 23]
    o22 = adj[0, 22, 22] * w[:, 22] + adj[0, 22, 20] * w[:, 20]
    o23 = adj[0, 23, 23] * w[:, 23] + adj[0, 23, 21] * w[:, 21]
    o = torch.stack([o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10,
                     o11, o12, o13, o14, o15, o16, o17, o18, o19,
                     o20, o21, o22, o23], dim=1)


    return o

class ResMLP(nn.Module):

    def __init__(self, in_channel, W, out_channel):
        super().__init__()

        self.layer0 = nn.Linear(in_channel, W)
        self.layer1 = nn.Linear(W, W)
        self.layer2 = nn.Linear(W, W)
        self.layer3 = nn.Linear(W, out_channel)

    def forward(self, x):

        x0 = self.layer0(x)
        x1 = self.layer1(F.relu(x0, inplace=True))
        x2 = self.layer2(F.relu(x1, inplace=True))
        x3 = self.layer3(F.relu(x2 + x0, inplace=True))
        return x3

# steal from MVP
def blockinit(k, stride):
    dim = k.ndim - 2
    return k \
            .view(k.size(0), k.size(1), *(x for i in range(dim) for x in (k.size(i+2), 1))) \
            .repeat(1, 1, *(x for i in range(dim) for x in (1, stride))) \
            .view(k.size(0), k.size(1), *(k.size(i+2)*stride for i in range(dim)))

# steal from MVP
class ConvTranspose2dELR(nn.Module):
    def __init__(self, inch, outch, kernel_size, stride, padding, affinelrmult=1., 
                 norm=None, ub=None, act=None, groups=1):
        super(ConvTranspose2dELR, self).__init__()

        self.inch = inch
        self.outch = outch
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.norm = norm
        self.ub = ub
        self.act = act
        self.groups = groups

        # compute gain from activation fn
        try:
            if isinstance(act, nn.LeakyReLU):
                actgain = nn.init.calculate_gain("leaky_relu", act.negative_slope)
            elif isinstance(act, nn.ReLU):
                actgain = nn.init.calculate_gain("relu")
            else:
                actgain = nn.init.calculate_gain(act)
        except:
            actgain = 1.

        fan_in = inch * (kernel_size ** 2 / (stride ** 2))

        initgain = stride if norm == "demod" else 1. / math.sqrt(fan_in)

        self.weightgain = actgain * initgain

        self.weight = nn.Parameter(blockinit(
            torch.randn(inch, outch // self.groups, kernel_size//self.stride, kernel_size//self.stride), self.stride))

        if ub is not None:
            self.bias = nn.Parameter(torch.zeros(outch, ub[0], ub[1]))
        else:
            self.bias = nn.Parameter(torch.zeros(outch))

        self.fused = False

    def extra_repr(self):
        return 'inch={}, outch={}, kernel_size={}, stride={}, padding={}, norm={}, ub={}, act={}'.format(
            self.inch, self.outch, self.kernel_size, self.stride, self.padding, self.norm, self.ub, self.act
        )

    def getweight(self, weight):
        if self.fused:
            return weight
        else:
            if self.norm is not None:
                if self.norm == "demod":
                    if weight.ndim == 5:
                        normdims = [1, 3, 4]
                    else:
                        normdims = [0, 2, 3]

                    if torch.jit.is_scripting():
                        # scripting doesn't support F.normalize(..., dim=list[int])
                        weight = weight / torch.linalg.vector_norm(weight, dim=normdims, keepdim=True)
                    else:
                        weight = F.normalize(weight, dim=normdims)

            weight = weight * self.weightgain

            return weight

    def fuse(self):
        with torch.no_grad():
            self.weight.data = self.getweight(self.weight)
        self.fused = True

    def forward(self, x, w : Optional[torch.Tensor]=None):
        b = x.size(0)

        weight = self.weight

        weight = self.getweight(weight)

        groups = self.groups

        out = F.conv_transpose2d(x, weight, None,
                stride=self.stride, padding=self.padding, dilation=1, groups=groups)

        if self.bias.ndim == 1:
            bias = self.bias[None, :, None, None]
        else:
            bias = self.bias[None, :, :, :]
        out = out + bias

        if self.act is not None:
            out = self.act(out)

        return out

def get_entropy_rgb(confd, encoded, eps=1e-7):
    """
    if 'part_invalid' in encoded:
        part_valid = 1 - encoded['part_invalid']
        max_logit = (part_valid * confd).max(dim=-1, keepdim=True)[0]
        nominator = torch.exp(confd - max_logit) * part_valid
        denominator = torch.sum(nominator + eps, dim=-1, keepdim=True)
        prob = nominator / denominator
    else:
        prob = F.softmax(confd, dim=-1)
    """
    prob = F.softmax(confd, dim=-1)
    max_ent = torch.tensor(confd.shape[-1]).log()
    ent = -(prob * (prob + eps).log()).sum(-1)

    ratio = (ent / max_ent)[..., None]
    start = torch.tensor([0., 0., 1.]).reshape(1, 1, 3)
    end = torch.tensor([1., 0., 0.]).reshape(1, 1, 3)
    rgb = torch.lerp(start, end, ratio)
 
    return rgb


def get_confidence_rgb(confd, encoded):
    # TODO: currently assume skeleton is SMPL!
    # pre-defined 24 colors
    colors = torch.tensor([[1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 0.5019607843137255, 0.0],
            [0.29411764705882354, 0.0, 0.5098039215686274],
            [1.0, 0.5490196078431373, 0.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.7529411764705882, 0.796078431372549],
            [0.6039215686274509, 0.803921568627451, 0.19607843137254902],
            [0.7372549019607844, 0.5607843137254902, 0.5607843137254902],
            [1.0, 0.4980392156862745, 0.3137254901960784],
            [0.8235294117647058, 0.4117647058823529, 0.11764705882352941],
            [1.0, 0.8941176470588236, 0.7686274509803922],
            [1.0, 0.8431372549019608, 0.0],
            [0.6039215686274509, 0.803921568627451, 0.19607843137254902],
            [0.4980392156862745, 1.0, 0.8313725490196079],
            [0.0, 0.7490196078431373, 1.0],
            [0.0, 0.0, 0.5019607843137255],
            [0.8549019607843137, 0.4392156862745098, 0.8392156862745098],
            [0.5019607843137255, 0.0, 0.0],
            [0.6274509803921569, 0.3215686274509804, 0.17647058823529413],
            [0.5019607843137255, 0.5019607843137255, 0.0],
            [0.5647058823529412, 0.9333333333333333, 0.5647058823529412],
            ])
    selected_color = confd.argmax(dim=-1)
    rgb = colors[selected_color]
    return rgb

def init_volume_scale(base_scale, skel_profile, skel_type):
    # TODO: hard-coded some parts for now ...
    # TODO: deal with multi-subject
    joint_names = skel_type.joint_names
    N_joints = len(joint_names)
    bone_lens = skel_profile['bone_lens'][0]
    bone_lens_to_child = skel_profile['bone_lens_to_child'][0]

    # indices to all body parts
    head_idxs = skel_profile['head_idxs']
    torso_idxs = skel_profile['torso_idxs']
    arm_idxs = skel_profile['arm_idxs']
    leg_idxs = skel_profile['leg_idxs']

    # some widths
    shoulder_width = skel_profile['shoulder_width'][0]
    knee_width = skel_profile['knee_width'][0]
    collar_width = skel_profile['knee_width'][0]

    # init the scale for x, y and z
    # width, depth
    x_lens = torch.ones(N_joints) * base_scale
    y_lens = torch.ones(N_joints) * base_scale

    # half-width of thighs cannot be wider than the distant between knees in rest pose
    x_lens[leg_idxs] = knee_width * 0.5
    y_lens[leg_idxs] = knee_width * 0.5

    #  half-width of your body and head cannot be wider than shoulder distance (to some scale) 
    x_lens[torso_idxs] = shoulder_width * 0.70
    y_lens[torso_idxs] = shoulder_width * 0.70
    x_lens[head_idxs] = shoulder_width * 0.60
    y_lens[head_idxs] = shoulder_width * 0.60

    #  half-width of your arms cannot be wider than collar distance (to some scale) 
    x_lens[arm_idxs] = collar_width * 0.60
    y_lens[arm_idxs] = collar_width * 0.60

    # set scale along the bone direction
    # don't need full bone lens because the volume is supposed to centered at the middle of a bone
    z_lens = torch.tensor(bone_lens_to_child.copy().astype(np.float32))
    z_lens = z_lens * 0.8

    # deal with end effectors: make them grow freely
    z_lens[z_lens < 0] = z_lens.max()
    # give more space to head as we do not have head-top joint
    z_lens[head_idxs] = z_lens.max() * 1.1 
    scale = torch.stack([x_lens, y_lens, z_lens], dim=-1)

    return scale
