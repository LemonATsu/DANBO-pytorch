from .embedding import Optcodes
from .misc import HyperParallelLinear, ParallelLinear, SymmetryLinear, \
                  unrolled_propagate, factorize_grid_sample, \
                  ParallelFactorizeCNN, ConvTranspose2dELR, init_volume_scale, \
                  PerNodeValidMLP, GroupedPNMLP

import torch
import torch.nn as nn
import torch.nn.functional as F
#import torch_geometric.nn as gnn
#from torch_geometric.nn.dense.linear import Linear
import numpy as np
import time

'''
Modified from Skeleton-aware Networks https://github.com/DeepMotionEditing/deep-motion-editing
'''
def skeleton_to_graph(skel=None, edges=None):

    if skel is not None:
        edges = []
        for i, j in enumerate(skel.joint_trees):
            if i == j:
                continue
            edges.append([j, i])

    n_nodes = np.max(edges) + 1
    adj = np.eye(n_nodes, dtype=np.float32)

    for edge in edges:
        adj[edge[0], edge[1]] = 1.0
        adj[edge[1], edge[0]] = 1.0

    return adj, edges

def find_seq(degrees, edges, root=0):
    seq_list = []

    def _find_seq(j, seq, degrees, edges):
        if degrees[j] > 2 and j != 0:
            seq_list.append(seq)
            seq = []
        if degrees[j] == 1:
            seq_list.append(seq)
            return
        for idx, edge in enumerate(edges):
             if edge[0] == j:
                _find_seq(edge[1], seq + [idx], degrees, edges)

    _find_seq(root, [], degrees, edges)
    return seq_list

def seq_to_pooling_nodes(seq_list, edges, last_pool=False):
    '''
    obtain groups of nodes to pool
    seq_list: list of edge indexs
    '''
    pooling_edge_list = []
    new_edges = []
    for seq in seq_list:
        if last_pool: # pool everything into as single node
            pooling_edge_list.append(seq)
            continue
        if len(seq) % 2 == 1:
            pooling_edge_list.append([seq[0]])
            new_edges.append(edges[seq[0]])
            seq = seq[1:]

        for i in range(0, len(seq), 2):
            pooling_edge_list.append([seq[i], seq[i + 1]])
            new_edges.append([edges[seq[i]][0], edges[seq[i + 1]][1]])

    pooling_node_list = []
    for edge_group in pooling_edge_list:
        nodes = []
        for edge_idx in edge_group:
            nodes += [n for n in edges[edge_idx]]
        nodes = np.unique(nodes)
        pooling_node_list.append(nodes.tolist())
    return pooling_node_list, new_edges

class SkeletonPooling(nn.Module):

    def __init__(self, edges, root_node=0):
        super().__init__()
        self.input_edges = edges
        self.input_degrees = [0 for i in range(np.max(edges) + 1)]
        #self.adj = np.eye(np.max(edges) + 1, dtype=np.float32)
        self.register_buffer('adj', torch.eye(np.max(edges)+1))


        for edge in edges:
            self.adj[edge[0], edge[1]] = 1.0
            self.adj[edge[1], edge[0]] = 1.0

            self.input_degrees[edge[0]] += 1
            self.input_degrees[edge[1]] += 1

        seq_list = find_seq(self.input_degrees, self.input_edges, root=root_node)
        pooling_list, new_edges = seq_to_pooling_nodes(seq_list, self.input_edges)

        self.pooling_list = pooling_list
        self.pooling_list.insert(0, [root_node])
        self.register_buffer('weights', torch.zeros(len(self.pooling_list), np.max(edges)+1))

        for i, nodes in enumerate(self.pooling_list):
            for n in nodes:
                self.weights[i, n] = 1. / len(nodes)

        # map from original node index to coarsen node idx
        mapping = {n[-1]: k for k, n in enumerate(self.pooling_list)}
        self.new_edges = [[mapping[a], mapping[b]] for (a, b) in new_edges]
        self.new_adj = skeleton_to_graph(edges=self.new_edges)[0]
        self.n_nodes = np.max(self.new_edges) + 1

    @property
    def output_graph(self):
        return self.new_adj, self.new_edges, self.n_nodes

    def forward(self, x):
        raise NotImplementedError('need to implement the forward approach for pooling!')

    def __repr__(self):
        return f'{self.__class__.__name__}: ({self.adj.shape[-1]}->{self.n_nodes})'

class SkeletonMeanPooling(SkeletonPooling):

    def forward(self, x):
        return torch.matmul(self.weights, x)

class SkeletonFolding(SkeletonPooling):
    '''
    folding several nodes into a single one based on pooling operation
    '''
    def __init__(self, *args, **kwargs):
        super(SkeletonFolding, self).__init__(*args, **kwargs)

        # make everything the same length so it's easier to do matrix ops
        max_len = max([len(p) for p in self.pooling_list])
        mask = torch.ones(len(self.pooling_list), max_len)
        for i, p in enumerate(self.pooling_list):
            while(len(p) < max_len):
                mask[i, len(p)] = 0
                p.append(p[0])
        self.pooling_idxs = torch.tensor(self.pooling_list)
        self.register_buffer('mask', mask)

    def forward(self, x):
        idxs = self.pooling_idxs.t()
        folded = torch.stack([x[:, idx] for idx in idxs], dim=2)
        return folded


class SkeletonMaxPooling(SkeletonFolding):

    def forward(self, x):
        folded = super().forward(x)
        return folded.max(-2)[0]

class SkeletonFoldingLinear(SkeletonFolding):
    '''
    collapse multiple nodes and apply linear layer
    '''

    def __init__(self, edges, in_feat, out_feat, **kwargs):
        super(SkeletonFoldingLinear, self).__init__(edges, **kwargs)

        self.in_edges = edges
        self._in_feat = in_feat
        self.in_feat = in_feat * self.mask.shape[-1]
        self.out_feat = out_feat
        self.lin = ParallelLinear(len(self.pooling_list), self.in_feat, self.out_feat)

    def forward(self, x):
        folded = super().forward(x)
        mask = self.mask.to(x.device).view(1, *self.mask.shape, 1)
        folded = (folded * mask).flatten(start_dim=-2)
        return self.lin(folded)

    def __repr__(self):
        return f'FoldingLinear({np.max(self.in_edges)+1}->{self.mask.shape[0]},' + \
                f'{self._in_feat}x{self.mask.shape[-1]},{self.out_feat})'

class DenseWGCN(nn.Module):
    '''
    def __init__(self, adj, *args, normalize_adj=False, bound_adj=False, adj_self_one=False,
                 skel_type=None, aggregate_dim=None, init_adj_w=0.05, **kwargs):
    '''
    def __init__(self, adj, in_channels, out_channels, normalize_adj=False, bound_adj=False, adj_self_one=False,
                 skel_type=None, aggregate_dim=None, init_adj_w=0.05, bias=True, no_adj=False, sep_bias=False, 
                 **kwargs):
        super(DenseWGCN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize_adj = normalize_adj
        self.bound_adj = bound_adj
        self.adj_self_one = adj_self_one
        self.aggregate_dim = aggregate_dim
        self.no_adj = no_adj
        adj = adj.clone()
        idx = torch.arange(adj.shape[-1])
        adj[:, idx, idx] = 1

        init_w = init_adj_w
        perturb = 0.1
        adj_w = (adj.clone() * (init_w + (torch.rand_like(adj) - 0.5 ) * perturb).clamp(min=0.01, max=1.0))
        adj_w[:, idx, idx] = 1.0

        if self.normalize_adj:
            adj_w = adj_w / adj_w.sum(-1, keepdim=True)

        self.lin = nn.Linear(in_channels, out_channels)
        if bias:
            if not sep_bias: # all nodes shared the same bias
                self.bias = nn.Parameter(torch.zeros(out_channels), requires_grad=True)
            else: # each node has their own bias term
                self.bias = nn.Parameter(torch.zeros(adj.shape[-1], out_channels), requires_grad=True)
        else:
            self.register_parameter('bias', None)

        self.register_buffer('adj', adj) # fixed, not learnable
        self.register_parameter('adj_w', nn.Parameter(adj_w, requires_grad=True))


    def get_adjw(self):
        adj, adj_w = self.adj, self.adj_w

        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
        adj_w = adj_w.unsqueeze(0) if adj_w.dim() == 2 else adj_w

        if self.adj_self_one:
            eye = torch.eye(adj_w.shape[-1])[None]
            offset = (eye - adj_w) * eye
            adj_w = adj_w + offset

        if self.bound_adj:
            adj_w = adj_w.clamp(min=0., max=1.)

        adj_w = adj_w * adj # masked out not connected part
        if self.normalize_adj:
            adj_w = adj_w / adj_w.sum(-1, keepdim=True)
        
        if self.no_adj:
            eye = torch.eye(adj_w.shape[-1])[None]
            adj_w = adj_w * eye

        return adj_w

    def forward(self, x):

        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj_w = self.get_adjw().to(x.device)

        out = self.lin(x)
        if not self.no_adj:
            if self.aggregate_dim is None:
                out = torch.matmul(adj_w, out)
            else:
                out, out_agg = out[..., :-self.aggregate_dim], out[..., -self.aggregate_dim:]
                out_agg = torch.matmul(adj_w, out_agg)
                out = torch.cat([out, out_agg], dim=-1)

        if self.bias is not None:
            out = out + self.bias

        return out

class DensePNGCN(DenseWGCN):

    def __init__(self, adj, in_channel, out_channel,
                 skel_type=None, *args, **kwargs):
        super(DensePNGCN, self).__init__(adj, in_channel, out_channel,
                                         *args, **kwargs)
        self.lin = ParallelLinear(adj.shape[-1], in_channel, out_channel, bias=False)


class DenseMIGCN(DensePNGCN):

    def __init__(self, *args, n_subjects=2, **kwargs):
        pass

class ConvWGCN(DenseWGCN):

    def __init__(self, adj, in_channel, out_channel,
                 kernel_size=3, stride=1, padding=0, 
                 skel_type=None, no_agg=False, *args, **kwargs):
        super(ConvWGCN, self).__init__(adj, in_channel, out_channel, 
                                       *args, **kwargs)
        groups = self.adj.shape[-1]
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.no_agg = no_agg
        self.lin = nn.ConvTranspose1d(in_channel * groups, out_channel * groups,
                                      kernel_size=kernel_size, stride=stride,
                                      padding=padding, groups=groups, bias=False)
    
    def forward(self, x):
        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj_w = self.get_adjw().to(x.device)

        # shape (N_B, N_node, N_feature * res)
        N_B, N_node, N_feat_res = x.shape
        x = x.reshape(N_B, N_node * self.in_channel, -1)

        out = self.lin(x).reshape(N_B, N_node, -1)

        if not self.no_agg:
            if self.aggregate_dim is None:
                out = torch.matmul(adj_w, out)
            else:
                out, out_agg = out[..., :-self.aggregate_dim], out[..., -self.aggregate_dim:]
                out_agg = torch.matmul(adj_w, out_agg)
                out = torch.cat([out, out_agg], dim=-1)
        if self.bias is not None:
            out = out.reshape(N_B, N_node, self.out_channel, -1)
            out = out + self.bias.reshape(1, 1, self.out_channel, 1)
            out = out.reshape(N_B, N_node, -1)

        return out

class SymmetryPNGCN(DenseWGCN):

    def __init__(self, adj, in_channel, out_channel,
                 *args, skel_type=None, **kwargs):
        super(SymmetryPNGCN, self).__init__(adj, in_channel, out_channel,
                                         *args, **kwargs)
        self.lin = SymmetryLinear(skel_type, in_channel, out_channel, bias=False)

class DensePNCATGCN(DenseWGCN):

    def __init__(self, adj, in_channel, out_channel,
                 *args, **kwargs):
        super(DensePNCATGCN, self).__init__(adj, in_channel, out_channel,
                                         *args, **kwargs)
        self.lin = ParallelLinear(adj.shape[-1], in_channel, out_channel // 2, bias=False)

    def forward(self, x):
        adj, adj_w = self.adj, self.adj_w

        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
        adj_w = adj_w.unsqueeze(0) if adj_w.dim() == 2 else adj_w
        B, N, _ = adj.size()

        adj_w = adj_w.to(x.device) * adj.to(x.device) - torch.eye(N)[None] # masked out not connected part
        out = self.lin(x)


        neighbor = torch.matmul(adj_w, out)
        out = torch.cat([out, neighbor], dim=-1)
        if self.bias is not None:
            out = out + self.bias

        return out


def get_gnn_backbone(per_node_input, backbone='PNGCN', skel_type=None,
                     gcn_D=4, gcn_sep_bias=False, node_W=64, skip_gcn=10, normalize_adj=False,
                     exclude_root=False, output_ch=None, bound_adj=False, rest_pose=None,
                     aggregate_dim=None, no_adj=False, init_adj_w=0.05):

    joint_trees = skel_type.joint_trees
    adj_matrix, edges = skeleton_to_graph(skel_type)

    shared_kwargs = {
        'adj_matrix': adj_matrix,
        'per_node_input': per_node_input,
        'W': node_W,
        'skip_gcn': skip_gcn,
        'skel_type': skel_type,
        'D': gcn_D,
        'output_ch': output_ch,
        'exclude_root': exclude_root,
    }

    gcn_module_kwargs = {
        'bound_adj': bound_adj,
        'normalize_adj': normalize_adj,
        'aggregate_dim': aggregate_dim,
        'init_adj_w': init_adj_w,
        'no_adj': no_adj,
        'sep_bias': gcn_sep_bias,
    }

    if backbone == 'PNGNN':
        return BasicGNN(gcn_module=DensePNGCN, gcn_module_kwargs=gcn_module_kwargs, 
                        **shared_kwargs)
    elif backbone == 'MIXGNN':
        return MixGNN(gcn_module=DensePNGCN, gcn_module_kwargs=gcn_module_kwargs, 
                      **shared_kwargs)
    elif backbone == 'SHARE':
        return SharedValidMLP(**shared_kwargs)
    elif backbone == 'VAN':
        return VoxelAggGNN(gcn_module=DensePNGCN, gcn_module_kwargs=gcn_module_kwargs, 
                            **shared_kwargs)
    elif backbone == 'PNBGNN':
        return BodyGNN(gcn_module=DensePNGCN, last_module=DensePNGCN, 
                       gcn_module_kwargs=gcn_module_kwargs, **shared_kwargs)

def get_volume_gnn_backbone(per_node_input, backbone='PNBGCN', skel_type=None,
                            gcn_D=4, gcn_fc_D=0, gcn_sep_bias=False, 
                            node_W=64, skip_gcn=10,
                            voxel_res=4, voxel_feat=4, normalize_adj=False,
                            exclude_root=False, mask_root=False, 
                            bound_adj=False, rest_pose=None,
                            adj_self_one=False, pred_residual=False, 
                            opt_scale=False, base_scale=0.5,
                            n_subjects=2, subjectcode_ch=128,
                            n_basis=32,
                            fix_interval=False,
                            attenuate_feat=False, attenuate_invalid=False,
                            no_adj=False,
                            skel_profile=None,
                            align_corners=False,
                            aggregate_dim=None, init_adj_w=0.05):

    joint_trees = skel_type.joint_trees
    adj_matrix, edges = skeleton_to_graph(skel_type)

    shared_kwargs = {
        'adj_matrix': adj_matrix,
        'per_node_input': per_node_input,
        'W': node_W,
        'skip_gcn': skip_gcn,
        'skel_type': skel_type,
        'D': gcn_D,
        'fc_D': gcn_fc_D,
        'voxel_res': voxel_res,
        'voxel_feat': voxel_feat,
        'exclude_root': exclude_root,
        'mask_root': mask_root,
        'align_corners': align_corners,
    }

    gcn_module_kwargs = {
        'bound_adj': bound_adj,
        'normalize_adj': normalize_adj,
        'adj_self_one': adj_self_one,
        'aggregate_dim': aggregate_dim,
        'init_adj_w': init_adj_w,
        'no_adj': no_adj,
        'sep_bias': gcn_sep_bias,
    }

    factorize_kwargs = {
        'opt_scale': opt_scale,
        'base_scale': base_scale,
        'skel_profile': skel_profile,
        'pred_residual': pred_residual,
    }


    if backbone == 'PNBGNN':
        return BodyGNN(gcn_module=DensePNGCN, last_module=DensePNGCN,
                       gcn_module_kwargs=gcn_module_kwargs, **shared_kwargs)
    elif backbone == 'PNBGNN_S':
        return BodyGNN(gcn_module=DensePNGCN, last_module=DenseWGCN, 
                       gcn_module_kwargs=gcn_module_kwargs, **shared_kwargs)
    elif backbone == 'PNBGNN_P':
        return BodyGNN(gcn_module=DensePNGCN, last_module=ParallelLinear, 
                       gcn_module_kwargs=gcn_module_kwargs, **shared_kwargs)
    elif backbone == 'VOLGNN':
        return VolumeGNN(gcn_module=DensePNGCN, last_module=ParallelLinear, 
                         gcn_module_kwargs=gcn_module_kwargs, **shared_kwargs)
    elif backbone.startswith(('FGNN')):
        factorize_type = backbone.split('FGNN')[-1]
        if len(factorize_type) == 0:
            factorize_type = 'sum'
        return FactorizeGNN(gcn_module=DensePNGCN, factorize_type=factorize_type,
                            last_module=ParallelLinear,
                            gcn_module_kwargs=gcn_module_kwargs, 
                            attenuate_invalid=attenuate_invalid,
                            attenuate_feat=attenuate_feat,
                            **factorize_kwargs, **shared_kwargs)
    elif backbone.startswith(('FBGNN')):
        factorize_type = backbone.split('FBGNN')[-1]
        if len(factorize_type) == 0:
            factorize_type = 'sum'
        return FactorizeBasisGNN(gcn_module=DensePNGCN, factorize_type=factorize_type,
                            last_module=ParallelLinear,
                            gcn_module_kwargs=gcn_module_kwargs, 
                            attenuate_invalid=attenuate_invalid,
                            attenuate_feat=attenuate_feat,
                            n_basis=n_basis,
                            **factorize_kwargs, **shared_kwargs)
    elif backbone.startswith(('FCGNN')):
        factorize_type = backbone.split('FCGNN')[-1]
        if len(factorize_type) == 0:
            factorize_type = 'sum'
        return FactorizeCubeGNN(gcn_module=DensePNGCN, factorize_type=factorize_type,
                            last_module=ParallelLinear,
                            gcn_module_kwargs=gcn_module_kwargs, 
                            attenuate_invalid=attenuate_invalid,
                            attenuate_feat=attenuate_feat,
                            **factorize_kwargs, **shared_kwargs)
    
    elif backbone == 'PoseCatGNN':
        return PoseCatGNN(gcn_module=DensePNGCN, last_module=ParallelLinear, 
                          gcn_module_kwargs=gcn_module_kwargs, 
                          attenuate_invalid=attenuate_invalid,
                          attenuate_feat=attenuate_feat,
                          **factorize_kwargs, **shared_kwargs)

    elif backbone.startswith('MVPCGNN'):
        return MVPCGNN(gcn_module=DensePNGCN,
                       last_module=ParallelLinear,
                       gcn_module_kwargs=gcn_module_kwargs, 
                       attenuate_invalid=attenuate_invalid,
                       attenuate_feat=attenuate_feat,
                       opt_scale=opt_scale,
                       base_scale=base_scale,
                       skel_profile=skel_profile,
                       **shared_kwargs)
    else:
        raise NotImplementedError

class BasicGNN(nn.Module):

    def __init__(self, adj_matrix, per_node_input, W=64, D=4,
                 skip_gcn=10, gcn_module=DenseWGCN, gcn_module_kwargs={}, 
                 nl=F.relu, exclude_root=False, mask_root=False,
                 output_ch=None, skel_type=None):
        super(BasicGNN, self).__init__()

        self.adj_matrix = adj_matrix
        self.skel_type = skel_type
        self.exclude_root = exclude_root
        self.mask_root = mask_root

        self.per_node_input = per_node_input
        self.W = W
        self.D = D

        self.skip_gcn = skip_gcn
        self.gcn_module_kwargs = gcn_module_kwargs

        self.nl = nl
        if output_ch is None:
            self.output_ch = self.W + 1
        else:
            self.output_ch = output_ch
        self.init_network(gcn_module)

    def init_network(self, gcn_module):
        adj_matrix, skel_type = self.adj_matrix, self.skel_type
        per_node_input = self.per_node_input
        W, D = self.W, self.D

        n_nodes = adj_matrix.shape[-1]
        adj_matrix = torch.tensor(adj_matrix).view(1, n_nodes, n_nodes)

        layers = [gcn_module(adj_matrix, per_node_input, W, improved=True,
                             skel_type=skel_type, **self.gcn_module_kwargs)]
        input_ch = W 
        for i in range(D-2):
            output_ch = W if i != (self.skip_gcn -2) else W - per_node_input
            layers += [gcn_module(adj_matrix, input_ch, output_ch, improved=True, skel_type=skel_type, 
                                  **self.gcn_module_kwargs)]
        layers += [gcn_module(adj_matrix, input_ch, self.output_ch, improved=True, skel_type=skel_type,
                              **self.gcn_module_kwargs)]
        self.layers = nn.ModuleList(layers)

        if self.exclude_root or self.mask_root:
            self.mask = torch.ones(1, len(self.skel_type.joint_names), 1)
            self.mask[:, self.skel_type.root_id, :] = 0.

    def forward(self, inputs, **kwargs):

        skip_gcn = self.skip_gcn
        n = inputs
        if self.exclude_root or self.mask_root:
            n = n * self.mask.to(n.device)

        for i, l in enumerate(self.layers):
            n = l(n)
            if i == 0:
                first_n = n
            if (i + 1) == len(self.layers):
                n, confd = n[..., :-1], n[..., -1:]

            if i == skip_gcn-1:
                n = torch.cat([n, inputs], dim=-1)

            if (i + 1) < len(self.layers) and self.nl is not None:
                n = self.nl(n, inplace=True)

        if self.exclude_root:
            # TODO: currently assume root_id=0
            n, confd = n[:, 1:], confd[:, 1:]

        return n, confd

    def get_adjw(self):
        adjw_list = []

        for m in self.modules():
            if hasattr(m, 'adj_w'):
                adjw_list.append(m.get_adjw())

        return adjw_list

class MixGNN(BasicGNN):

    def __init__(self, *args, N_P=2, **kwargs):
        self.N_P = N_P # numbers of parallel linear after GNN
        super(MixGNN, self).__init__(*args, **kwargs)

    def init_network(self, gcn_module):
        adj_matrix, skel_type = self.adj_matrix, self.skel_type
        per_node_input = self.per_node_input
        W, D, N_P = self.W, self.D, self.N_P

        n_nodes = adj_matrix.shape[-1]
        adj_matrix = torch.tensor(adj_matrix).view(1, n_nodes, n_nodes)

        layers = [gcn_module(adj_matrix, per_node_input, W, improved=True, skel_type=skel_type,
                             **self.gcn_module_kwargs)]
        input_ch = output_ch = W
        for i in range(D-N_P-1):
            layers += [gcn_module(adj_matrix, input_ch, output_ch, improved=True, skel_type=skel_type,
                                  **self.gcn_module_kwargs)]
        for i in range(N_P-1):
            layers += [ParallelLinear(n_nodes, input_ch, output_ch)]
        layers += [ParallelLinear(n_nodes, input_ch, self.output_ch)]
        self.layers = nn.ModuleList(layers)

        if self.exclude_root:
            self.mask = torch.ones(1, len(self.skel_type.joint_names), 1)
            self.mask[:, self.skel_type.root_id, :] = 0.

class BodyGNN(BasicGNN):

    def __init__(self, *args, voxel_res=4, voxel_feat=4, fc_D=0, 
                 align_corners=False, last_module=DensePNGCN, **kwargs):
        self.last_module = last_module
        self.voxel_res = voxel_res
        self.voxel_feat = voxel_feat
        self.fc_D = fc_D
        self.align_corners = align_corners

        super(BodyGNN, self).__init__(*args, **kwargs)

    @property
    def output_size(self):
        return self.voxel_res**3 * self.voxel_feat

    @property
    def sample_feat_size(self):
        return self.voxel_feat

    def init_network(self, gcn_module):
        adj_matrix, skel_type = self.adj_matrix, self.skel_type
        per_node_input = self.per_node_input
        W, D, fc_D = self.W, self.D, self.fc_D

        n_nodes = adj_matrix.shape[-1]
        adj_matrix = torch.tensor(adj_matrix).view(1, n_nodes, n_nodes)

        layers = [gcn_module(adj_matrix, per_node_input, W, improved=True, skel_type=skel_type,
                             **self.gcn_module_kwargs)]
        for i in range(D-fc_D-2):
            layers += [gcn_module(adj_matrix, W, W, improved=True, skel_type=skel_type,
                                  **self.gcn_module_kwargs)]
        for i in range(fc_D):
            layers += [ParallelLinear(n_nodes, W, W)]

        offset = 0
        if self.exclude_root or self.mask_root:
            self.mask = torch.ones(1, len(self.skel_type.joint_names), 1)
            self.mask[:, self.skel_type.root_id, :] = 0.
            offset = 1 if self.exclude_root else offset

        if self.last_module == ParallelLinear:
            offset = 1 if self.exclude_root else 0
            layers += [ParallelLinear(n_nodes - offset, W, self.output_size)]
        else:
            layers += [self.last_module(adj_matrix, W, self.output_size, improved=True, skel_type=skel_type, 
                                        **self.gcn_module_kwargs)]
        self.layers = nn.ModuleList(layers)
        self.volume_shape = [len(self.skel_type.joint_trees) - offset, self.voxel_feat] + \
                                3 * [self.voxel_res]

    def forward(self, n, **kwargs):

        skip_gcn = self.skip_gcn

        if self.exclude_root or self.mask_root:
            n = self.mask.to(n.device) * n

        for i, l in enumerate(self.layers):
            if ((i + 1) == len(self.layers)) and self.exclude_root:
                # TODO: fix this, currently assume root_id = 0
                n = n[:, 1:]
            n = l(n)
            if i == 0:
                first_n = n

            if i == skip_gcn:
                n = n + first_n

            if (i + 1) < len(self.layers) and self.nl is not None:
                n = self.nl(n, inplace=True)

        return n

    def sample_from_volume(self, graph_feat, x):
        '''
        graph_feat: predicted factorized volume (N_graph, N_joints, voxel_res^3*voxel_feat)
        x: points in local joint coordinates for sampling
        '''
        align_corners = self.align_corners
        N_rays, N_samples = x.shape[:2]
        N_graphs = graph_feat.shape[0]
        N_expand = N_rays // N_graphs

        offset = 1 if self.exclude_root else 0
        N_joints = len(self.skel_type.joint_trees) - offset

        # turns graph_feat into (N, F, H, W, D) format for more efficient grid_sample
        # -> (N_graphs * N_joints, F, H, W, D)
        graph_feat = graph_feat.reshape(-1, *self.volume_shape[1:])
        # reshape and permute x similarly
        x = x.reshape(N_graphs, N_expand, N_samples, N_joints, 3)
        # turns x into (N, H, W, D, 3) format for efficient_grid_sample
        # -> (N_graphs * N_joints, N_expands, N_samples, 1, 3)
        x = x.permute(0, 3, 1, 2, 4).reshape(N_graphs * N_joints, N_expand, N_samples, 1, 3)
        # (N_rays * N_samples * N_joints, 1, 1, 1, 3)
        # mode='bilinear' is actually trilinear for 5D input
        part_feat = F.grid_sample(graph_feat, x, mode='bilinear',
                                  padding_mode='zeros', align_corners=align_corners)
        # turn it back to (N_rays, N_samples, N_joints, voxel_feat)
        part_feat = part_feat.reshape(N_graphs, N_joints, self.voxel_feat, N_expand, N_samples)
        part_feat = part_feat.permute(0, 3, 4, 1, 2).reshape(N_rays, N_samples, N_joints, -1)

        return part_feat.reshape(N_rays, N_samples, N_joints, -1), torch.ones(N_rays, N_samples, N_joints)
    
class FactorizeGNN(BodyGNN):

    def __init__(self, *args, factorize_type='sum', pred_residual=False, 
                 opt_scale=False, base_scale=0.5, attenuate_feat=False,
                 attenuate_invalid=False, skel_profile=None, **kwargs):

        self.factorize_type = factorize_type
        self.pred_residual = pred_residual
        self.attenuate_feat = attenuate_feat
        self.attenuate_invalid = attenuate_invalid
        self.opt_scale  = opt_scale
        self.skel_profile = skel_profile
        super(FactorizeGNN, self).__init__(*args, **kwargs)

        self.construct = None

        self._sample_feat_size = self.voxel_feat
        if self.factorize_type == 'sum':
            self.construct = lambda feat: feat.sum(-1) / 3.
        elif self.factorize_type == 'sumn':
            self.construct = lambda feat: F.normalize(feat.sum(-1), dim=-1, p=2)
        elif self.factorize_type == 'mul':
            self.construct = lambda feat: feat.prod(dim=-1)
        elif self.factorize_type == 'muln':
            self.construct = lambda feat: F.normalize(feat.prod(dim=-1), dim=-1, p=2)
        elif self.factorize_type == 'cat':
            self.construct = lambda feat: feat.flatten(start_dim=-2)
            self._sample_feat_size = self.voxel_feat * 3
        
        self.base_scale = base_scale
        self.init_scale()

    @property
    def output_size(self):
        return self.voxel_res * self.voxel_feat * 3

    @property
    def sample_feat_size(self):
        return self._sample_feat_size
    
    def init_scale(self):
        offset = 1 if self.exclude_root else 0
        N_joints = len(self.skel_type.joint_names) - offset

        scale = torch.ones(N_joints, 3) * self.base_scale
        if self.skel_profile is not None:
            scale = init_volume_scale(self.base_scale, self.skel_profile, self.skel_type)
        self.init_scale = scale.clone()
        self.register_parameter('axis_scale', nn.Parameter(scale, requires_grad=self.opt_scale))
    
    def sample_from_volume(self, graph_feat, x, *args, **kwargs):
        '''
        graph_feat: predicted factorized volume (N_graph, N_joints, voxel_res*voxel_feat*3)
        x: points in local joint coordinates for sampling
        '''
        N_rays, N_samples = x.shape[:2]
        N_graphs = graph_feat.shape[0]
        N_expand = N_rays // N_graphs

        offset = 1 if self.exclude_root else 0
        N_joints = len(self.skel_type.joint_trees) - offset

        # rescale x (if opt_scale=True)

        # reshape scale to (1, 1, N_joints, 3)
        x = x / self.axis_scale.reshape(1, 1, -1, 3).abs()
        alpha, beta = 2, 6
        coord_window =  torch.exp(-alpha * ((x**beta).sum(-1))).detach()
        if self.attenuate_invalid:
            invalid = 1 - coord_window
        else:
            invalid = ((x.abs() > 1).sum(-1) > 0).float()

        # (N_graphs * N_joints, C, H, W)
        graph_feat = graph_feat.reshape(N_graphs * N_joints, self.voxel_feat, self.voxel_res, 3)
        #graph_feat = F.normalize(graph_feat, dim=-3)
        # (N_graphs, N_expand * N_samples, N_joints, 3) -> (N_graphs * N_joints, N_expand * N_samples, 3)
        # permute to make it looks like (N, H, W) for faster, cheaper grid_sampling
        x = x.reshape(N_graphs, -1, N_joints, 3).permute(0, 2, 1, 3).flatten(end_dim=1)
        graph_feat = factorize_grid_sample(graph_feat, x, training=self.training, 
                                           need_hessian=kwargs['need_hessian'])

        # TODO: windowing, make it a flag
        graph_feat = graph_feat.reshape(N_graphs, N_joints, self.voxel_feat, N_expand, N_samples, 3)

        # turn it back to (N_rays, N_samples, N_joints, voxel_feat, 3)
        graph_feat = graph_feat.permute(0, 3, 4, 1, 2, 5).flatten(end_dim=1)
        graph_feat = self.construct(graph_feat)
        if self.attenuate_feat:
            graph_feat = graph_feat * coord_window[..., None]

        return graph_feat, invalid
    
    def get_axis_scale(self):
        return self.axis_scale

class PoseCatGNN(FactorizeGNN):

    @property
    def output_size(self):
        return self.voxel_feat

    def sample_from_volume(self, graph_feat, x, *args, **kwargs):
        '''
        Simply concatenate them together
        '''
        N_rays, N_samples = x.shape[:2]
        N_graphs = graph_feat.shape[0]
        N_expand = N_rays // N_graphs

        offset = 1 if self.exclude_root else 0
        N_joints = len(self.skel_type.joint_trees) - offset

        x = x / self.axis_scale.reshape(1, 1, -1, 3).abs()
        alpha, beta = 2, 6
        coord_window =  torch.exp(-alpha * ((x**beta).sum(-1))).detach()
        if self.attenuate_invalid:
            invalid = 1 - coord_window
        else:
            invalid = ((x.abs() > 1).sum(-1) > 0).float()

        graph_feat = graph_feat.reshape(-1, 1, 1, N_joints, self.voxel_feat)
        graph_feat = graph_feat.expand(N_graphs, N_expand, N_samples,
                                       N_joints, self.voxel_feat).flatten(end_dim=1)
        if self.attenuate_feat:
            graph_feat = graph_feat * coord_window[..., None]
            x = x * invalid[..., None]
        graph_feat = torch.cat([graph_feat, x], dim=-1)
        
        return graph_feat, invalid

class FactorizeCubeGNN(FactorizeGNN):

    def __init__(self, *args, cube_size=3, cube_width=1.0, **kwargs):
        self.cube_size = cube_size
        super(FactorizeCubeGNN, self).__init__(*args, **kwargs)

        # find the interval (length of one voxel)
        #grid_intv = (cube_width * 1.0) / self.voxel_res 
        grid_intv = (cube_width * 2.0) / self.voxel_res 
        cube = torch.linspace(-grid_intv, grid_intv, cube_size)
        cube_d = cube.reshape(cube_size, 1, 1, 1).expand(-1, cube_size, cube_size, -1)
        cube_h = cube.reshape(1 ,cube_size, 1, 1).expand(cube_size, -1, cube_size, -1)
        cube_w = cube.reshape(1, 1, cube_size, 1).expand(cube_size, cube_size, -1, -1)
        cube = torch.cat([cube_d, cube_h, cube_w], dim=-1)
        assert cube_size == 3

        # actually we only need 7 locations on the cube (considering cube size == 3)
        # namely top, bot, left, right, front, back, and center (original sample point)
        top = cube[1, 0, 1]
        bot = cube[1, 2, 1]
        left = cube[1, 1, 0]
        center = cube[1, 1, 1]
        right = cube[1, 1, 2]
        front = cube[2, 1, 1]
        back = cube[0, 1, 1]
        self.center_idx = 3

        #self.register_buffer('cube', torch.cat([top, bot, left, center, right, front, back], dim=-1))
        #self.register_buffer('cube', torch.stack([top, bot, left, center, right, front, back], dim=0))
        self.register_buffer('cube', torch.stack([top, bot, left, right, front, back, center], dim=0))
        w = self.voxel_feat * 3 * len(self.cube) 
        
        """
        self.mapper = nn.Sequential(
            nn.Linear(w, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.voxel_feat * 3),
        )
        """
        # TODO: we really want to use GNN with shared MLP here?
        #       because the local pattern should be rotation invariant?
        #       --> extremely slow
        adj = torch.eye(7)[None]
        # center connects to everyone
        adj[:, 3, :] = 1.
        # everyone connects to center
        adj[:, :, 3] = 1.
        self.mapper = nn.Sequential(
            nn.Linear(self.voxel_feat * 3 * 2, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, self.voxel_feat * 3),
        )
        """
        self.mapper = nn.Sequential(
            ParallelLinear(24, w, 32),
            nn.ReLU(inplace=True),
            ParallelLinear(24, 32, 32),
            nn.ReLU(inplace=True),
            ParallelLinear(24, 32, self.voxel_feat * 3),
        )
        """

    def sample_from_volume(self, graph_feat, x, *args, **kwargs):
        '''
        graph_feat: predicted factorized volume (N_graph, N_joints, voxel_res*voxel_feat*3)
        x: points in local joint coordinates for sampling
        '''
        N_rays, N_samples = x.shape[:2]
        N_graphs = graph_feat.shape[0]
        N_expand = N_rays // N_graphs

        offset = 1 if self.exclude_root else 0
        N_joints = len(self.skel_type.joint_trees) - offset

        # rescale x (if opt_scale=True)

        # reshape scale to (1, 1, N_joints, 3)
        x = x / self.axis_scale.reshape(1, 1, -1, 3).abs()
        invalid = ((x.abs() > 1).sum(-1) > 0).float()

        # (N_graphs * N_joints, C, H, W)
        graph_feat = graph_feat.reshape(N_graphs * N_joints, self.voxel_feat, self.voxel_res, 3)
        # (N_graphs, N_expand * N_samples, N_joints, 3) -> (N_graphs * N_joints, N_expand * N_samples, 3)
        # permute to make it looks like (N, H,o W) for faster, cheaper grid_sampling
        # creat a cube of cooridnates
        csize = self.cube_size
        # shape (N_rays, N_samples, N_joints, D, H, W, 3)
        '''
        # the actual 3x3x3 cube case, which we don't need as we are using factorized volume
        x_cube = x.reshape(N_rays, N_samples, N_joints, 1, 1, 1, 3) + self.cube.reshape(1, 1, 1, csize, csize, csize, 3)
        # permute to (N_graphs, N_joints, N_samples, cube_size**3, 3)
        x_cube = x_cube.reshape(N_graphs, -1, N_joints, csize**3, 3).permute(0, 2, 1, 3, 4)
        # collapse dimension to (N_graphs * N_joints, all points, 3)
        x_cube = x_cube.reshape(N_graphs * N_joints, -1, 3)
        x = x.reshape(N_graphs, -1, N_joints, 3).permute(0, 2, 1, 3).flatten(end_dim=1)

        graph_feat = factorize_grid_sample(graph_feat, x_cube, training=self.training, 
                                           need_hessian=kwargs['need_hessian'])

        '''
        # permute to (N_graphs, N_joints, N_samples, -1, 3)
        x_cube = x.reshape(N_rays, N_samples, N_joints, 1, 3) + self.cube.reshape(1, 1, 1, -1, 3)
        x_cube = x_cube.reshape(N_graphs, -1, N_joints, csize * 2 + 1, 3).permute(0, 2, 1, 3, 4)
        # collapse dimension to (N_graphs * N_joints, all points, 3)
        x_cube = x_cube.reshape(N_graphs * N_joints, -1, 3)

        graph_feat = factorize_grid_sample(graph_feat, x_cube, training=self.training, 
                                           need_hessian=kwargs['need_hessian'])
        
        

        # TODO: windowing, make it a flag
        graph_feat = graph_feat.reshape(N_graphs, N_joints, self.voxel_feat, N_expand, N_samples, -1, 3)
        graph_feat = F.normalize(graph_feat, dim=2)
        # turn it to (N_rays, N_samples, N_joints, cube_sample, voxel_feat, 3)
        graph_feat = graph_feat.permute(0, 3, 4, 1, 5, 2, 6)
        graph_feat = self.construct(graph_feat).reshape(N_rays, N_samples, N_joints, -1, self.voxel_feat * 3)
        center_feat = graph_feat[..., self.center_idx, :]

        graph_feat = graph_feat.flatten(end_dim=2)
        neighbor_feat = graph_feat[..., :-1, :]
        center_feat = graph_feat[..., -1:, :]
        cat_feat = torch.cat([center_feat.expand(-1, 6, -1), neighbor_feat], dim=-1)
        graph_feat = self.mapper(cat_feat).mean(-2) + center_feat[..., 0, :]
        graph_feat = graph_feat.reshape(N_rays, N_samples, N_joints, -1)

        """
        graph_feat = graph_feat.flatten(start_dim=-2).flatten(end_dim=-3)

        graph_feat = self.mapper(graph_feat).reshape(N_rays, N_samples, N_joints, -1)
        graph_feat = graph_feat + center_feat
        """
        """
        graph_feat = graph_feat.flatten(start_dim=-2).flatten(end_dim=-2)
        valid_idxs = torch.where(invalid.reshape(-1) < 1)[0]
        mapped_feat = torch.zeros_like(center_feat.reshape(-1, self.voxel_feat * 3))
        mapped_feat[valid_idxs] = self.mapper(graph_feat[valid_idxs])
        mapped_feat = mapped_feat.reshape(N_rays, N_samples, N_joints, self.voxel_feat * 3)
        graph_feat = mapped_feat + center_feat
        """

        x = x.reshape(N_rays, N_samples, N_joints, 3)
        if self.attenuate_feat:
            alpha, beta = 2, 6
            coord_window =  torch.exp(-alpha * ((x**beta).sum(-1))).detach()
            graph_feat = graph_feat * coord_window[..., None]

        return graph_feat, invalid
 

class FactorizeBasisGNN(FactorizeGNN):

    def __init__(self, *args, n_basis=32, **kwargs):
        self.n_basis = n_basis
        super(FactorizeBasisGNN, self).__init__(*args, **kwargs)
    
    @property
    def output_size(self):
        return self.n_basis
    
    def init_network(self, *args, **kwargs):
        super(FactorizeBasisGNN, self).init_network(*args, **kwargs)
        #return self.voxel_res * self.voxel_feat * 3
        self.register_parameter('basis',
            nn.Parameter(torch.randn(self.n_basis, self.voxel_res * self.voxel_feat * 3) * 0.5,
                         requires_grad=True
            )
        )
    
    def forward(self, *args, **kwargs):
        n = super(FactorizeBasisGNN, self).forward(*args, **kwargs)
        #graph_feat = graph_feat.reshape(N_graphs * N_joints, self.voxel_feat, self.voxel_res, 3)
        basis = self.basis.reshape(1, 1, self.n_basis, -1)
        n = torch.tanh(n)
        n = (basis * n[..., None]).sum(-2)
        N_graphs, N_joints = n.shape[:2]
        n = n.reshape(N_graphs, N_joints, self.voxel_feat, self.voxel_res, 3)
        n = F.normalize(n, dim=-3).reshape(N_graphs, N_joints, -1)
        return n

class MVPCGNN(BodyGNN):

    def __init__(self, *args, attenuate_feat=False, attenuate_invalid=False, 
                 opt_scale=False, skel_profile=None, base_scale=0.5, **kwargs):
        self.attenuate_feat = attenuate_feat
        self.attenuate_invalid = attenuate_invalid
        self.opt_scale  = opt_scale
        self.skel_profile = skel_profile
        super(MVPCGNN, self).__init__(*args, **kwargs)

        self.base_scale = base_scale
        self.init_scale()

    def init_network(self, gcn_module):

        adj_matrix, skel_type = self.adj_matrix, self.skel_type
        per_node_input = self.per_node_input
        W, D, fc_D = self.W, self.D, self.fc_D

        self.n_nodes = n_nodes = adj_matrix.shape[-1]
        adj_matrix = torch.tensor(adj_matrix).view(1, n_nodes, n_nodes)

        layers = [gcn_module(adj_matrix, per_node_input, W, improved=True, skel_type=skel_type,
                             **self.gcn_module_kwargs)]
        for i in range(D-fc_D-2):
            layers += [gcn_module(adj_matrix, W, W, improved=True, skel_type=skel_type,
                                  **self.gcn_module_kwargs)]
        for i in range(fc_D):
            layers += [ParallelLinear(n_nodes, W, W)]

        self.layers = nn.ModuleList(layers)
        # see if root is excluded
        offset = 0
        if self.exclude_root or self.mask_root:
            self.mask = torch.ones(1, len(self.skel_type.joint_names), 1)
            self.mask[:, self.skel_type.root_id, :] = 0.
            offset = 1 if self.exclude_root else offset

        voxel_feat = self.voxel_feat
        voxel_rest = self.voxel_res 
        n_channels = self.n_nodes * self.W // 4
        out_feature = (self.voxel_feat - 1) * self.n_nodes * self.voxel_res
        out_density = 1 * self.n_nodes * self.voxel_res
        n_groups = n_nodes 
        kernel_size = 4

        assert self.voxel_res == 16

        conv_density = [
            ConvTranspose2dELR(n_channels, 
                               n_channels,
                               kernel_size=4, stride=2, 
                               padding=1, 
                               groups=n_groups, 
                               norm=True,
                               ub=(4,4),
                               act=nn.LeakyReLU(0.2)
                               ), # -> 4x4
            ConvTranspose2dELR(n_channels, 
                               n_channels,
                               kernel_size=4, stride=2, 
                               padding=1, 
                               groups=n_groups, 
                               norm=True,
                               ub=(8,8),
                               act=nn.LeakyReLU(0.2)
                               ), # -> 8x8
            ConvTranspose2dELR(n_channels, 
                               out_density,
                               kernel_size=4, stride=2, 
                               padding=1, 
                               groups=n_groups, 
                               norm=False,
                               ub=(16,16)), # -> 8x8
        ]
        conv_feat = [
            ConvTranspose2dELR(n_channels, 
                               n_channels,
                               kernel_size=4, stride=2, 
                               padding=1, 
                               groups=n_groups, 
                               norm=True,
                               ub=(4,4),
                               act=nn.LeakyReLU(0.2)
                               ), # -> 4x4
            ConvTranspose2dELR(n_channels, 
                               n_channels,
                               kernel_size=4, stride=2, 
                               padding=1, 
                               groups=n_groups, 
                               norm=True,
                               ub=(8,8),
                               act=nn.LeakyReLU(0.2)
                               ), # -> 8x8
            ConvTranspose2dELR(n_channels, 
                               out_feature,
                               kernel_size=4, stride=2, 
                               padding=1, 
                               groups=n_groups, 
                               norm=False,
                               ub=(16,16)), # -> 8x8
        ]
        self.conv_feat = nn.Sequential(*conv_feat)
        self.conv_density = nn.Sequential(*conv_density)
        self.volume_shape = (self.n_nodes, self.voxel_feat, *(3*(self.voxel_res,)))

    def init_scale(self):
        offset = 1 if self.exclude_root else 0
        N_joints = len(self.skel_type.joint_names) - offset

        scale = torch.ones(N_joints, 3) * self.base_scale
        if self.skel_profile is not None:
            scale = init_volume_scale(self.base_scale, self.skel_profile, self.skel_type)
        self.init_scale = scale.clone()
        self.register_parameter('axis_scale', nn.Parameter(scale, requires_grad=self.opt_scale))

    def forward(self, n, **kwargs):

        part_feat = super(MVPCGNN, self).forward(n, **kwargs)
        N_B = part_feat.shape[0]
        part_feat = part_feat.reshape(N_B, -1, 2, 2)


        feat = self.conv_feat(part_feat)
        feat = feat.reshape(N_B, self.n_nodes, self.voxel_feat-1, 16, 16, 16)

        density = self.conv_density(part_feat)
        density = density.reshape(N_B, self.n_nodes, 1, 16, 16, 16)
        out_feat = torch.cat([density, feat], dim=2)

        # important: the predicted dimension is W (x)
        out_feat = out_feat.permute(0, 1, 2, 4, 5, 3)
        out_feat = out_feat.reshape(N_B, self.n_nodes, -1)

        return out_feat

    def sample_from_volume(self, graph_feat, x):
        '''
        graph_feat: predicted factorized volume (N_graph, N_joints, voxel_res^3*voxel_feat)
        x: points in local joint coordinates for sampling
        '''
        align_corners = self.align_corners
        N_rays, N_samples = x.shape[:2]
        N_graphs = graph_feat.shape[0]
        N_expand = N_rays // N_graphs

        offset = 1 if self.exclude_root else 0
        N_joints = len(self.skel_type.joint_trees) - offset

        # turns graph_feat into (N, F, H, W, D) format for more efficient grid_sample
        # -> (N_graphs * N_joints, F, H, W, D)
        graph_feat = graph_feat.reshape(-1, *self.volume_shape[1:])
        # reshape and permute x similarly

        x = x / self.axis_scale.reshape(1, 1, -1, 3).abs()

        # create invalid
        alpha, beta = 2, 6
        coord_window = torch.exp(-alpha * ((x**beta).sum(-1))).detach() 
        if self.attenuate_invalid:
            invalid = 1 - coord_window
        else:
            invalid = ((x.abs() > 1).sum(-1) > 0).float()

        x = x.reshape(N_graphs, N_expand, N_samples, N_joints, 3)
        # turns x into (N, H, W, D, 3) format for efficient_grid_sample
        # -> (N_graphs * N_joints, N_expands, N_samples, 1, 3)
        x = x.permute(0, 3, 1, 2, 4).reshape(N_graphs * N_joints, N_expand, N_samples, 1, 3)
        # (N_rays * N_samples * N_joints, 1, 1, 1, 3)
        # mode='bilinear' is actually trilinear for 5D input
        part_feat = F.grid_sample(graph_feat, x, mode='bilinear',
                                  padding_mode='zeros', align_corners=align_corners)
        # turn it back to (N_rays, N_samples, N_joints, voxel_feat)
        part_feat = part_feat.reshape(N_graphs, N_joints, self.voxel_feat, N_expand, N_samples)
        part_feat = part_feat.permute(0, 3, 4, 1, 2).reshape(N_rays, N_samples, N_joints, -1)
        logit = part_feat[..., :1]
        if self.attenuate_feat:
            part_feat = part_feat * coord_window[..., None]

        return part_feat, invalid

class VoxelAggGNN(MixGNN):

    def init_network(self, gcn_module):
        adj_matrix, skel_type = self.adj_matrix, self.skel_type
        per_node_input = self.per_node_input
        W, D, N_P = self.W, self.D, self.N_P

        n_nodes = adj_matrix.shape[-1]
        adj_matrix = torch.tensor(adj_matrix).view(1, n_nodes, n_nodes)

        gnn_layers = [gcn_module(adj_matrix, per_node_input, W, improved=True, skel_type=skel_type,
                             **self.gcn_module_kwargs)]
        input_ch = output_ch = W
        for i in range(D-N_P-1):
            gnn_layers += [gcn_module(adj_matrix, input_ch, output_ch, improved=True, skel_type=skel_type,
                                  **self.gcn_module_kwargs)]
        self.gnn_layers = nn.ModuleList(gnn_layers)
        self.mlps = GroupedPNMLP(input_ch=self.W, output_ch=self.output_ch, 
                                 n_nodes=n_nodes, D=N_P)

        if self.exclude_root:
            self.mask = torch.ones(1, len(self.skel_type.joint_names), 1)
            self.mask[:, self.skel_type.root_id, :] = 0.

    def forward(self, inputs, joint_codes=None, valid=None):

        skip_gcn = self.skip_gcn
        n = inputs
        if self.exclude_root or self.mask_root:
            n = n * self.mask.to(n.device)

        for i, l in enumerate(self.gnn_layers):
            n = l(n)
            if i == 0:
                first_n = n

            if i == skip_gcn-1:
                n = torch.cat([n, inputs], dim=-1)

            if (i + 1) < len(self.gnn_layers) and self.nl is not None:
                n = self.nl(n, inplace=True)

        confd = self.mlps(n, valid)

        if self.exclude_root:
            # TODO: currently assume root_id=0
            n, confd = n[:, 1:], confd[:, 1:]

        return n, confd

class SharedValidMLP(BasicGNN):

    def __init__(self, *args, **kwargs):
        super(SharedValidMLP, self).__init__(*args, **kwargs)

    def init_network(self, gcn_module):
        adj_matrix, skel_type = self.adj_matrix, self.skel_type
        per_node_input = self.per_node_input

        self.layers = nn.Sequential(
            nn.Linear(per_node_input, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256), 
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )


    def forward(self, inputs, valid=None, **kwargs):
        valid_b, valid_j = torch.where(valid > 0)
        outputs = torch.zeros(*inputs.shape[:2], self.output_ch)
        outputs[valid_b, valid_j] = self.layers(inputs[valid_b, valid_j])
        return None, outputs
