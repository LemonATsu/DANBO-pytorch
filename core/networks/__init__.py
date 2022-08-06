from .nerf import *
from .danbo import *
from .embedding import *

from .misc import *

def create_nerf(args, shared_nerf_kwargs, data_attrs):

    caster_class = None
    model_fine = None



    if args.nerf_type in ['nerf']:
        nerf_class = NeRF # default
    elif args.nerf_type in ['graph', 'danbo']:
        nerf_class = DANBO
        caster_class = 'graph'
    else:
        raise NotImplementedError(f'nerf class {args.nerf_type} is not implemented.')
    

    # TODO: create embedder

    # TODO: create NeRF

    nerf_class_kwargs = {}
    if 'graph' in args.nerf_type: #args.nerf_type.startswith('graph'):
        nerf_class_kwargs['node_W'] = args.node_W
        nerf_class_kwargs['voxel_res'] = args.voxel_res
        nerf_class_kwargs['voxel_feat'] = args.voxel_feat
        nerf_class_kwargs['rest_pose'] = data_attrs['rest_pose']
        nerf_class_kwargs['backbone'] = args.gnn_backbone
        nerf_class_kwargs['gcn_D'] = args.gcn_D
        nerf_class_kwargs['align_corners'] = args.align_corners
        nerf_class_kwargs['agg_backbone'] = args.agg_backbone
        nerf_class_kwargs['mask_root'] = args.mask_root
        nerf_class_kwargs['adj_self_one'] = args.adj_self_one
        nerf_class_kwargs['gnn_concat'] = args.gnn_concat
        nerf_class_kwargs['opt_scale'] = args.opt_vol_scale
        nerf_class_kwargs['aggregate_dim'] = args.aggregate_dim
        nerf_class_kwargs['init_adj_w'] = args.init_adj_w
        nerf_class_kwargs['attenuate_feat'] = args.attenuate_feat
        nerf_class_kwargs['attenuate_invalid'] = args.attenuate_invalid
        nerf_class_kwargs['agg_W'] = args.agg_W
        nerf_class_kwargs['agg_D'] = args.agg_D
        nerf_class_kwargs['gcn_fc_D'] = args.gcn_fc_D
        nerf_class_kwargs['no_adj'] = args.no_adj
        nerf_class_kwargs['gcn_sep_bias'] = args.gcn_sep_bias
        nerf_class_kwargs['detach_agg_grad'] = args.detach_agg_grad
        nerf_class_kwargs['use_posecode'] = args.opt_posecode

        nerf_class_kwargs['base_scale'] = 0.4
        if args.vol_cal_scale:
            nerf_class_kwargs['skel_profile'] = data_attrs['skel_profile']

    if 'graphrot' in args.nerf_type:
        nerf_class_kwargs['part_dims'] = args.part_dims
        
    model = nerf_class(**shared_nerf_kwargs, **nerf_class_kwargs)

    # Model learned from fine-grained sampling
    if args.N_importance > 0:
        if not args.single_net:
            model_fine = nerf_class(**shared_nerf_kwargs, **nerf_class_kwargs)
        else:
            model_fine = model

    return model, model_fine, caster_class
