import cv2
import copy
import os, glob

import json
import h5py
import imageio

import torch
import numpy as np

from smplx import SMPL
from .dataset import BaseH5Dataset
from .utils.skeleton_utils import *
from .process_spin import SMPL_JOINT_MAPPER

def read_cameras(data_path):
    
    intrinsic_paths = sorted(glob.glob(os.path.join(data_path, 'intrinsic', '*.txt')))
    c2w_paths = sorted(glob.glob(os.path.join(data_path, 'pose', '*.txt')))
    assert len(intrinsic_paths) == len(c2w_paths)
    
    intrinsics, c2ws = [], []
    for int_path, c2w_path in zip(intrinsic_paths, c2w_paths):
        intrinsics.append(np.loadtxt(int_path).astype(np.float32))
        c2ws.append(np.loadtxt(c2w_path).astype(np.float32))
    intrinsics = np.array(intrinsics)
    focals = np.stack([intrinsics[:, 0, 0], intrinsics[:, 1, 1]],axis=-1)
    centers = intrinsics[..., :2, -1]
    return np.array(intrinsics), focals, centers, swap_mat(np.array(c2ws))

def read_poses(data_path, frames=None):
    json_paths = sorted(glob.glob(os.path.join(data_path, 'transform_smoth3e-2_withmotion', '*.json')))
    if frames is not None:
        json_paths = np.array(json_paths)[frames]
    kp3ds, poses, motions, joints_RTs, Rs, Ts = [], [], [], [], [], []
    for p in json_paths:
        with open(p, 'r') as f:
            json_data = json.load(f)
            kp3d = np.array(json_data['joints']).astype(np.float32)
            pose = np.array(json_data['pose']).reshape(-1, 3).astype(np.float32)
            joints_RT = (np.array(json_data['joints_RT']).astype(np.float32).transpose(2, 0, 1))
            # important: the rotation is from world-to-local. .T to make it local-to-worl
            R = np.array(json_data['rotation']).astype(np.float32).T
            T = np.array(json_data['translation']).astype(np.float32)
            motion = np.array(json_data['motion'])
            kp3ds.append(kp3d)
            poses.append(pose)
            joints_RTs.append(joints_RT)
            Ts.append(T)
            Rs.append(R)
            motions.append(motion)
    return np.array(kp3ds), np.array(poses), np.array(motions), np.array(joints_RTs), np.array(Rs), np.array(Ts)

@torch.no_grad()
def get_smpls_with_global_trans(
        bones,
        betas,
        Rg,
        Tg,
        smpl_path='smpl/SMPL_300',
        gender='MALE',
        joint_mapper=SMPL_JOINT_MAPPER,
        device=None,
    ):
    '''
    Rg: additional global rotation
    Tg: additional global transloation
    '''
    bones = torch.tensor(bones).float().clone().to(device)
    betas = torch.tensor(betas).float().clone().to(device)
    Rg = torch.tensor(Rg).float().to(device)
    Tg = torch.tensor(Tg).float().to(device)
    
    
    smpl_model = SMPL(
                    smpl_path, 
                    gender=gender, 
                    num_betas=betas.shape[-1], 
                    joint_mapper=SMPL_JOINT_MAPPER
                 ).to(device)

    # directly incorporate global rotation R into pose parmaeter
    # Original equation: SMPL(pose) = Rp X + Tp
    # Neural actor additionally add global rotation/translation by: Rg (Rp X + Tp) + Tg
    # Now, we want Rk = RgRp, but still translate to the same location
    # -> (Rk X + Tp) - Tp + RgTp + Tg
    
    # Step 1: get Tp
    dummy = torch.eye(3).reshape(1, 1, 3, 3).expand(len(bones), 24, -1, -1).to(device)
    # assume the body has the same shape since they are the same person
    betas = betas.mean(0, keepdim=True) 
    rest_pose = smpl_model(
            betas=betas,
            body_pose=dummy[:, 1:], 
            global_orient=dummy[:, :1], 
            pose2rot=False,
        ).joints
    Tp = rest_pose[:, :1].clone()
    rest_pose = rest_pose[0].cpu().numpy()
    # center rest pose
    rest_pose -= rest_pose[:1] 
    # this is actually Tp^T Rg^T = Rg
    RgTp = Tp @ Rg.permute(0, 2, 1)
    
    # Step 2: make Rk
    axisang_Rg = rot_to_axisang(Rg)
    bones[:, :1] = axisang_Rg.reshape(-1, 1, 3)
    Rk = axisang_to_rot(bones)
    
    # Step 3: run SMPL to get (Rk X + Tp)
    RkX_Tp = smpl_model(
            betas=betas,
            body_pose=Rk[:, 1:], 
            global_orient=Rk[:, :1], 
            pose2rot=False,
        ).joints

    kp3d = (RkX_Tp - Tp  + RgTp + Tg).cpu().numpy()
    root_locs = kp3d[:, :1]
    l2ws = np.array([get_smpl_l2ws(bone, rest_pose=rest_pose) for bone in bones.cpu().numpy()])
    l2ws[..., :3, -1] += root_locs
    skts = np.linalg.inv(l2ws)
    
    betas = betas.cpu().numpy()
    bones = bones.cpu().numpy()

    return betas, kp3d, bones, skts, rest_pose

def farthest_point_sampling(pts, n_pts=10, init_idx=0):
    idxs = np.zeros((n_pts,)).astype(np.int64)
    idxs[0] = init_idx
    
    distance = ((pts - pts[init_idx:init_idx+1])**2).sum(-1)
    for i in range(1, n_pts):
        idxs[i] = np.argmax(distance)
        d = ((pts - pts[idxs[i]:idxs[i]+1])**2).sum(-1)
        distance = np.where(d < distance, d, distance)
    return idxs

def process_neural_actor_data(
        data_path,
        save_path,
        subject='vlad',
        ext_scale=0.001,
        split='train',
        frames=np.arange(100, 17001),
        training_views=None,
        test_views=[7, 18, 27, 40],
        n_views=20,
        H=940,
        W=1285,
        skel_type=SMPLSkeleton,
        compression='gzip',
        chunk_size=64,
    ):
    
    # set up path for data reading
    subject_path = os.path.join(data_path, subject)
    video_path = os.path.join(subject_path, 'training' if split == 'train' else 'testing')
    h5_path = os.path.join(save_path, f'{subject}_{split}.h5')
    
    _, focals, centers, c2ws = read_cameras(subject_path)
    views = training_views if split == 'train' else test_views
    if split == 'train':
        views = training_views
        if views is None:
            farthest_idxs = np.sort(farthest_point_sampling(c2ws[..., :3, -1], n_pts=n_views))
            print(f'selected cameras {farthest_idxs}')
            views = np.array([i if i not in test_views else i + 1 for i in farthest_idxs])
            views = np.unique(views)
    else:
        views = test_views
        
    # Note: the bones here do not have global rotation. We will process it later
    kp3ds, bones, _, _, Rs, Ts= read_poses(video_path, frames=frames)
    
    # read body shape from reference data
    betas = json.load(open(os.path.join(subject_path, 'raw_smpl', '000000.json'), 'r'))[0]['shapes']
    betas = np.array(betas).repeat(len(frames), 0)
    
    # compute pose-related data
    _, kp3d, bones, skts, rest_pose = get_smpls_with_global_trans(
                                          bones,
                                          betas,
                                          Rs, 
                                          Ts,
                                       )
    
    cyls = get_kp_bounding_cylinder(kp3d,
                                    ext_scale=ext_scale,
                                    skel_type=skel_type,
                                    extend_mm=250,
                                    top_expand_ratio=1.00,
                                    bot_expand_ratio=0.25,
                                    head='y')

    c2ws = c2ws[views]
    focals = focals[views]
    centers = centers[views]
    cam_idxs = np.arange(len(views)).reshape(-1, 1).repeat(len(frames), 1).reshape(-1)
    kp_idxs = np.arange(len(frames)).reshape(1, -1).repeat(len(views), 0).reshape(-1)

    # all data except for the images are ready.
    # since the amount of data is large, we can't collect all frames and write at once
    # have to read and write simultaneously

    if os.path.exists(h5_path):
        print(f'old {h5_path} exist, remove it')
        os.remove(h5_path)
    
    data_dict = {
        'c2ws': c2ws.astype(np.float32),
        'img_pose_indices': cam_idxs.astype(np.int64),
        'kp_idxs': kp_idxs.astype(np.int64),
        'centers': centers.astype(np.float32),
        'focals': focals.astype(np.float32),
        'kp3d': kp3d.astype(np.float32),
        'betas': betas.astype(np.float32),
        'bones': bones.astype(np.float32),
        'skts': skts.astype(np.float32),
        'cyls': cyls.astype(np.float32),
        'rest_pose': rest_pose.astype(np.float32),
    }
    
    h5_file = h5py.File(h5_path, 'w')
    img_shape = (len(frames) * len(views), H, W, 3)
    
    # first, write the basic data
    ds = h5_file.create_dataset('img_shape', (4,), np.int32)
    ds[:] = np.array(img_shape)
    
    for k in data_dict:
        if np.issubdtype(data_dict[k].dtype, np.floating):
            dtype = np.float32
        elif np.issubdtype(data_dict[k].dtype, np.integer):
            dtype = np.int64
        else:
            raise NotImplementedError(f'Unknown datatype for key {k}: {data_dict[k].dtype}')
        ds = h5_file.create_dataset(k, data_dict[k].shape, dtype,
                                    compression=compression)
        ds[:] = data_dict[k][:]

    # next, write image data
    # create datasets
    flatten_shape = (len(views) * len(frames), H * W,)
    img_chunk = (1, chunk_size**2,)
    ds_imgs = h5_file.create_dataset('imgs', flatten_shape + (3,), np.uint8, 
                                     chunks=img_chunk + (3,), compression=compression)
    ds_masks = h5_file.create_dataset('masks', flatten_shape + (1,), np.uint8, 
                                      chunks=img_chunk + (1,), compression=compression)
    # sampling mask is to keep whole
    ds_sampling_masks = h5_file.create_dataset('sampling_masks', flatten_shape + (1,), np.uint8, 
                                               chunks=(1, H * W, 1), compression=compression)    
    

    # dilation kernel for mask
    d_kernel = np.ones((5, 5))
    bkgd = 255 * np.ones((H, W, 3), dtype=np.uint8)
    
    for i, view in enumerate(views):
        reader = imageio.get_reader(os.path.join(video_path, 'rgb_video', f'{view:03d}.avi'), 'avi')
        meta = reader.get_meta_data()
        n_frames = int(meta['fps'] * meta['duration'])
        
        view_ptr = len(frames) * i
        print(f'processing view {i}: cam {view}')
        # set the reader to the right starting point
        reader.get_data(0)
        for j, frame_ptr in enumerate(frames):
            img = reader.get_data(frame_ptr)
            if j % 100 == 0:
                print(f'process frame {j:05d}/{len(frames):05d}')
            backsub = cv2.createBackgroundSubtractorMOG2()
            
            # for background subtraction
            backsub.apply(bkgd)
            mask = backsub.apply(img)[..., None]
            mask[mask < 127] = 0
            mask[mask >= 127] = 1
            sampling_mask = cv2.dilate(
                                mask, 
                                kernel=d_kernel,
                                iterations=1,
                            )[..., None]

            save_ptr = view_ptr + j
            ds_imgs[save_ptr] = img.reshape(H * W, 3)
            ds_masks[save_ptr] = mask.reshape(H * W, 1)
            ds_sampling_masks[save_ptr] = sampling_mask.reshape(H * W, 1)

class NeuralActorDataset(BaseH5Dataset):
    render_skip = 30
    N_render = 15

    def init_meta(self):
        super(NeuralActorDataset, self).init_meta()

        self.has_bg = True
        self.bgs = 255 * np.ones((1, np.prod(self.HW), 3), dtype=np.uint8)
        self.bg_idxs = np.zeros((self._N_total_img,), dtype=np.int64) * 0

    def get_kp_idx(self, idx, q_idx):
        '''
        idx: the actual index(s) for the full .h5 dataset
        q_idx: the 'queried' index(s) received from the sampler,
               may not coincide with idx.
        '''
        # TODO: check if this is right
        return idx % len(self.kp3d), q_idx % len(self.kp3d)
    
    def get_cam_idx(self, idx, q_idx):
        '''
        idx: the actual index(s) for the full .h5 dataset
        q_idx: the 'queried' index(s) received from the sampler,
               may not coincide with idx.
        '''
        return idx // len(self.kp3d), q_idx // len(self.kp3d)

    def _get_subset_idxs(self, render=False):
        '''return idxs for the subset data that you want to train on.
        Returns:
        k_idxs: idxs for retrieving pose data from .h5
        c_idxs: idxs for retrieving camera data from .h5
        i_idxs: idxs for retrieving image data from .h5
        kq_idxs: idx map to map k_idxs to consecutive idxs for rendering
        cq_idxs: idx map to map c_idxs to consecutive idxs for rendering
        '''
        if self._idx_map is not None:
            # queried_idxs
            i_idxs = self._idx_map
            _k_idxs = self._idx_map
            _c_idxs = self._idx_map
            _kq_idxs = np.arange(len(self._idx_map))
            _cq_idxs = np.arange(len(self._idx_map))

        else:
            # queried == actual index
            i_idxs = np.arange(self._N_total_img)
            _k_idxs = _kq_idxs = np.arange(self._N_total_img)
            _c_idxs = _cq_idxs = np.arange(self._N_total_img)

        # call the dataset-dependent fns to get the true kp/cam idx
        k_idxs, kq_idxs = self.get_kp_idx(_k_idxs, _kq_idxs)
        c_idxs, cq_idxs = self.get_cam_idx(_c_idxs, _cq_idxs)

        return k_idxs, c_idxs, i_idxs, kq_idxs, cq_idxs

    def get_meta(self):
        data_attrs = super(NeuralActorDataset, self).get_meta()
        data_attrs['n_views'] = self._N_total_img // len(self.kp3d)
        return data_attrs



if __name__ == '__main__':
    #from renderer import Renderer
    import argparse
    parser = argparse.ArgumentParser(description='Arguments for processing neural actor data')
    parser.add_argument("-s", "--subject", type=str, default="vlad",
                        help='subject to extract')
    parser.add_argument("-p", "--path", type=str, 
                        default="/scratch/st-rhodin-1/users/shihyang/dataset/neuralactor",
                        help='path to save the .ht')
    parser.add_argument("--split", type=str, default="train",
                        help='split to use')
    args = parser.parse_args()
    subject = args.subject
    split = args.split
    save_path = os.path.join(args.path, subject)

    data_path = 'data/neuralactor'
    print(f"Processing {subject}_{split}...")
    if split == 'train':
        frames = np.arange(100, 17000+1)[::2]
        process_neural_actor_data(
            data_path=data_path, 
            save_path=save_path,
            subject=subject, 
            split=split,
            n_views=15,
            frames=frames,
        )
    elif split.startswith('test'):
        if split == 'test':
            frames = np.arange(100, 7000+1)[::10]
        elif split == 'test2':
            frames = np.arange(99, 7000)[::10]
        process_neural_actor_data(
            data_path=data_path, 
            save_path=save_path,
            subject=subject, 
            split='test',
            frames=frames,
        )
    else:
        raise NotImplementedError(f'Split {split} not defined')

