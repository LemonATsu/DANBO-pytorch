import os, glob, json
import cv2
import imageio
import numpy as np
import torch
import h5py

from smplx import SMPL
from aist_plusplus.loader import AISTDataset
from aist_plusplus.utils import ffmpeg_video_read
from torchvision.transforms.functional import center_crop

from .utils.skeleton_utils import *
from .process_spin import SMPL_JOINT_MAPPER, write_to_h5py
from .dataset import BaseH5Dataset

class AISTSubjectWrapper(AISTDataset):
    
    def __init__(self, anno_dir, *args, **kwargs):
        self.anno_dir = anno_dir
        super(AISTSubjectWrapper, self).__init__(anno_dir, *args, **kwargs)
        seq_names = self.mapping_seq2env.keys()
        self.mapping_sub2seq = {}
        self.ignore_seqs = self.read_ignore_list()

        for seq_name in seq_names:
            if seq_name in self.ignore_seqs:
                continue
            subject = seq_name.split('_')[3]
            if subject not in self.mapping_sub2seq:
                self.mapping_sub2seq[subject] = [seq_name]
            else:
                self.mapping_sub2seq[subject].append(seq_name)
    
    def read_ignore_list(self):
        ignore_txt = os.path.join(self.anno_dir, 'ignore_list.txt')
        assert os.path.exists(ignore_txt)
        
        with open(ignore_txt, 'r') as f:
            seq_names = f.readlines()
            ignore_seqs = [seq_name.strip() for seq_name in seq_names]

        return set(ignore_seqs)
    
    def get_subject_seq_names(self, subject, music=None):
        '''
        music: specify the music to use
        Note: for the same choreography, the music only changes the poses slightly
              (in speed, and maybe a bit of the poses, but the dance move is the same)
        '''
        assert subject in self.mapping_sub2seq, f'subject {subject} is not in the dataset'
        seq_names = self.mapping_sub2seq[subject]
        if music is None:
            return seq_names
        selected_seqs = []
        
        for seq_name in seq_names:
            m = int(seq_name.split('_')[-2][-1])
            if m == music:
                selected_seqs.append(seq_name)
        return selected_seqs

    
    def load_frames(self, *args, color='RGB', **kwargs):
        images = super(AISTSubjectWrapper, self).load_frames(*args, **kwargs)
        
        if color == 'RGB':
            images = images[..., ::-1] # original code reads BGR image.
        elif color == 'BGR':
            images = images
        else:
            raise NotImplementedError(f'Unknown color format {color}')
        return images
    
    def load_smpl_data(self, motion_dir, seq_name, apply_scale=True):
        '''
        apply_scale: scale to the camera's scale
        '''
        # load smpl motion
        smpl_poses, smpl_scaling, smpl_trans = self.load_motion(motion_dir, seq_name)
        smpl = SMPL(model_path='smpl/', gender='MALE', batch_size=1,
                    joint_mapper=SMPL_JOINT_MAPPER)
        
        # the shape of global_orient and body_pose doesn't matter match
        # because .forward() will concatenate the two together
        global_orient = torch.from_numpy(smpl_poses[:, 0:1]).float()
        body_pose = torch.from_numpy(smpl_poses[:, 1:]).float()
        transl = torch.from_numpy(smpl_trans / smpl_scaling).float()
        
        # so that we can get the root translation.
        # note that the root location is T + transl
        # where T is the root location after kinematic chain.
        keypoints3d = smpl.forward(
                    global_orient=global_orient,
                    body_pose=body_pose,
                    transl=transl,
        ).joints.detach().numpy()
        
        
        # generate rest pose
        root_locs = keypoints3d[..., :1, :].copy()
        
        dummy_pose = torch.zeros(1, 72).float()
        rest_pose = smpl.forward(
                    global_orient=dummy_pose[:, 0:1],
                    body_pose=dummy_pose[:, 1:],
        ).joints.detach().numpy().reshape(24, 3)
        # remove the unnecessary root translation on rest pose
        rest_pose -= rest_pose[:1] 
        bones = smpl_poses.reshape(-1, 24, 3)
        l2ws = np.array([get_smpl_l2ws(bone, rest_pose) for bone in bones])
        l2ws[..., :3, -1] += root_locs

        if apply_scale:
            l2ws[..., :3, -1] *= smpl_scaling
        
        rest_pose = rest_pose.astype(np.float32)
        l2ws = l2ws.astype(np.float32)
        bones = bones.astype(np.float32)
        kp3d = l2ws[..., :3, -1].copy()
        skts = np.linalg.inv(l2ws)

        return kp3d, bones, skts, rest_pose
    
    def load_camera(self, camera_dir, seq_name, scale_to_pose=False):
        '''
        scale_to_pose: scale the camera extrinsic so that it matches SMPL scale
        '''
        env_name = self.mapping_seq2env[seq_name]
        file_path = os.path.join(camera_dir, f'{env_name}.json')
        assert os.path.exists(file_path), f'File {file_path} does not exist!'
        with open(file_path, 'r') as f:
            params = json.load(f)
        intrinsics, extrinsics, distortions = [], [], []
        if scale_to_pose:
            _, smpl_scaling, _ = self.load_motion(self.motion_dir, seq_name)
        else:
            smpl_scaling = 1.0

        for param_dict in params:
            intrinsic = np.array(param_dict['matrix'])
            R = axisang_to_rot(torch.FloatTensor(param_dict['rotation'])).numpy()
            T = np.array(param_dict['translation']).astype(np.float32)
            extrinsic = np.eye(4).astype(np.float32)
            extrinsic[:3, :3] = R
            extrinsic[:3, -1] = T / smpl_scaling
            
            #W, H = param_dict['size']
            distortion = np.array(param_dict['distortions'])
            
            intrinsics.append(intrinsic)
            extrinsics.append(extrinsic)
            distortions.append(distortion)
        return np.array(intrinsics), np.array(extrinsics), np.array(distortions)
    
    def load_bkgds(self, seg_dir, env):
        seq_names = self.mapping_env2seq[env]
        views = self.VIEWS
        bkgds = []
        for view in views:
            view_bkgds = []
            print(f'handling {env}-{view}')
            for seq_name in seq_names:
                filename = seq_name.replace('cAll', view)
                if os.path.exists(os.path.join(seg_dir, f'{filename}_bg.png')):
                    view_bkgd = imageio.imread(os.path.join(seg_dir, f'{filename}_bg.png'))
                else:
                    file_path = os.path.join(seg_dir, f'{filename}_bg.png')
                    print(f'file {file_path} does not exist')
                if view_bkgd.shape[0] != 1080:
                    continue
                view_bkgds.append(view_bkgd)
            try:
                bkgds.append(np.median(view_bkgds, axis=0).astype(np.uint8))
            except:
                import pdb; pdb.set_trace()
                print
        return bkgds

def compute_bkgds(base_path, anno_path):
    aist_dataset = AISTSubjectWrapper(anno_path)
    envs = aist_dataset.mapping_env2seq.keys()
    envs = list(envs)[-5:]
    #import pdb; pdb.set_trace()
    #print
    
    for env in envs:
        bkgd_path = os.path.join(base_path, 'background', env)
        os.makedirs(bkgd_path, exist_ok=True)
        bkgds = aist_dataset.load_bkgds(os.path.join(base_path, 'segmentation'), env)
        for view, bkgd in zip(aist_dataset.VIEWS, bkgds):
            print(f'writing {env}-{view}')
            imageio.imwrite(os.path.join(bkgd_path, f'{view}_bg.png'), bkgd)
            
def dilate_masks(masks, extend_iter=2):
    d_kernel = np.ones((5, 5))
    dilated_masks = []

    for mask in masks:
        dilated = cv2.dilate(mask, kernel=d_kernel,
                             iterations=extend_iter)
        dilated_masks.append(dilated)

    return np.array(dilated_masks)

from torchvision.transforms.functional import center_crop
# Problem: does the sequence for the same subject always have the same scale?
# Problem: is the keypoint for the same subject always the same -> yes, they don't change skeleton..
def process_aist_data(data_path, subject='d01', training_views=[0, 3, 6], 
                      split='train', fps=10, skel_type=SMPLSkeleton, crop=None,
                      ext_scale=0.001):
    '''
    crop: center crop size (H, W)
    '''
    H, W = 1080, 1920
    
    # base fps of the raw data
    base_fps = 60
    skip = base_fps // fps
    # use their provided dataset helper
    anno_path = os.path.join(data_path, 'AIST++_anno')
    aist_dataset = AISTSubjectWrapper(anno_path)

    if split == 'train':
        views = training_views
    else:
        views = [v for v in range(9) if v not in training_views]
    
    # only take part of the sequences.
    # Note: for the same choreography, different music may affect  
    #       the poses slightly, but the dance move is the same.
    seq_names = aist_dataset.get_subject_seq_names(subject, music=0)

    # read out camera 
    video_names, view_ids, envs = [], [], set()
    
    intrinsics, distortions  = [], []
    c2ws, focals, centers = [], [], []

    camera_path = os.path.join(anno_path, 'cameras')
    motion_path = os.path.join(anno_path, 'motions')
    for view in views:
        # the filenaming starts from 1 instead of 0
        view_id = f'c{view + 1:02d}'
        for seq_name in seq_names:
            video_name = aist_dataset.get_video_name(seq_name, view_id)
            
            # This will read out all 9 cameras.
            # note that we need to read this for different seq, because they have different scaling factor
            intrins, extrins, distorts = aist_dataset.load_camera(camera_path, seq_name, scale_to_pose=True)
            intrin, extrin, distort = intrins[view], extrins[view], distorts[view]
            
            # turn intrinsic into focal length and center
            # and extrinsic into cam2world matrix
            # TODO: would it be easier if we just keep intrinsic a matrix?          
            c2w = swap_mat(np.linalg.inv(extrin)).astype(np.float32)         
            focal = np.stack([intrin[0, 0], intrin[1, 1]]).astype(np.float32)
            center = np.stack([intrin[0, -1], intrin[1, -1]]).astype(np.float32)
            
            if crop is not None:
                # change the intrinsic to fit the crop
                scale_w, scale_h = W / center[0], H / center[1]
                crop_H, crop_W = crop
                center = np.array([crop_W / scale_w, crop_H / scale_h]).astype(np.float32)
                
            intrinsics.append(intrin)
            distortions.append(distort)
            video_names.append(video_name)
            view_ids.append(view)
           
            c2ws.append(c2w)
            focals.append(focal)
            centers.append(center)
            
            envs.add(aist_dataset.mapping_seq2env[seq_name])

    intrinsics = np.array(intrinsics).reshape(-1, 3, 3)
    distortions = np.array(distortions).reshape(-1, 5)
    c2ws = np.array(c2ws).reshape(-1, 4, 4)
    focals = np.array(focals).reshape(-1, 2)
    centers = np.array(centers).reshape(-1, 2)
    
    # read pose data
    # TODO: currently we save the pose for each views separately
    #       because the video lengths are different. 
    #       If we really want to save space, we can use a mapping.
    motion_path = os.path.join(anno_path, 'motions')
    kp3d, bones, skts, rest_pose = [], [], [], None
    for seq_name in seq_names:
        # rest pose is always the same for AIST
        seq_kps, seq_bones, seq_skts, rest_pose = aist_dataset.load_smpl_data(motion_path, 
                                                                              seq_name, 
                                                                              apply_scale=False)
        # skip poses based on the specified fps
        kp3d.extend(seq_kps[::skip])
        bones.extend(seq_bones[::skip])
        skts.extend(seq_skts[::skip])
    kp3d = np.array(kp3d)
    bones = np.array(bones)
    skts = np.array(skts)
    cyls = get_kp_bounding_cylinder(kp3d,
                                    ext_scale=ext_scale,
                                    skel_type=skel_type,
                                    extend_mm=250,
                                    top_expand_ratio=1.00,
                                    bot_expand_ratio=0.25,
                                    head='y')
    kp_idxs = np.arange(len(kp3d)).reshape(1, -1).repeat(len(views), 0).reshape(-1)
    assert len(envs) == 1, f'exception: subject {subject} is filmed with different setting.'
    env = list(envs)[0]
    
    # retrieve frames/segmentations/bkgd
    
    bkgd_path = os.path.join(data_path, 'background')
    video_path = os.path.join(data_path, 'raw_videos')
    mask_path = os.path.join(data_path, 'segmentation')
    
    # first, retrieve and undistort bkgds
    intrins, _, distorts = aist_dataset.load_camera(camera_path, seq_name)
    bkgd_paths = sorted(glob.glob(os.path.join(bkgd_path, env, '*_bg.png')))
    bkgds = [cv2.undistort(imageio.imread(p), intrins[i], distorts[i] * 0.0) 
                                     for i, p in enumerate(bkgd_paths)]
    bkgds = np.array(bkgds)
    
    if crop is not None:
        crop_H, crop_W = crop
        bkgds = center_crop(torch.tensor(bkgds).permute(0, 3, 1, 2), (crop_H, crop_W))
        bkgds = bkgds.permute(0, 2, 3, 1).numpy()

    imgs, masks, sampling_masks = [], [], []
    cam_idxs, bkgd_idxs = [], []
    for i, (video_name, view_id, intrin, distort) in enumerate(zip(video_names, view_ids, intrinsics, distortions)):
        print(f'Processing {i+1}/{len(video_names)}')

        # load images and masks
        seq_imgs = aist_dataset.load_frames(os.path.join(video_path, f'{video_name}.mp4'), 
                                            frame_ids=np.arange(1e6).tolist(), 
                                            fps=fps)
        seq_masks = aist_dataset.load_frames(os.path.join(mask_path, f'{video_name}_alpha2.mp4'), 
                                             frame_ids=np.arange(1e6).tolist(), 
                                             fps=fps)
        # undistort the images, masks and bkgd
        assert len(seq_imgs) == len(seq_masks)
        for j, (seq_img, seq_mask) in enumerate(zip(seq_imgs, seq_masks)):
            seq_imgs[j] = cv2.undistort(seq_img, intrin, distort)
            seq_masks[j] = cv2.undistort(seq_mask, intrin, distort)
        # filter out low-confidence alpha channel
        seq_masks = (seq_masks.mean(axis=-1, keepdims=True) > 128).astype(np.uint8)
       
        if crop is not None:
            # crop images / masks if needed
            crop_H, crop_W = crop
            
            seq_imgs = center_crop(torch.tensor(seq_imgs.copy()).permute(0, 3, 1, 2), (crop_H, crop_W))
            seq_imgs = seq_imgs.permute(0, 2, 3, 1).numpy()
            seq_masks = center_crop(torch.tensor(seq_masks.copy()).permute(0, 3, 1, 2), (crop_H, crop_W))
            seq_masks = seq_masks.permute(0, 2, 3, 1).numpy()
        
        seq_sampling_masks = dilate_masks(seq_masks, extend_iter=3)
        
        # create the camera / bkgd idxs mapping
        seq_cam_idxs = (np.ones((len(seq_imgs),)) * i).astype(np.int32)
        seq_bkgd_idxs = (np.ones((len(seq_imgs),)) * view_id).astype(np.int32)
        
        imgs.extend(seq_imgs)
        masks.extend(seq_masks)
        sampling_masks.extend(seq_sampling_masks)
        cam_idxs.extend(seq_cam_idxs)
        bkgd_idxs.extend(seq_bkgd_idxs)
        
    H, W = imgs[0].shape[-3:-1]
    
    return {'imgs': np.array(imgs),
            'bkgds': np.array(bkgds),
            'bkgd_idxs': np.array(bkgd_idxs),
            'masks': np.array(masks).reshape(-1, H, W, 1),
            'sampling_masks': np.array(sampling_masks).reshape(-1, H, W, 1),
            'c2ws': c2ws.astype(np.float32),
            'img_pose_indices': np.array(cam_idxs),
            'kp_idxs': np.array(kp_idxs),
            'centers': centers.astype(np.float32),
            'focals': focals.astype(np.float32),
            'kp3d': kp3d.astype(np.float32),
            'betas': np.zeros((len(kp3d), 10)).astype(np.float32),
            'bones': bones.astype(np.float32),
            'skts': skts.astype(np.float32),
            'cyls': cyls.astype(np.float32),
            'rest_pose': rest_pose.astype(np.float32),
            }

class AISTDataset(BaseH5Dataset):
    N_render = 10
    render_skip = 100

    def init_meta(self):
        super(AISTDataset, self).init_meta()

        dataset = h5py.File(self.h5_path, 'r')
        self.kp_idxs = dataset['kp_idxs'][:]
        self.cam_idxs = dataset['img_pose_indices'][:]
        dataset.close()
    
    def get_kp_idx(self, idx, q_idx):
        '''
        idx: the actual index(s) for the full .h5 dataset
        q_idx: the 'queried' index(s) received from the sampler,
               may not coincide with idx.
        '''
        return self.kp_idxs[idx], q_idx
    
    def get_cam_idx(self, idx, q_idx):
        '''
        idx: the actual index(s) for the full .h5 dataset
        q_idx: the 'queried' index(s) received from the sampler,
               may not coincide with idx.
        '''
        return self.cam_idxs[idx], q_idx

    def _get_subset_idxs(self, render=False):
        '''
        get the part of data that you want to train on
        '''
        if self._idx_map is not None:
            i_idxs = self._idx_map
            _k_idxs = self._idx_map
            _c_idxs = self._idx_map
            _kq_idxs = np.arange(len(self._idx_map))
            _cq_idxs = np.arange(len(self._idx_map))
        else:
            i_idxs = np.arange(self._N_total_img)
            _k_idxs = _kq_idxs = np.arange(self._N_total_img)
            _c_idxs = _cq_idxs = np.arange(self._N_total_img)

        # call the dataset-dependent fns to get the true kp/cam idx
        k_idxs, kq_idxs = self.get_kp_idx(_k_idxs, _kq_idxs)
        c_idxs, cq_idxs = self.get_cam_idx(_c_idxs, _cq_idxs)

        return k_idxs, c_idxs, i_idxs, kq_idxs, cq_idxs

if __name__ == '__main__':
    #from renderer import Renderer
    import argparse
    parser = argparse.ArgumentParser(description='Arguments for parsing AIST++.')
    parser.add_argument("-s", "--subject", type=str, default="d04",
                        help='subject to extract')
    parser.add_argument("--split", type=str, default="train",
                        help='split to use')
    parser.add_argument("--fps", type=int, default=12,
                        help='fps to extract the data')
    args = parser.parse_args()
    subject = args.subject
    split = args.split
    fps = args.fps

    data_path = 'data/AIST'
    print(f"Processing {subject}_{split}...")
    data = process_aist_data(data_path, subject, split=split, fps=fps, crop=(800, 800))
    write_to_h5py(os.path.join(data_path, f"{subject}_{split}.h5"), data)

