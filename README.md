# DANBO: Disentangled Articulated Neural Body Representations via Graph Neural Networks 
### [Paper](https://arxiv.org/abs/2205.01666) | [Website](https://lemonatsu.github.io/danbo/) | [Data](https://github.com/LemonATsu/DANBO-pytorch#dataset)
![](imgs/teaser.gif)
>**DANBO: Disentangled Articulated Neural Body Representations via Graph Neural Networks**\
>[Shih-Yang Su](https://lemonatsu.github.io/), [Timur Bagautdinov](https://scholar.google.ch/citations?user=oLi7xJ0AAAAJ&hl=en), and [Helge Rhodin](http://helge.rhodin.de/)\
>ECCV 2022

DANBO is a follow-up work of our [A-NeRF](https://github.com/LemonATsu/A-NeRF) in NeurIPS 2021.
DANBO enables learning a more generalizable 3D body model with better data efficiency.

The repo supports both DANBO and A-NeRF training, allowing for easy comparisons to our methods.


## Updates
(Update Oct 20): add fast training config (`config/h36m_zju/danbo_fast.txt`), which speeds up training over 3x and less memory consumption with nearly the same performance. 
(Update Sep 26): Add missing file (core/networks/danbo.py). 
(Update Aug 02): Add environment setup and training instruction. 

The current code should work without any issue. We may still improve/change the code here and there.

## Setup

### Setup environment
```
conda create -n danbo python=3.8
conda activate danbo

# install pytorch for your corresponding CUDA environments
pip install torch

# install pytorch3d: note that doing `pip install pytorch3d` directly may install an older version with bugs.
# be sure that you specify the version that matches your CUDA environment. See: https://github.com/facebookresearch/pytorch3d
pip install pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu102_pyt190/download.html

# install other dependencies
pip install -r requirements.txt

```

### Dataset
We are not allowed to share the pre-processed data for H3.6M and MonoPerfcap due to license terms. If you need access to the pre-trained models and the pre-processed dataset, please reach out to `shihyang[at]cs.ubc.ca`.

Note that this repository also support SURREAL dataset used in A-NeRF. Please check the [instruction here](https://github.com/LemonATsu/A-NeRF/tree/main/data) to set up SURREAL dataset.

## Training
We provide template training configurations in `configs/` for different settings. 

To train DANBO on the H36M dataset with L1 loss
```
python run_nerf.py --config configs/h36m_zju/danbo_base.txt --basedir logs  --expname danbo_h36m --loss_fn L1
```
The trained weights and log can be found in ```logs/danbo_h36m```.

**Update**: you can also train DANBO with the *fast configuration*
```
python run_nerf.py --config configs/perfcap/danbo_fast.txt --basedir logs  --expname danbo_perfcap --vol_scale_penalty 0.0001
```
This config speeds up training for 3x with less memory consumption by (1) sampling only within the [per-part volumes](https://github.com/LemonATsu/DANBO-pytorch/blob/main/configs/h36m_zju/danbo_fast.txt#L66), which requires (2) [less samples-per-ray for training and rendering](https://github.com/LemonATsu/DANBO-pytorch/blob/main/configs/h36m_zju/danbo_fast.txt#L60-L61). Note that this is not included in the original paper. The flag `vol_scale_penalty` here constraints the size of the per-part volumes.

You can also train A-NeRF without pose refinement via
```
python run_nerf.py --config configs/h36m_zju/anerf_base --basedir logs_anerf --num_workers 8 --subject S6 --expname anerf_S6
```
This will train A-NeRF on H36M subject S6 with with 8 worker threads for the dataloader. 

## Testing
You can use [`run_render.py`](run_render.py) to render the learned models under different camera motions, or retarget the character to different poses by
```
python run_render.py --nerf_args logs/surreal_model/args.txt --ckptpath logs/surreal_model/150000.tar \
                     --dataset surreal --entry hard --render_type bullet --render_res 512 512 \
                     --white_bkgd --runname surreal_bullet
```
Here, 
- `--dataset` specifies the data source for poses, 
- `--entry` specifices the particular subset from the dataset to render, 
- `--render_type` defines the camera motion to use, and
- `--render_res` specifies the height and width of the rendered images.

The output can be found in `render_output/surreal_bullet/`.
	
You can also extract mesh for the learned character:
```
python run_render.py --nerf_args logs/surreal_model/args.txt --ckptpath logs/surreal_model/150000.tar \
                     --dataset surreal --entry hard --render_type mesh --runname surreal_mesh
```
You can find the extracted `.ply` files in `render_output/surreal_mesh/meshes/`.

To render the mesh as in the paper, run
```
python render_mesh.py --expname surreal_mesh 
```
which will output the rendered images in `render_output/surreal_mesh/mesh_render/`.

You can change the setting in [`run_render.py`](run_render.py) to create your own rendering configuration.

## Citation
```
@inproceedings{su2022danbo,
    title={DANBO: Disentangled Articulated Neural Body Representations via Graph Neural Networks},
    author={Su, Shih-Yang and Bagautdinov, Timur and Rhodin, Helge},
    booktitle={European Conference on Computer Vision},
    year={2022}
}
```
```
@inproceedings{su2021anerf,
    title={A-NeRF: Articulated Neural Radiance Fields for Learning Human Shape, Appearance, and Pose},
    author={Su, Shih-Yang and Yu, Frank and Zollh{\"o}fer, Michael and Rhodin, Helge},
    booktitle = {Advances in Neural Information Processing Systems},
    year={2021}
}
```
## Acknowledgements
- The code is built upon [A-NeRF](https://github.com/LemonATsu/A-NeRF).
<!--
- We use [SPIN](https://github.com/nkolot/SPIN) for estimating the initial 3D poses for our Mixamo dataset.
- We generate the data using [SURREAL](https://github.com/gulvarol/surreal) and [Adobe Mixamo](https://www.mixamo.com/) characters.
-->
