# DANBO: Disentangled Articulated Neural Body Representations via Graph Neural Networks 
### [Paper](https://arxiv.org/abs/2205.01666) | [Website](https://lemonatsu.github.io/danbo/) | 
![](imgs/teaser.gif)
>**DANBO: Disentangled Articulated Neural Body Representations via Graph Neural Networks**\
>[Shih-Yang Su](https://lemonatsu.github.io/), [Timur Bagautdinov](https://scholar.google.ch/citations?user=oLi7xJ0AAAAJ&hl=en), and [Helge Rhodin](http://helge.rhodin.de/)\
>ECCV 2022

DANBO is a follow-up work of our [A-NeRF](https://github.com/LemonATsu/A-NeRF) in NeurIPS 2021.
DANBO enables learning a more generalizable 3D body model with better data efficiency.

The repo supports both DANBO and A-NeRF training, allowing for easy comparisons to our methods.


## Cleaning in progress
We are currently cleaning the code, so you may encounter runtime errors when running this repo.
(Update Aug 2): Add environment setup and training instruction. 

<!--
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

Therefore, the above command will render 512x512 the learned SURREAL character with bullet-time effect like the following (resizsed to 256x256):

![](imgs/bullet_time.gif)

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
which will output the rendered images in `render_output/surreal_mesh/mesh_render/` like the following:

![](imgs/mesh_render.gif)

You can change the setting in [`run_render.py`](run_render.py) to create your own rendering configuration.
-->
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

## Training
We provide template training configurations in `configs/` for different settings. 

To train DANBO on the H36M dataset 
```
python run_nerf.py --config configs/h36m_zju/danbo_base.txt --basedir logs  --expname danbo_h36m
```
The trained weights and log can be found in ```logs/danbo_h36m```.

You can also train A-NeRF without pose refinement via
```
python run_nerf.py --config configs/h36m_zju/anerf_base --basedir logs_anerf --num_workers 8 --subject S6 --expname anerf_S6
```
This will train A-NeRF on H36M subject S6 with with 8 worker threads for the dataloader. 

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
