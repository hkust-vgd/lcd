# LCD: Learned Cross-domain Descriptors for 2D-3D Matching

This is the official PyTorch implementation of the following publication:

> **LCD: Learned Cross-domain Descriptors for 2D-3D Matching**<br/>
> Quang-Hieu Pham, Mikaela Angelina Uy, Binh-Son Hua, Duc Thanh Nguyen, Gemma Roig, Sai-Kit Yeung<br/>
> *AAAI Conference on Artificial Intelligence, 2020* (**Oral**)<br/>
> [**Paper**](https://arxiv.org/pdf/1911.09326.pdf) | [**Homepage**](https://hkust-vgd.github.io/lcd/)

## 2D-3D Match Dataset
[**Download**](http://103.24.77.34:8080/2d3dmatch/)

We collect a new dataset of 2D-3D correspondences by leveraging the
availability of several 3D datasets from RGB-D scans. Specifically, we use the
data from [SceneNN](http://scenenn.net/) and [3DMatch](http://3dmatch.cs.princeton.edu/).
Our training dataset consists of 110 RGB-D scans, of which 56 scenes are from
SceneNN and 54 scenes are from 3DMatch.  The 2D-3D correspondence data is
generated as follows. Given a 3D point which is randomly sampled from a 3D
point cloud, we extract a set of 3D patches from different scanning views.  To
find a 2D-3D correspondence, for each 3D patch, we re-project its 3D position
into all RGB-D frames for which the point lies in the camera frustum, taking
occlusion into account. We then extract the corresponding local 2D patches
around the re-projected point. In total, we collected around **1.4 millions**
2D-3D correspondences.

## Usage
### Prerequisites
Required PyTorch 1.2 or newer. Some other dependencies are:
- h5py
- [Open3D](http://www.open3d.org/)

### Pre-trained models
We released three pre-trained LCD models with different descriptor size: LCD-D256, LCD-D128, and LCD-D64.
All of the models can be found in the `logs` folder.

### Training
After downloading our dataset, put all of the hdf5 files into the `data/train` folder.

To train a model on the 2D-3D Match dataset, use the following command:

    $ python train.py --config config.json --logdir logs/LCD

Log files and network parameters will be saved to the `logs/LCD` folder.

### Applications
#### Aligning two point clouds with LCD
This demo aligns two 3D colored point clouds using our pre-trained LCD descriptor with RANSAC.
How to run:

    $ python -m apps.align_point_cloud samples/000.ply samples/002.ply --logdir logs/LCD-D256/

For more information, use the `--help` option.

After aligning two input point clouds, the final registration result will be shown. For example:

<p align="center">
  <img src="https://github.com/hkust-vgd/lcd/blob/master/assets/aligned.png?raw=true" alt="Aligned point clouds"/>
</p>

> **Note**: This demo requires Open3D installed.

### Prepare your own dataset
Coming soon!

## Citation
If you find our work useful for your research, please consider citing:

    @inproceedings{pham2020lcd,
      title = {{LCD}: {L}earned cross-domain descriptors for 2{D}-3{D} matching},
      author = {Pham, Quang-Hieu and Uy, Mikaela Angelina and Hua, Binh-Son and Nguyen, Duc Thanh and Roig, Gemma and Yeung, Sai-Kit},
      booktitle = {AAAI Conference on Artificial Intelligence},
      year = 2020
    }

Please also cite the [3DMatch](http://3dmatch.cs.princeton.edu/) paper if you use our dataset.

## License
Our code is released under BSD 3-Clause license (see [LICENSE](LICENSE) for more details).

Our dataset is released under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) license.

**Contact**: Quang-Hieu Pham (pqhieu1192@gmail.com)
