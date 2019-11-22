# LCD: Learned Cross-domain Descriptors for 2D-3D Matching

This is the official PyTorch implementation of the following publication.

> **LCD: Learned Cross-domain Descriptors for 2D-3D Matching**<br/>
> Quang-Hieu Pham, Mikaela Angelina Uy, Binh-Son Hua, Duc Thanh Nguyen, Gemma Roig, Sai-Kit Yeung<br/>
> *AAAI Conference on Artificial Intelligence, 2020* (**Oral**)<br/>
> [Paper](https://arxiv.org/pdf/1911.09326.pdf),
> [Homepage](https://hkust-vgd.github.io/lcd/)

## Prerequisites
Required PyTorch 1.2 or newer. Some other dependencies are:
- h5py

## 2D-3D Match Dataset
Coming soon!

## Usage
### Pre-trained models
Coming soon!

### Training
To train a model on the 2D-3D Match dataset:

    python train.py --config config.json --logdir logs/LCD

Log files and network parameters will be saved to the `logs/LCD` folder.

### Applications
Coming soon!

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
