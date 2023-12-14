# Hiding Visual Information via Obfuscating Adversarial Perturbations
This repository is an implementation of the ICCV 2023 paper "Hiding Visual Information via Obfuscating Adversarial Perturbations".

## Environment settings and libraries we used in our experiments

This project is tested under the following environment settings:
- OS: Ubuntu 18.04.6
- Cuda: 11.1, Cudnn: v8.05
- Python: 3.7.12
- TensorFlow: 1.9.0
- PyTorch: >= 1.10.2
- Torchvision: >= 0.11.3

## Preparation

1. Align and crop the face image to $112 \times 112$ and put it under `data/align`.
2. Put the pre-trained generative model (key model) under the `checkpoints`. We provide a [pre-trained model](https://drive.google.com/drive/folders/1ogIV18DxHNYlnuF4N4dQF-Ft00QwG2ov?usp=sharing) for testing in face recognition tasks.

## Image Encryption for Face Recognition
Running command to encrypt the image:
```python
python main.py --model pix2pix --task face_recognition
```
## Image Encryption for Classification
Running command to encrypt the image:
```python
python main.py --model pix2pix --task classification --src_model Resnet18 --batch_size 10 --root ./data/cif
```

## Acknowledgements
This code is built on [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix), [TIP-IM](https://github.com/ShawnXYang/TIP-IM) and [MS-SSIM](https://github.com/VainF/pytorch-msssim/tree/master). We thank the authors for sharing the codes.
