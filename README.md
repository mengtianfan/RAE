# Frequency Domain-Based Reversible Adversarial Attacks for Privacy Protection in IoT
Pytorch implementation for “Frequency Domain-Based Reversible Adversarial Attacks for Privacy Protection in IoT”
![image text]( https://github.com/mengtianfan/RAE/blob/main/img.png)
## Requisites
PyTorch>=1.0 <br>
Python>=3.7 <br>
NVIDIA GPU + CUDA CuDNN <br>
## Prepare data
In this paper, we use the commonly used dataset DIV2K, ImageNet, Caltech-256 and CIFAR-10.  <br>
The DIV2K dataset is available for download at：https://data.vision.ee.ethz.ch/cvl/DIV2K/. <br>
The ImageNet dataset is available for download at：https://image-net.org/challenges/LSVRC/2012/2012-downloads.php#Images. <br>
The Caltech-256 dataset is available for download at：https://data.caltech.edu/records/nyy15-4j048. <br>
The CIFAR-10 dataset is available for download at：https://www.cs.toronto.edu/%7Ekriz/cifar.html. <br>

## RUN
Copy the path to the dataset to args.py and config.py. <br>
RAE generation requires two phases, the first phase is the training phase, you can run train.py inside the train folder for training. The second stage is RAE generation, you can run generation.py inside RAE-generation folder to generate RAE.


## Description of the files in this repository
1.	train.py: Execute this file to train the model
2.	generation.py: Execute this file to generate the RAEs
3.	args.py: Image and model parameters setting
4.	config.py: Hyperparameters setting
5.	model/ : Architecture of the proposed network
6.	calculate_PSNR_SSIM.py: Calculating psnr and ssim
7.	util/: Some necessary functions and files


