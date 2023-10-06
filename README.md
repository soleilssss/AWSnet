## AWSnet
Our paper "***" has been accepted by MedIA, link：https://www.sciencedirect.com/science/article/pii/S1361841522000159

This is the code that was used for our participation in 2020's MyoPS Challenge. 

The code is in the master branch.

The challenge website is available here:
http://www.sdspeople.fudan.edu.cn/zhuangxiahai/0/myops20/

## How to use
This code was cleaned up and made usable for external users, but is still what the authors would like to call 'messy'. We do our best to improve usability so that you can run through our architecture easily.

This includes a network architecture based on reinforcement learning to automatically search for deep supervision weights, not only for the MyoPS2020 challenge, but also for common segmentation tasks.

## Prerequisites
Our code is based on python3.6 and pytorch.

## Training the networks 

python pure_training.py 

train_datapath: Folder to which you downloaded and extracted the training data

val_datapath: Folder to which you downloaded and extracted the val data

init_weight: File to which you stored the initial weight

First go into the `pure_training` and adapt all the paths to match your file system and the download locations of training and test sets.
Then python pure_training.py to train your dataset.

# Citation

If you find the code useful for your research, please cite our paper.

Wang, Kai-Ni, et al. "AWSnet: an auto-weighted supervision attention network for myocardial scar and edema segmentation in multi-sequence cardiac magnetic resonance images." Medical Image Analysis 77 (2022): 102362.
