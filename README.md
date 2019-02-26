## _SqueezeSegV2_: Improved Model Structure and Unsupervised Domain Adaptation for Road-Object Segmentation from a LiDAR Point Cloud

By Bichen Wu, Xuanyu Zhou, Sicheng Zhao, Xiangyu Yue, Kurt Keutzer (UC Berkeley)

This repository contains a tensorflow implementation of SqueezeSegV2, an improved convolutional neural network model for LiDAR segmentation and unsupervised domain adaptation for road-object segmentation from a LiDAR point cloud. 


Please refer to our video for a high level introduction of this work: https://www.youtube.com/watch?v=ZitFO1_YpNM. For more details, please refer to our paper: https://arxiv.org/abs/1809.08495. If you find this work useful for your research, please consider citing:

    @article{DBLP:journals/corr/abs-1809-08495,
      author    = {Bichen Wu and
                   Xuanyu Zhou and
                   Sicheng Zhao and
                   Xiangyu Yue and
                   Kurt Keutzer},
      title     = {SqueezeSegV2: Improved Model Structure and Unsupervised Domain Adaptation
                   for Road-Object Segmentation from a LiDAR Point Cloud},
      journal   = {ICRA},
      year      = {2019},
    }
## Installation:

The instructions are tested on Ubuntu 16.04 with python 2.7 and tensorflow 1.0 with GPU support. 
- Clone the SqueezeSeg repository:
    ```Shell
    git clone https://github.com/xuanyuzhou98/SqueezeSegV2.git
    ```
    We name the root directory as `$SQSG_ROOT`.

- Setup virtual environment:
    1. By default we use Python2.7. Create the virtual environment
        ```Shell
        virtualenv env
        ```

    2. Activate the virtual environment
        ```Shell
        source env/bin/activate
        ```

- Use pip to install required Python packages:
    ```Shell
    pip install -r requirements.txt
    ```

## Training/Validation
- First, download training and validation data (3.9 GB) from this [link](https://www.dropbox.com/s/pnzgcitvppmwfuf/lidar_2d.tgz?dl=0). This dataset contains LiDAR point-cloud projected to a 2D spherical surface. Refer to our paper for details of the data conversion procedure. This dataset is converted from [KITTI](http://www.cvlibs.net/datasets/kitti/) raw dataset and is distrubited under the [Creative Commons Attribution-NonCommercial-ShareAlike 3.0 License](https://creativecommons.org/licenses/by-nc-sa/3.0/).
    ```Shell
    cd $SQSG_ROOT/data/
    wget https://www.dropbox.com/s/pnzgcitvppmwfuf/lidar_2d.tgz
    tar -xzvf lidar_2d.tgz
    rm lidar_2d.tgz
    ```

- Now we can start training by
    ```Shell
    cd $SQSG_ROOT/
    ./scripts/train.sh -gpu 0,1,2 -image_set train -log_dir ./log/
    ```
   Training logs and model checkpoints will be saved in the log directory.
   
- We can launch evaluation script simutaneously with training
    ```Shell
    cd $SQSG_ROOT/
    ./scripts/eval.sh -gpu 1 -image_set val -log_dir ./log/
    ```
    
- We can monitor the training process using tensorboard.
    ```Shell
    tensorboard --logdir=$SQSG_ROOT/log/
    ```
    Tensorboard displays information such as training loss, evaluation accuracy, visualization of detection results in the training process, which are helpful for debugging and tunning models
