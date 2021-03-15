# Solution for ChaLearn 2021 Looking at People Large Scale Signer Independent Isolated SLR CVPR Challenge @ CVPR'2021 by GTM - Uvigo

In this repository you can find the code, logs, pretrained models and instructions for its use.

![alt General Structure](fact_sheets/figures/general_structure.PNG?raw=true "General Structure")


Our contribution for the challenge is largely based on the implementation described in "Disentangling and Unifying Graph Convolutions for Skeleton-Based Action Recognition", CVPR 2020 Oral.
[[PDF](https://arxiv.org/pdf/2003.14111.pdf)]


## Dependencies
- Python >= 3.6
- PyTorch >= 1.2.0
- [NVIDIA Apex](https://github.com/NVIDIA/apex) (auto mixed precision training)
- PyYAML, tqdm, pandas, numpy, tensorboardX
- Openpose (https://github.com/CMU-Perceptual-Computing-Lab/openpose)


## Data Preparation
### Primary directory structure.
Workpace is composed of 4 primary folders:

```
  - dataset/
  - keypoints/
  - data/
  - MS-G3D/
```

To train and evaluate the models, only the MS-G3D and data folders are needed. In the data folder, the network input features are generated from the keypoints extracted from the videos of the original dataset.

### Dataset (Optional)

The dataset used is called AUTSL. It could be downloaded from: http://chalearnlap.cvc.uab.es/dataset/40/description/ 

**Note:** decryption keys are provided on Codalab after registration, based on the schedule of the challenge.

Once downloaded and unzipped, we will only keep the videos with the suffix _color, which we will place inside the folder called 'color'. In our implementation we have used only the videos from this folder.

```
  - dataset\
    - train\
      - color\
        - signer0_sample1_color.mp4
        ...
        - signer42_sample672_color.mp4
    - val\
    - test\
```


### Keypoints (Optional)

Our implementation is based on  keypoints, these keypoints have been extracted using OpenPose : https://github.com/CMU-Perceptual-Computing-Lab/openpose

```
  - keypoints/
    - autsl_generateOpenPoseKps.py
    - train/
      - signer0_sample1000_color/
        - signer0_sample1000_color_000000000000_keypoints.json
        ...
        - signer0_sample1000_color_000000000000_keypoints.json
      - val/
      - test/
      - splits/
        - train
          - signer0_sample1000_color_000000000000_keypoints.json
          ...
          - signer0_sample1000_color_000000000042_keypoints.json
        - val
       - test
```


#### Download keypoints folders: 
**Link:** https://drive.google.com/file/d/1_kXRRddU8szreHR-mOKOsJduhEtZcWnz/view?usp=sharing

Extract and keep the directory structure.

#### Generate keypoints.

The following configuration parameters have been used to generate the keypoins when using openpose:  ' --net_resolution "-1x512" --display 0 --scale_number 4 --scale_gap 0.25 --hand --hand_scale_number 6 --hand_scale_range 0.4 --render-pose=0'
   
Installation guide: https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/installation/0_index.md#compiling-and-running-openpose-from-source

Then, depending on the path where you have installed Openpose, edit the script keypoints/autsl_generateOpenPoseKps.py. To do this, modify the path of the variable "openpose_models_folder". Example: openpose_models_folder = '/home/gts/projects/mvazquez/openpose/models'.

Once Openpose is installed, the video set has been splitted into small parts that allows running it on different GPUs. At least 800 MB of GPU memory will be necessary to use the configuration proposed.

```
cd keypoints
python autsl_generateOpenPoseKps.py --set train --split 0 --gpu 0
..
python autsl_generateOpenPoseKps.py --set train --split 13 --gpu 0
python autsl_generateOpenPoseKps.py --set val --split 0 --gpu 0
python autsl_generateOpenPoseKps.py --set val --split 1 --gpu 0
python autsl_generateOpenPoseKps.py --set test --split 0 --gpu 0
python autsl_generateOpenPoseKps.py --set test --split 1 --gpu 0
```

### Data

```
  - data/
    - autsl/
    - autsl_labels
    - autsl_raw
      - test
      - train
      - val
```

In this section we prepare the labels files and features files '.npy'.

#### Download
**Link:** https://drive.google.com/file/d/1spuwftpg5WloUgv7L8ia4Xcxt8Sy5ZjZ/view?usp=sharing

Extract and keep the directory structure.

#### Generate features data. (Optional)
Inside the downloaded and unzipped file, you will find all the folders, files and code used to generate, from the keypoints, the data prepared to feed the network. But if you want to repeat the keypoint and feature extraction process, you must run the following commands:

```
cd data/autsl_raw
python convert_autsl_labels.py
python convert_autsl_op_data.py --set train
python convert_autsl_op_data.py --set val
python convert_autsl_op_data.py --set test

python autsl_gendata.py
python gen_bone_data.py --dataset autsl
```

**Note:** The 'dataset' and 'keypoints' folders are not necessary to run inference in our implementation. It is key that the features that will feed our network are available in **data/autsl**.

# MS-3GD

In the folder called MS-3GD you will find all the code used in the development of the solution presented to the challenge, as well as the logs and pre-trained models to reproduce the results finally reported.

```
  - MS-3GD/
    - config/
    - eval/
    - feeders/
    - graph/
    - model/
    - pretrained-models/
    - test/
    - work_dir/
    - ensemble_multi.py
    - main_tta.py
    - utils.py
```


## MS-3GD Implementation.
### Keypoints & Graph Definitions.
As indicated above, our implementation is fed only by the keypoints extracted from the videos in the dataset.
#### JOINTS:
Using Openpose, we obtain 2D real-time keypoint detection:
 * 25-keypoint body/foot keypoint estimation
 * 2x21-keypoint hand keypoint estimation.
These would be stored in the folder called /keypoints.

From these keypoints we have discarded the lower-body keypoints and concatenate the rest as follows: kps_body + kps_hand_left + kps_hand_right (a total of 54 keypoints or joints).

The result and their ids can be seen in the figure below.

#### BONES:
We have defined as 'bones' the difference in x,y between two adjacent points. Defining up to 68 'bones'.

The result and their ids can be seen in the figure below.

![alt General Structure](fact_sheets/figures/keypoints_bones.PNG?raw=true "Id joints and bones")

#### GRAPH
Given the joints and bones, and based on their idx, two graphs have been defined for each of the trained and combined models:
* MS-G3D\graph\autsl_bones.py
* MS-G3D\graph\autsl_joints.py

These can be seen grouped in the following excel: [Definition of graphs](fact_sheets/kps_bones_edges.xlsx)


### Data-Augmentation

A number of data augmentation strategies have been applied:
* Flip all the frames of a video previous to its entry into the model, with a probability of 50%.
* Randomly discard joints and bones. With a probability of 10% and applying an increment of this probability based on the confidence value estimated by Openpose for each of these keypoints, discarding with higher probability the keypoints with a lower confidence value.
* Apply a multiplicative random factor within a range [0.9 - 1.1] of the original location of the keypoint or size of bone.

### TTA (Test Time Augmentation)

Data augmentation is used also in inference and results are averaged.

The following configuration is used for the training and evaluation phase:
```
#tta [[flip_bool (boolean), resize (float)]]
tta : [
  [False,1],
  [True,1],
  [False,1.1],
  [True,1.1]
  [False,0.9],
  [True,0.9]
  ]
```



## Training  (Optional)

We have trained two models, one with joints and other with bones:

```
python main_tta.py --work-dir ./work_dir/msg3d_joint_aug_drop_resize_tta_train --config ./config/autsl-skeleton/train_joint_tta.yaml --half --device 0 1
python main_tta.py --work-dir ./work_dir/msg3d_bone_aug_drop_resize_tta_train --config ./config/autsl-skeleton/train_bone_tta.yaml --half --device 0 1
```
-- work-dir. The directory where all the information generated in the training process will be stored: logs, chekpoints and weights.

-- config. The path to the configuration file. 

-- device. Indicate the gpu to use. The training requires approximately 23000MB of RAM, you can use a single GPU or several GPUs. In case you do not have enough computational resources you should reduce the batch-size by adding the following parameters by modifying the configuration file or by command line: ' --batch size 64 --forward_batch_size 32 test_batch_size 32 '

**Note:** 
The pre-trained models are placed in the folder **pretrained-models/**
* *pretrained-model/msg3d_joint_aug_drop_resize_tta_train.pt*
* *pretrained-models/msg3d_bone_aug_drop_resize_tta_train.pt*


## Evaluating  (Optional)

To evaluate the performance of combining the outputs of the models of joints and bones, we used the weights located in the folder called **/pretrained-models**. Then, we combine their outputs and obtain the performance with respect to the validation labels.

```
python main_tta.py --work-dir ./eval/msg3d_joint_aug_drop_resize_tta --config ./config/autsl-skeleton/val_joint_tta.yaml --weights pretrained-models/msg3d_joint_aug_drop_resize_tta_train.pt --device 0
python main_tta.py --work-dir ./eval/msg3d_bone_aug_drop_resize_tta --config ./config/autsl-skeleton/val_bone_tta.yaml --weights pretrained-models/msg3d_bone_aug_drop_resize_tta_train.pt --device 0

python3 ensemble_multi.py --set val --inputs eval/msg3d_joint_aug_drop_resize_tta/ eval/msg3d_bone_aug_drop_resize_tta/
```
-- work-dir. Evaluation phase output directory: logs and outputs.

-- config. Configuration file to be used.

-- weights. Path of the pre-trained model used to estimate the predictions.

-- device. Indicate the gpu to use.

-- set. Specify which set we are combining 'val' or 'test'.

-- inputs. Define the paths where the output files of the evaluation phase from the command that precedes it are located.

As can be seen in the previous commands, it uses the pre-trained model and work-dir included in this repository.

## Generate predictions.csv (Follow these instructions to reproduce the results on test set)

To generate the prediction file we evaluated using the pretrained weights on the data from the test set. 

This test set lacks the corresponding annotations, so the performance of the following executions does not provide any value until the corresponding annotations are available, which are currently not yet released and the results cannot be obtained until the file predictions.csv are uploaded to the platform.

This time, we use the parameter '--csv' to generate the corresponding file 'predictions.csv'.

```
python main_tta.py --work-dir ./test/msg3d_joint_aug_drop_resize_tta --config ./config/autsl-skeleton/test_joint_tta.yaml --weights pretrained-models/msg3d_joint_aug_drop_resize_tta_train.pt --device 0
python main_tta.py --work-dir ./test/msg3d_bone_aug_drop_resize_tta --config ./config/autsl-skeleton/test_bone_tta.yaml --weights pretrained-models/msg3d_bone_aug_drop_resize_tta_train.pt --device 0

python3 ensemble_multi.py --set test --inputs test/msg3d_joint_aug_drop_resize_tta/ test/msg3d_bone_aug_drop_resize_tta/ --csv
```

-- work-dir. Evaluation phase output directory: logs and outputs.

-- config. Configuration file to be used.

-- weights. Path of the pre-trained model used to estimate the predictions.

-- device. Indicate the gpu to use.

-- csv. It will generate a file named **'predictions.csv'** in **MS-3GD/** with the output resulting from the combination process.

**Note:** The file **MS-3GD/predictions.csv** present in this repository corresponds to the file already uploaded to the platform for evaluation.

## Performance

Accuracy on test: 0.961500


## Acknowledgements

This repo is based on
  - [MS-G3D] (https://github.com/kenziyuliu/MS-G3D)
  
    ```
    @inproceedings{liu2020disentangling,
      title={Disentangling and Unifying Graph Convolutions for Skeleton-Based Action Recognition},
      author={Liu, Ziyu and Zhang, Hongwen and Chen, Zhenghao and Wang, Zhiyong and Ouyang, Wanli},
      booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
      pages={143--152},
      year={2020}
    }
    ```


## Contact
Please email `mvazquez@gts.uvigo.es` for further questions
