# ChaLearn UVIGO

En este repositorio se encuentra el código e indicaciones para su utilización.

Dicha implementación parte de la implementación descrita en "Disentangling and Unifying Graph Convolutions for Skeleton-Based Action Recognition", CVPR 2020 Oral.

[[PDF](https://arxiv.org/pdf/2003.14111.pdf)][[Demo](https://youtu.be/5TcHIIece38)][[Abstract/Supp](https://openaccess.thecvf.com/content_CVPR_2020/html/Liu_Disentangling_and_Unifying_Graph_Convolutions_for_Skeleton-Based_Action_Recognition_CVPR_2020_paper.html)]


## Dependencies
- Python >= 3.6
- PyTorch >= 1.2.0
- [NVIDIA Apex](https://github.com/NVIDIA/apex) (auto mixed precision training)
- PyYAML, tqdm, pandas, numpy, tensorboardX
- Openpose (https://github.com/CMU-Perceptual-Computing-Lab/openpose)


## Data Preparation
### Primary directory structure.
Workpace is composed for 4 primary folders:
-- dataset
-- keypoints
-- data
-- MS-G3D

To train and evaluate the models, only the MS-G3D and data folders are needed. In the data folder, the network input features are generated and stored from the keypoints that have been extracted from the initial dataset.

### Download Datasets

The dataset used is called AUTSL. It could be downloaded from: http://chalearnlap.cvc.uab.es/dataset/40/description/ 
Note, decryption keys are provided on Codalab after registration, based on the schedule of the challenge.

Once it has been downloaded and decompressed, it have been distributed int 3 folders: color, depth and all. According to the suffix _color or _depth contained into the filenames.
In our implementation we have used only the videos from the "color" folder.

-- dataset
  -- train
    -- all
      -- signer0_sample1_color.mp4
        ...
      -- signer42_sample672_depth.mp4
    -- color
    -- depth
  -- val
  -- test


### Generate Keypoints

Our implemented is based on the keypoints, these keypoints had been extracted using OpenPose : https://github.com/CMU-Perceptual-Computing-Lab/openpose

-- keypoints
  -- autsl_generateOpenPoseKps.py
  -- train
    -- signer0_sample1000_color
      -- signer0_sample1000_color_000000000000_keypoints.json
       ...
      -- signer0_sample1000_color_000000000042_keypoints.json
  -- val
  -- test
  -- splits
    -- train
      -- op_train_split_0
        ...
      - op_train_split_13
    -- val
    -- test


#### Download keypoints folders: 
Extract keeping the directory structure.

#### Generate keypoints.

The following configuration parameters have been used to generate the keypoins when using openpose:  ' --net_resolution "-1x512" --display 0 --scale_number 4 --scale_gap 0.25 --hand --hand_scale_number 6 --hand_scale_range 0.4 --render-pose=0'
   
Installation guide: https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/installation/0_index.md#compiling-and-running-openpose-from-source

Then, depending on the path where you have installed Openpose, edit the file keypoints/autsl_generateOpenPoseKps.py. To do this, modify the path of the variable "openpose_models_folder". Ex) openpose_models_folder = '/home/gts/projects/mvazquez/openpose/models'.

Once Openpose is installed. The video set has been divided into small parts that allow it to run on different GPUs. As requirements at least 800 MB of GPU memory will be necessary to use the configuration used.

cd keypoints
python autsl_generateOpenPoseKps.py --set train --split 0 --gpu 0
..
python autsl_generateOpenPoseKps.py --set train --split 13 --gpu 0
python autsl_generateOpenPoseKps.py --set val --split 0 --gpu 0
python autsl_generateOpenPoseKps.py --set val --split 1 --gpu 0
python autsl_generateOpenPoseKps.py --set test --split 0 --gpu 0
python autsl_generateOpenPoseKps.py --set test --split 1 --gpu 0

# Generating Data

-- data
  -- autsl
  -- autsl_labels
  -- autsl_raw
    -- test
    -- train
    -- val

In this section we will prepare the labels files and features files '.npy' that we will use to feed our network.
## Download
Link: 
Extract keep the directory mentioned.

## Generate features data.
Inside the downloaded and unzipped file, you will find all the folders, files and codes used to generate from the keypoints the data prepared to feed the network. But if you want to repeat or replicate the routine that has been followed, you must run the following commands:

cd data/autsl_raw
python convert_autsl_labels.py
python convert_autsl_op_data.py --set train
python convert_autsl_op_data.py --set val
python convert_autsl_op_data.py --set test

python autsl_gendata.py
python gen_bone_data.py --dataset autsl

As well as the 'dataset' and 'keypoints' folders are not necessary to use our implementation. It is key that the features that will feed our network are available in data/autsl.

## Training 

We have trained two models, using joints on the one hand and bones on the other:

python main_tta.py --work-dir ./work_dir/autsl_evaluation/msg3d_joint_aug_drop_resize_tta_train --config ./config/autsl-skeleton/train_joint_tta.yaml --half --device 0 1
python main_tta.py --work-dir ./work_dir/autsl_evaluation/msg3d_bone_aug_drop_resize_tta_train --config ./config/autsl-skeleton/train_bone_tta.yaml --half --device 0 1

-- device X. Indicate the gpu to use. The training requires approximately 23000MB of RAM, you can use a single GPU or several GPUs. In case you do not have enough computational resources you should reduce the batch-size by adding the following parameters by modifying the configuration file or by command line: ' --batch size 64 --forward_batch_size 32 test_batch_size 32 '


# Evaluating

To evaluate the performance we will use the weights corresponding to the epoch with the best accuracy from the previous experiments. And subsequently, we will combine their outputs.

python main_tta.py --work-dir ./eval/msg3d_joint_aug_drop_resize_tta --config ./config/autsl-skeleton/val_joint_tta.yaml --weights pretrained-models\msg3d_joint_aug_drop_resize_tta_train.pt --device 0
python main_tta.py --work-dir ./eval/msg3d_bone_aug_drop_resize_tta --config ./config/autsl-skeleton/val_bone_tta.yaml --weights pretrained-models\msg3d_bone_aug_drop_resize_tta_train.pt --device 0

python3 ensemble_multi.py --set val --inputs eval/msg3d_joint_aug_drop_resize_tta/ eval/msg3d_bone_aug_drop_resize_tta/


# Generate predictions.csv
# TEST & GENERATE PREDICTIONS.CSV

To generate the prediction file that would later be uploaded to the platform, we evaluated using the previous weights in a similar way to how we have evaluated, but in this case using the data from the test set. 
This test set lacks the corresponding annotations, so the performance of the following executions does not provide any value.
This time, we use the parameter '--csv' to generate the corresponding file 'predictions.csv'.

python main_tta.py --work-dir ./test/msg3d_joint_aug_drop_resize_tta --config ./config/autsl-skeleton/test_joint_tta.yaml --weights pretrained-models\msg3d_joint_aug_drop_resize_tta_train.pt --device 0
python main_tta.py --work-dir ./test/msg3d_bone_aug_drop_resize_tta --config ./config/autsl-skeleton/test_bone_tta.yaml --weights pretrained-models\msg3d_bone_aug_drop_resize_tta_train.pt --device 0

python3 ensemble_multi.py --set test --inputs test/msg3d_joint_aug_drop_resize_tta/ test/msg3d_bone_aug_drop_resize_tta/ --csv


# Performance
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