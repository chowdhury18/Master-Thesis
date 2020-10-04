# Distributed deep learning inference in fog networks

## Abstract
---
oday's smart devices are equipped with powerful integrated chips and built-in heterogeneous sensors that can leverage their potential to execute heavy computation and produce a large amount of sensor data. For instance, modern smart cameras integrate artificial intelligence to capture images that detect any objects in the scene and change parameters, such as contrast and color based on environmental conditions. The accuracy of the object recognition and classification achieved by intelligent applications has improved due to recent advancements in artificial intelligence (AI) and machine learning (ML), particularly, deep neural networks (DNNs).

Despite the capability to carry out some AI/ML computation, smart devices have limited battery power and computing resources. Therefore, DNN computation is generally offloaded to powerful computing nodes such as cloud servers. However, it is challenging to satisfy latency, reliability, and bandwidth constraints in cloud-based AI. Thus, in recent years, AI services and tasks have been pushed closer to the end-users by taking advantage of the fog computing paradigm to meet these requirements. Generally, the trained DNN models are offloaded to the fog devices for DNN inference. This is accomplished by partitioning the DNN and distributing the computation in fog networks.

This thesis addresses offloading DNN inference by dividing and distributing a pre-trained network onto heterogeneous embedded devices. Specifically, it implements the adaptive partitioning and offloading algorithm based on matching theory proposed in an article, titled "Distributed inference acceleration with adaptive dnn partitioning and offloading". The implementation was evaluated in a fog testbed, including Nvidia Jetson nano devices. The obtained results show that the adaptive solution outperforms other schemes (Random and Greedy) with respect to computation time and communication latency.

### Keywords
---
DNN inference, task partitioning, task offloading, distributed algorithm, DNN framework and architectures

**ePrint**: [Distributed deep learning inference in fog networks](https://aaltodoc.aalto.fi/handle/123456789/46082)

## File structure
---
```
.
└─── Algorithms
    │   Partitioning-offloading-Alexnet
    │   Partitioning-offloading-NiN
    |   Partitioning-offloading-ResNet34
    |   Partitioning-offloading-VGG16
└─── Data
    |   README.md
└─── Experiment-Results
└─── Implementation
    └─── gRPC
        └─── FogNodes
        └─── Users
        |   image-loader-and-partitioner
        |   matching-algorithm
└─── Logs
└─── Models
    |   BDDDataset-loader
    |   Models-training-BDDDataset
```

## Description
---
- Algorithm <br>
    |   --> This folder contains the partitioning and offloading algorithms for four different Convolutional Neural Networkss (CNN), namely, [AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf), [VGG](https://arxiv.org/pdf/1409.1556.pdf), [ResNet](https://arxiv.org/pdf/1512.03385.pdf) and [NiN](https://arxiv.org/pdf/1312.4400.pdf). It also contains the definition of Network-in-Network architecture and ResNet34 model architecture implementation.

 - Data <br>
    |   --> This folder contains the information about the dataset that is used in this thesis. [Berkeley DeepDrive](https://bdd-data.berkeley.edu/) dataset is used for implementation of partitioning and offloading algorithms.

 - Experimental-Results <br>
    |   --> This folder contains the results of experiments contucted on four different CNN architectures individually and combined. The graphs show the performance results of the implemented algorithms.

- Implementation <br>
    |   --> This folder contains the implementation of the partitioning and offloading algorithm, and the communication protocols between the user and the fog devices. Implementation folders are seperated based on the devices.

- Logs <br>
    |   --> This folder contains the logging information of the algorithms.

- Models <br>
    |   --> This folder contains the BDDDataset loader. The **bdd-data-loader** extracts the categories of objects from each image and generates a one-hot array for each image. The **training_model** can train four different CNN architectures that are used in this thesis. The trained models are stored seperately for future inference purpose.

