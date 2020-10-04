import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.utils import data
from torchvision import transforms
import importlib.util
spec = importlib.util.spec_from_file_location("module.name", "/home/arnab/Desktop/dnn_offloading/Models/bdddataloader.py")
bddloader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bddloader)

class User():
    def __init__(self, pretrained_model, CA):
        self.pretrained_model = pretrained_model
        self.worker = 2
        #self.CA = [float(x) for x in np.random.randint(1, 10, size=self.worker+1)]
        #self.CA = [0.0, 4.0, 6.0]
        self.CA = CA
        self.modify_CA()
        self.BATCH_SIZE = 1
        if self.pretrained_model == "AlexNet":
            self.IMAGE_DIM = 227 # alexnet
            self.kernel_filters = [11,5,3,3,3] # alexnet
        elif self.pretrained_model == "VGG16":
            self.IMAGE_DIM = 224 # vgg16
            self.kernel_filters = 3 # vgg16
        elif self.pretrained_model == "ResNet34":
            self.IMAGE_DIM = 224 # resnet34
        elif self.pretrained_model == "NiN":
            self.IMAGE_DIM = 32 # NiN
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(self.IMAGE_DIM),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])

    def modify_CA(self):
        self.CA.insert(0,0.0)


    def image_loader(self):
        loader = data.DataLoader(
            bddloader.BDDDataset('/home/arnab/Desktop/Data', train=True, transform=self.transform),
            batch_size=self.BATCH_SIZE,
            shuffle=True)
        
        return loader

    def partition_algorithm(self, NN, img, f):
        partition_size = []
        init = [0] * (self.worker + 1)
        r = img.shape[2]
        c = img.shape[3]
        #self.CA[0] = 0.0
        if r > c:
            m = r
        else:
            m = c

        C2 = np.sum(self.CA[:(self.worker + 1)])
    
        for i in range(1,self.worker+1):
            C1 = np.sum(self.CA[:i+1])
            Pi = C1 / C2
            if NN == "conv":
                init[i] = math.floor(Pi * (m - (f - 1)))
                partition_size.append((init[i-1],init[i]+(f-1)))
            elif NN == "ff":
                init[i] = math.floor(Pi * m)
                partition_size.append((init[i - 1],init[i]))
        return partition_size

    
    def random_partition(self, img, kernel):
        partition_size = []
        row_dimension = img.shape[2]
        first_index = 0
        #second_index = np.random.randint(kernel+1, row_dimension-(kernel+1))
        #second_index = np.random.randint(kernel+1, int(math.floor(row_dimension/2)))
        second_index = np.random.randint(kernel, int(math.floor(row_dimension/2)))

        for i in range(self.worker):
            if i == (self.worker - 1):
                partition_size.append((first_index,row_dimension))
            else:
                partition_size.append((first_index,first_index+second_index))
                first_index = first_index + second_index
        return partition_size


