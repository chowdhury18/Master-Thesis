import torch
import glob
import os
import json
from torch.utils import data
from torchvision.datasets.folder import pil_loader
from torchvision import transforms
#from utils import load_json
import importlib.util
spec = importlib.util.spec_from_file_location("module.name", "/home/arnab/Desktop/dnn-offloading/Models/utils.py")
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)

class BDDDataset(data.Dataset):

    def __init__(self, root, train, transform=None):
        self.root = root
        self.train = train
        self.transform = transform
        self.samples = None
        self.image_namelist_ = None
        self.img_label = None
        self.CATEGORY = None
        self.LABELS = None
        self.category_label()
        
        if self.train:
            self.image_namelist()
        else:
            self.image_namelist_test()
        
        self.prepare_data_by_labels()
        
    def category_label(self):
        self.CATEGORY = ['rider', 'traffic light', 'lane', 'traffic sign', 'bike', 'motor', 'truck', 'bus', 'car', 'drivable area', 'person', 'train']
        self.LABELS={}
        for i,key in enumerate(self.CATEGORY):
            self.LABELS.update({key : i})
    # train imageList  
    def image_namelist(self):
        #train_label_path = "/home/arnab/Desktop/Data/labels/bdd1k_labels_images_train.json"
        train_label_path = os.path.join(self.root, 'labels/bdd1k_labels_images_train.json')
        self.img_label = utils.load_json(train_label_path)
        self.image_namelist_ = []
        for img in self.img_label:
            #print("Name: {}".format(img['name']))
            self.image_namelist_.append(img['name'])
            
    # test imageList
    def image_namelist_test(self):
        #test_label_path = "/home/arnab/Desktop/Data/labels/bdd1k_labels_images_test.json"
        test_label_path = os.path.join(self.root, 'labels/bdd1k_labels_images_test.json')
        self.img_label = utils.load_json(test_label_path)
        self.image_namelist_ = []
        for img in self.img_label:
            #print("Name: {}".format(img['name']))
            self.image_namelist_.append(img['name'])

    # (image->image_stat(label)) is considered
    def prepare_data_by_labels(self):
        self.samples = []
        if self.train:
            image_files = glob.glob(
                os.path.join(self.root, 'images/train/*.jpg'))
            image_dir = os.path.join(self.root, 'images/train')
        else:
            image_files = glob.glob(
                os.path.join(self.root, 'images/test/*.jpg'))
            image_dir = os.path.join(self.root, 'images/test')
        
        for image_file in self.image_namelist_:
            image_path = os.path.join(image_dir,image_file)
            for img_stat in self.img_label:
                if img_stat['name'] == image_file:
                    if os.path.exists(image_path):
                        categories_list = []
                        categories_list_bool = [0] * len(self.CATEGORY)
                        for l in img_stat['labels']:
                            categories_list.append(l['category'])
                            categories_list = list(set(categories_list))
                        for cat in categories_list:
                            categories_list_bool[self.LABELS[cat]] = 1
                        
                        self.samples.append([image_path, torch.Tensor(categories_list_bool)])
                    else:
                        raise FileNotFoundError
    
    def __getitem__(self, index):
        # TODO: handle label dict
        
        image_path, img_stat = self.samples[index]

        image = pil_loader(image_path)

        if self.transform is not None:
            image = self.transform(image)

        return image, img_stat
    

    def __len__(self):
        return len(self.samples)