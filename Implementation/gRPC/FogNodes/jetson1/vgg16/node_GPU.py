import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # GPU

def assign_weight_bias_ff(classifier,model,input_units,prev_input_units):
    num_classes = 12
    print(classifier.state_dict().keys())
    for key,value in classifier.state_dict().items():
        k = "classifier." + str(key)
        x = key.split(".")
        if x[1] == "weight":
            if x[0] == "0": # "1" to update weight size of first FF layer based on first layer input
                weight = model[k]
                weight = weight[:,prev_input_units: (prev_input_units + input_units)]
                classifier[int(x[0])].weight.data = weight # GPU
            elif x[0] == "6": # last linear layer weight row to class_num
                weight = model[k]
                weight = weight[:num_classes,:]
                classifier[int(x[0])].weight.data = weight # GPU
            else:
                classifier[int(x[0])].weight.data = model[k] # GPU
        elif x[1] == "bias":
            if x[0] == "6": # last linear layer bias to class_num
                bias = model[k]
                bias = bias[:num_classes]
                classifier[int(x[0])].bias.data = bias # GPU
            else:
                classifier[int(x[0])].bias.data = model[k] # GPU     
    return classifier


in_channel = 3
def nn_layer(NN, img_part, part_inx, current_layer, model, prev_input_units):
    global in_channel, device, model_list, model_dict
    weight = None
    bias = None
    out_channel = None

    print(f"Current Layer: {current_layer}")
    if NN == "conv":
        #img_part = adaptive_partitioning(img_part,part_inx)
        img_part = img_part.to(device) # GPU
        if len(model_dict[current_layer]) > 1:
            out_channel = model_dict[current_layer][0]
            weight = model[model_dict[current_layer][1]]
            bias = model[model_dict[current_layer][2]]
        else:
            out_channel = model_dict[current_layer]
            
        if out_channel == 'M':
            print("\tBefore MaxPool: {}".format(img_part.size()))
            m = nn.MaxPool2d(kernel_size=2, stride=2)
            m = m.to(device) # GPU
            return m(img_part)
        else:
            print("\tBefore Conv: {}".format(img_part.size()))
            m0 = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1)
            m1 = nn.ReLU(inplace=True)
            m0.weight.data = weight
            m0.bias.data = bias
            m0 = m0.to(device) # GPU
            m1 = m1.to(device) # GPU
            in_channel = out_channel
            return m1(m0(img_part))
    elif NN == "ff":
        num_classes = 12
        #img_part = adaptive_partitioning(img_part,part_inx)
        img_part = torch.flatten(img_part,1)
        input_units = img_part.shape[1]
        img_part = img_part.to(device) # GPU
 
        print("\tBefore ff: {}".format(img_part.size()))
        classifier = nn.Sequential(
            nn.Linear(input_units, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        classifier = assign_weight_bias_ff(classifier,model,input_units,prev_input_units)
        classifier = classifier.to(device) # GPU 
        return classifier(img_part)

def adaptive_partitioning(img,partition_size):
    index = partition_size
    temp = img.detach().numpy()
    temp = torch.from_numpy(temp[:,:,index[0]:index[1],:])
    return temp

def create_model_list(model):
    model_list = []
    for key in model:
        model_list.append(key)        
    return model_list

def create_model_dict(cfg,model_list):
    model_dict = {}
    model_inx = 0
    for i,v in enumerate(cfg):
        if v != 'M':
            model_dict.update({i:(v,model_list[model_inx],model_list[model_inx+1])})
            model_inx += 2
        else:
            model_dict.update({i:v})
    return model_dict

trans_diff = 0
total_time = 0
prev_img_inx = -1
model = None
model_list = None
model_dict = None
layer_comp_time = 0
def fog_node(msg):
    global total_time,trans_diff,prev_img_inx,model,layer_comp_time, model_list, model_dict
    cfgs = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
    try:
        trans_end = time.time()
        message = eval(msg)
        """
        0 = NN  1 = img_part  2 = part_inx  3 = Batch  4 = channel  5 = row  6 = col  7 = current_layer  8 = trans_start  9 = epoch  10 = pretrained_model
        11 = prev_input_units
        """
        if prev_img_inx != message[9]:
            total_time = 0
            prev_img_inx = message[9]
            if message[9] == 0:
                if message[10] == "VGG16":
                    start = time.time()
                    model = torch.load("/home/jetson-nano-1/Desktop/fog-1/data/models/vgg16.pth") # GPU
                    model_list = create_model_list(model)
                    model_dict = create_model_dict(cfgs,model_list)
                    end = time.time()
                    print(f"loading model: {end-start}")
        if message[0] == "ff":
            prev_input_units = message[11]
        else:
            prev_input_units = 0

        part_inx = message[2]
        trans_start = message[8]
        trans_diff = trans_end - trans_start
        img_part = np.frombuffer(message[1],dtype='float32')
        img_part = torch.from_numpy(img_part.reshape(message[3],message[4],message[5],message[6]))
        start = time.time() 
        out = nn_layer(message[0], img_part, part_inx, message[7], model, prev_input_units)
        end = time.time()
        layer_comp_time = end - start
        print(f"layer_comp_time: {layer_comp_time}")
        total_time = total_time + (end - start)
        if message[0] == "conv":
            return send_message(out,"conv")
        elif message[0] == "ff":
            return send_message(out,"ff")
    
    except Exception as e:
        print("fog_node:ERROR")
        print(e)


def send_message(out, NN):
    global trans_diff, total_time, layer_comp_time, device
    try:
        trans_start = time.time()
        if NN == "conv":
            b = out.shape[0]
            ch = out.shape[1]
            r = out.shape[2]
            col = out.shape[3]
            out = out.cpu() # GPU
            img_part = out.detach().numpy().tobytes() # GPU
            message = [NN, img_part, b, ch, r, col, trans_diff, trans_start, layer_comp_time]
            
        elif NN == "ff":
            row = out.shape[0]
            col = out.shape[1]
            out = out.cpu() # GPU
            img_part = out.detach().numpy().tobytes() # GPU
            message = [NN, img_part, row, col, total_time, trans_diff, trans_start, layer_comp_time]

        return str(message)
    except Exception as e:
        print("send_message:ERROR")
        print(e)