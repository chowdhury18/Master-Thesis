import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # GPU

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, padding, stride):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel, padding=padding, stride = stride)
        self.activation = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        
    def forward(self, x, weight, bias, current_layer):
        print("\tBefore Conv: {}".format(x.size()))
        self.conv = assign_weight_bias(self.conv,weight,bias)
        x = self.conv(x)
        x = self.activation(x)
        if current_layer == 0 or current_layer == 1 or current_layer == 4:
            return self.pool(x)
        else:
            return x

class FcBlock(nn.Module):
    def __init__(self, in_channels,num_classes = 12):
        super(FcBlock, self).__init__()
        self.in_channels = in_channels

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(in_channels, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        
    def forward(self, x, model, prev_input_units):
        print("\tBefore fc: {}".format(x.size()))
        self.classifier = assign_weight_bias_ff(self.classifier,model,self.in_channels,prev_input_units)
        return self.classifier(x)

def assign_weight_bias(m,weight,bias):
    m.weight.data = weight # GPU
    m.bias.data = bias # GPU
    return m

def assign_weight_bias_ff(classifier,model,input_units,prev_input_units):    
    for key,value in classifier.state_dict().items():
        k = "classifier." + str(key)
        x = key.split(".")
        if x[1] == "weight":
            if x[0] == "1": # "1" to update weight size of first FF layer based on first layer input
                weight = model[k]
                weight = weight[:,prev_input_units:(prev_input_units + input_units)]
                classifier[int(x[0])].weight.data = weight # GPU
            else:
                classifier[int(x[0])].weight.data = model[k] # GPU
        elif x[1] == "bias":
            classifier[int(x[0])].bias.data = model[k] # GPU
        
    return classifier


in_channel = 3
out_channel = [64,192,384,256,256]
kernel_filters = [11,5,3,3,3]
padding = [2,2,1,1,1]
stride = [4,1,1,1,1]

def nn_layer(NN, img_part, part_inx, current_layer, model, prev_input_units):
    global in_channel,out_channel,kernel_filters,padding,stride,device
    weight = None
    bias = None
    if NN == "conv":
        c_l = current_layer * 2
        i = 0
        for key, value in model.items(): 
            if i == c_l:
                weight = model[key]
            elif i == (c_l + 1):
                bias = model[key]
                break
            i += 1
        #img_part = adaptive_partitioning(img_part,part_inx)
        img_part = img_part.to(device) # GPU
        block = ConvBlock(in_channel,out_channel[current_layer],kernel_filters[current_layer],padding[current_layer],stride[current_layer])
        block = block.to(device) # GPU
        out = block(img_part,weight,bias,current_layer)
        in_channel = out_channel[current_layer]
        return out
    elif NN == "ff":
        #img_part = adaptive_partitioning(img_part,part_inx)
        img_part = torch.flatten(img_part,1)
        input_units = img_part.shape[1]
        img_part = img_part.to(device) # GPU

        block = FcBlock(input_units)
        block = block.to(device) # GPU
        out = block(img_part,model,prev_input_units)
        return out

def adaptive_partitioning(img,partition_size):
    index = partition_size
    temp = img.detach().numpy()
    temp = torch.from_numpy(temp[:,:,index[0]:index[1],:])
    return temp

trans_diff = 0
total_time = 0
prev_img_inx = -1
model = None
layer_comp_time = 0

def fog_node(msg):
    global total_time,trans_diff,prev_img_inx,model,layer_comp_time
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
                if message[10] == "AlexNet":
                    start = time.time()
                    model = torch.load("/home/jetson-nano-2/Desktop/fog-2/data/models/alexnet.pt") # GPU
                    end = time.time()
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
        print(f"layer computation time: {layer_comp_time*1000}ms")
        total_time = total_time + (end - start)
        if message[0] == "conv":
            return send_message(out,"conv")
        elif message[0] == "ff":
            return send_message(out,"ff")
    
    except Exception as e:
        print("fog_node:ERROR")
        print(e)


def send_message(out, NN):
    global trans_diff, total_time, layer_comp_time
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