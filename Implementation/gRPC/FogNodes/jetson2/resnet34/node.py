import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

def first_block(img_part,model,model_parameters):
    print("\tBefore Conv: {}".format(img_part.size()))
    conv1 = nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3, bias = False)
    bn1 = nn.BatchNorm2d(64) 
    conv1,bn1 = assign_model_conv([conv1,bn1],0,model,model_parameters)
    activation1 = nn.ReLU(inplace = True)
    pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding = 1)
    return pool1(activation1(bn1(conv1(img_part))))

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        super(Block, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias = False, stride = stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias = False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size = 1, bias = False, stride = stride)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.activation = nn.ReLU(inplace=True)
        
    def forward(self, x, model, model_parameters):
        print("\tBefore Conv: {}".format(x.size()))
        self.conv1,self.bn1 = assign_model_conv([self.conv1,self.bn1],0,model,model_parameters)
        self.conv2,self.bn2 = assign_model_conv([self.conv2,self.bn2],6,model,model_parameters)

        out1 = self.conv1(x)
        out1 = self.bn1(out1)
        out1 = self.activation(out1)
        
        out1 = self.conv2(out1)
        out1 = self.bn2(out1)
        if self.in_channels != self.out_channels or self.stride > 1:
            self.conv3,self.bn3 = assign_model_conv([self.conv3,self.bn3],12,model,model_parameters)
            out2 = self.conv3(x)
            out2 = self.bn3(out2)
            return self.activation(out1+out2)
        else:
            return self.activation(out1+x)

def assign_model_conv(conv_layers,current_index,model,model_parameters):
    conv_layers[0].weight.data = model[model_parameters[current_index]]
    conv_layers[1].weight.data = model[model_parameters[current_index+1]]
    conv_layers[1].bias.data = model[model_parameters[current_index+2]]
    conv_layers[1].running_mean.data = model[model_parameters[current_index+3]]
    conv_layers[1].running_var.data = model[model_parameters[current_index+4]]
    conv_layers[1].num_batches_tracked.data = model[model_parameters[current_index+5]]
    return conv_layers[0],conv_layers[1]

def assign_model_fc(fc,model,input_units,prev_input_units):
    fc.weight.data = model['fc.weight'][:,prev_input_units: (prev_input_units + input_units)]
    fc.bias.data = model['fc.bias']
    return fc


in_channel = 3
def nn_layer(NN, img_part, part_inx, current_layer, model, prev_input_units, model_dict):
    global in_channel
    out_channel = None
    stride = None
    model_parameters = None
    cfgs = [(64,6), (64,12), (64,12), (64,12), (128,18,2), (128,12), (128,12), (128,12), (256,18,2), (256,12), (256,12), (256,12), (256,12), (256,12), (512,18,2), (512,12), (512,12)]

    if NN == "conv":
        print(f"Current Conv Layer: {current_layer}")

        if len(cfgs[current_layer]) == 2:
            out_channel = cfgs[current_layer][0]
            in_channel = out_channel
            stride = 1
        elif len(cfgs[current_layer]) == 3:
            out_channel = cfgs[current_layer][0]
            stride = cfgs[current_layer][2]

        model_parameters = model_dict[current_layer]

        if current_layer == 0:
            out = first_block(img_part,model,model_parameters)
            return out
        else:
            block = Block(in_channels = in_channel, out_channels = out_channel, stride = stride)
            out = block(img_part,model,model_parameters)
            return out

    elif NN == "ff":
        print(f"Current Fc Layer: {current_layer}")
        num_classes = 12
        img_part = torch.flatten(img_part,1)
 
        print("\tBefore ff: {}".format(img_part.size()))
        fc = nn.Linear(512, num_classes)
        fc.weight.data = model['fc.weight'] # assigning model
        fc.bias.data = model['fc.bias'] # assigning model

        return fc(img_part)

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

def create_model_dict(cfgs,model_list):
    model_dict = {}
    temp = []
    current_model_list_inx = 0
    for i,v in enumerate(cfgs):
        if len(v) == 2:
            out_channel = v[0]
            num_model_parameters = v[1]
        elif len(v) == 3:
            out_channel = v[0]
            num_model_parameters = v[1]
            stride = v[2]
            
        for j in range(current_model_list_inx,current_model_list_inx+num_model_parameters):
            temp.append(model_list[j])
        model_dict.update({i:temp})
        temp = []
        current_model_list_inx += num_model_parameters
    return model_dict


trans_diff = 0
total_time = 0
prev_img_inx = -1
model = None
model_dict = None
layer_comp_time = 0
def fog_node(msg):
    global total_time,trans_diff,prev_img_inx,model,layer_comp_time,model_dict
    try:
        trans_end = time.time()
        message = eval(msg)
        """
        0 = NN  1 = img_part  2 = part_inx  3 = Batch  4 = channel  5 = row  6 = col  7 = current_layer  8 = trans_start  9 = epoch  10 = pretrained_model 11 = prev_input_units
        """
        if prev_img_inx != message[9]:
            total_time = 0
            prev_img_inx = message[9]
            if message[9] == 0:
                if message[10] == "ResNet34":
                    start = time.time()
                    cfgs = [(64,6), (64,12), (64,12), (64,12), (128,18,2), (128,12), (128,12), (128,12), (256,18,2), (256,12), (256,12), (256,12), (256,12), (256,12), (512,18,2), (512,12), (512,12)]
                    model = torch.load("/home/jetson-nano-2/Desktop/fog-2/data/models/resnet34.pt", map_location=torch.device('cpu'))
                    model_list = create_model_list(model)
                    model_dict = create_model_dict(cfgs,model_list)
                    end = time.time()
                    print(f"load model: {(end-start)*1000} ms")
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
        out = nn_layer(message[0], img_part, part_inx, message[7], model, prev_input_units, model_dict)
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
    global trans_diff, total_time, layer_comp_time
    try:
        trans_start = time.time()
        if NN == "conv":
            b = out.shape[0]
            ch = out.shape[1]
            r = out.shape[2]
            col = out.shape[3]
            img_part = out.data.numpy().tobytes()
            message = [NN, img_part, b, ch, r, col, trans_diff, trans_start, layer_comp_time]
            
        elif NN == "ff":
            row = out.shape[0]
            col = out.shape[1]
            img_part = out.data.numpy().tobytes()
            message = [NN, img_part, row, col, total_time, trans_diff, trans_start, layer_comp_time]

        return str(message)
    except Exception as e:
        print("send_message:ERROR")
        print(e)