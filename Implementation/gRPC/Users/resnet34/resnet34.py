import torch
import torch.nn as nn
import torch.nn.functional as F
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

in_channel = 3
def nn_layer(NN, img_part, current_layer, model, prev_input_units, model_dict):
    global in_channel
    out_channel = None
    stride = None
    model_parameters = None
    cfgs = [(64,6), (64,12), (64,12), (64,12), (128,18,2), (128,12), (128,12), (128,12), (256,18,2), (256,12), (256,12), (256,12), (256,12), (256,12), (512,18,2), (512,12), (512,12)]
    #model_list = create_model_list(model) # TODO: remove from layer execution
    #model_dict = create_model_dict(cfgs,model_list) # TODO: remove from layer execution
    if NN == "conv":
        print(f"Current Conv Layer: {current_layer}")
        #img_part = adaptive_partitioning(img_part,part_inx)

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
        #img_part = adaptive_partitioning(img_part,part_inx)
        img_part = torch.flatten(img_part,1)
        #input_units = img_part.shape[1]
 
        print("\tBefore ff: {}".format(img_part.size()))
        fc = nn.Linear(512, num_classes)
        fc.weight.data = model['fc.weight'] # assigning model
        fc.bias.data = model['fc.bias'] # assigning model
        #fc = assign_model_fc(fc,model,input_units,prev_input_units) 
        return fc(img_part)