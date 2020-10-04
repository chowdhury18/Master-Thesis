import torch
import torch.nn as nn
import torch.nn.functional as F
import time

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
    m.weight.data = weight.cpu()
    m.bias.data = bias.cpu()
    return m


def assign_weight_bias_ff(classifier,model,input_units,prev_input_units):    
    for key,value in classifier.state_dict().items():
        k = "classifier." + str(key)
        x = key.split(".")
        if x[1] == "weight":
            if x[0] == "1": # "1" to update weight size of first FF layer based on first layer input
                weight = model[k]
                weight = weight[:,prev_input_units:(prev_input_units + input_units)]
                classifier[int(x[0])].weight.data = weight.cpu()
            else:
                classifier[int(x[0])].weight.data = model[k].cpu()
        elif x[1] == "bias":
            classifier[int(x[0])].bias.data = model[k].cpu()
        
    return classifier

in_channel = 3
out_channel = [64,192,384,256,256]
kernel_filters = [11,5,3,3,3]
padding = [2,2,1,1,1]
stride = [4,1,1,1,1]
def nn_layer(NN, img_part, current_layer, model, prev_input_units):
    global in_channel,out_channel,kernel_filters,padding,stride
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
        block = ConvBlock(in_channel,out_channel[current_layer],kernel_filters[current_layer],padding[current_layer],stride[current_layer])
        out = block(img_part,weight,bias,current_layer)
        in_channel = out_channel[current_layer]
        return out
    elif NN == "ff":
        #img_part = adaptive_partitioning(img_part,part_inx)
        img_part = torch.flatten(img_part,1)
        input_units = img_part.shape[1]
 
        block = FcBlock(input_units)
        out = block(img_part,model,prev_input_units)
        return out