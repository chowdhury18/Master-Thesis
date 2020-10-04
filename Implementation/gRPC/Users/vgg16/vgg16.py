import torch
import torch.nn as nn
import torch.nn.functional as F
import time

def assign_weight_bias_ff(classifier,model,input_units,prev_input_units):
    num_classes = 12
    for key,value in classifier.state_dict().items():
        k = "classifier." + str(key)
        x = key.split(".")
        if x[1] == "weight":
            if x[0] == "0": # "1" to update weight size of first FF layer based on first layer input
                weight = model[k]
                weight = weight[:,prev_input_units: (prev_input_units + input_units)]
                classifier[int(x[0])].weight.data = weight.cpu()
            elif x[0] == "6": # last linear layer weight row to class_num
                weight = model[k]
                weight = weight[:num_classes,:]
                classifier[int(x[0])].weight.data = weight.cpu()
            else:
                classifier[int(x[0])].weight.data = model[k].cpu()
        elif x[1] == "bias":
            if x[0] == "6": # last linear layer bias to class_num
                bias = model[k]
                bias = bias[:num_classes]
                classifier[int(x[0])].bias.data = bias.cpu()
            else:
                classifier[int(x[0])].bias.data = model[k].cpu()        
    return classifier

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

in_channel = 3
def nn_layer(NN, img_part, current_layer, model, prev_input_units, model_dict):
    global in_channel
    weight = None
    bias = None
    out_channel = None
    cfgs = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
    #model_list = create_model_list(model) # TODO: remove from layer execution
    #model_dict = create_model_dict(cfgs,model_list)  # TODO: remove from layer execution
    if NN == "conv":

        if len(model_dict[current_layer]) > 1:
            out_channel = model_dict[current_layer][0]
            weight = model[model_dict[current_layer][1]]
            bias = model[model_dict[current_layer][2]]
        else:
            out_channel = model_dict[current_layer]
            
        if out_channel == 'M':
            print("\tBefore MaxPool: {}".format(img_part.size()))
            m = nn.MaxPool2d(kernel_size=2, stride=2)
            return m(img_part)
        else:
            print("\tBefore Conv: {}".format(img_part.size()))
            m0 = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1)
            m1 = nn.ReLU(inplace=True)
            m0.weight.data = weight
            m0.bias.data = bias
            in_channel = out_channel
            return m1(m0(img_part))
    elif NN == "ff":
        num_classes = 12
        img_part = torch.flatten(img_part,1)
        input_units = img_part.shape[1]
 
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
        return classifier(img_part)
