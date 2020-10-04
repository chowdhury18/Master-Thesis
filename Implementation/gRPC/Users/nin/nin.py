import torch
import torch.nn as nn
import torch.nn.functional as F
import time

in_channel = 3
def nn_layer(NN, img_part, current_layer, model, prev_input_units, model_dict):
    global in_channel
    weight = None
    bias = None
    out_channel = None
    layer_kernel = None
    layer_stride = None
    layer_padding = None
    cfgs = [192, 160, 96, 'M', 192, 192, 192, 'A', 192, 192, 12, 'A']
    #model_list = create_model_list(model) # TODO: remove from layer execution
    #model_dict = create_model_dict(cfgs,model_list) # TODO: remove from layer execution
    try:
        print(f"Current Layer: {current_layer}")
        if NN == "conv":
            #img_part = adaptive_partitioning(img_part,part_inx)

            if len(model_dict[current_layer]) > 4:
                out_channel = model_dict[current_layer][0]
                layer_kernel = model_dict[current_layer][1]
                layer_stride = model_dict[current_layer][2]
                layer_padding = model_dict[current_layer][3]
                weight = model[model_dict[current_layer][4]]
                bias = model[model_dict[current_layer][5]]
            else:
                out_channel = model_dict[current_layer][0]
                layer_kernel = model_dict[current_layer][1]
                layer_stride = model_dict[current_layer][2]
                layer_padding = model_dict[current_layer][3]

            dropout = nn.Dropout(0.5)
                
            if out_channel == 'M':
                print("\tBefore MaxPool: {}".format(img_part.size()))
                m = nn.MaxPool2d(kernel_size=layer_kernel, stride=layer_stride, padding=layer_padding)
                return dropout(m(img_part))
            elif out_channel == 'A':
                print("\tBefore AvgPool: {}".format(img_part.size()))
                m = nn.AvgPool2d(kernel_size=layer_kernel, stride=layer_stride, padding=layer_padding)
                if current_layer != (len(cfgs)-1):
                    return dropout(m(img_part))
                else:
                    return m(img_part)
            else:
                print("\tBefore Conv: {}".format(img_part.size()))
                m0 = nn.Conv2d(in_channel, out_channel, kernel_size=layer_kernel, stride=layer_stride, padding=layer_padding)
                m1 = nn.ReLU(inplace=True)
                m0.weight.data = weight
                m0.bias.data = bias
                in_channel = out_channel
                return m1(m0(img_part))
    except Exception as e:
        print("nn_layer:ERROR")
        print(e)