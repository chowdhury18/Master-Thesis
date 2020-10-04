import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

in_channel = 3
def nn_layer(NN, img_part, part_inx, current_layer, model, prev_input_units, model_dict):
    global in_channel
    weight = None
    bias = None
    out_channel = None
    layer_kernel = None
    layer_stride = None
    layer_padding = None
    cfgs = [192, 160, 96, 'M', 192, 192, 192, 'A', 192, 192, 12, 'A']

    try:
        print(f"Current Layer: {current_layer}")
        if NN == "conv":

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

def create_model_dict(cfg,model_list,kernel_filters,stride,padding):
    model_dict = {}
    model_inx = 0
    for i,v in enumerate(cfg):
        if v == 'M' or v == 'A':
            model_dict.update({i:(v,kernel_filters[i],stride[i],padding[i])})
        else:
            model_dict.update({i:(v,kernel_filters[i],stride[i],padding[i],model_list[model_inx],model_list[model_inx+1])})
            model_inx += 2
            
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
                if message[10] == "NiN":
                    start = time.time()
                    cfgs = [192, 160, 96, 'M', 192, 192, 192, 'A', 192, 192, 12, 'A']
                    kernel_filters = [5,1,1,3,5,1,1,3,3,1,1,8]
                    stride = [1,1,1,2,1,1,1,2,1,1,1,1]
                    padding = [2,0,0,1,2,0,0,1,1,0,0,0]
                    model = torch.load("/home/jetson-nano-1/Desktop/fog-1/data/models/nin.pt", map_location=torch.device('cpu'))
                    model_list = create_model_list(model)
                    model_dict = create_model_dict(cfgs,model_list,kernel_filters,stride,padding)
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
            
        return str(message)
    except Exception as e:
        print("send_message:ERROR")
        print(e)