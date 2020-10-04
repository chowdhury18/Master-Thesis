# https://www.semantics3.com/blog/a-simplified-guide-to-grpc-in-python-6c4e25f0c506/
# generate gRPC class: python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. calculator.proto
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import time
import logging
import csv
import grpc
import message_pb2
import message_pb2_grpc
import importlib.util
spec = importlib.util.spec_from_file_location("module.name", "/home/arnab/Desktop/DNN/gRPC/img_loader_partitioner.py")
user = importlib.util.module_from_spec(spec)
spec.loader.exec_module(user)

custom_logging_format = '%(asctime)s : [%(levelname)s] - %(message)s'
logging.basicConfig(filename= "/home/arnab/Desktop/DNN/gRPC/logs/gRPC_time_state_resnet34_cpu_random_part.log" , filemode="a", level= logging.INFO, format=custom_logging_format)

def send_message(NN, img_part, part_inx, BATCH_SIZE, current_layer, epoch, pretrained_model, prev_input_units):
    try:
        trans_start = time.time()
        channel = img_part.shape[1]
        row = img_part.shape[2]
        col = img_part.shape[3]
        img_part = img_part.numpy().tobytes()
        if NN == "conv":
            msg = [NN, img_part, part_inx, BATCH_SIZE, channel, row, col, current_layer, trans_start, epoch, pretrained_model]
        elif NN == "ff":
            msg = [NN, img_part, part_inx, BATCH_SIZE, channel, row, col, current_layer, trans_start, epoch, pretrained_model, prev_input_units]
        return str(msg)
    except Exception as e:
        print("send_message:ERROR")
        print(e)

transmission_time = 0
transmission_latency  = {}
computation_latency = {}
transmission_data_size = {}
def received_message(msg, current_layer, fog_node):
    global transmission_time, sent_goodput, received_goodput
    try:
        trans_fog_end = time.time()
        message = eval(msg)
        if message[0] == "conv":
            """
            Conv:
            0 = NN  1 = img_part  2 = Batch  3 = channel  4 = row  5 = col  6 = trans_diff  7 = trans_start 8 = layer_comp_time
            """
            trans_user = message[6]
            trans_fog_start = message[7]
            layer_comp_time = message[8] * 1000 # ms
            trans_fog = trans_fog_end - trans_fog_start
            transmission_time += trans_user + trans_fog
            img_part = np.frombuffer(message[1],dtype='float32')
            img_part = torch.from_numpy(img_part.reshape(message[2],message[3],message[4],message[5]))

            sent_data = sent_goodput[fog_node][current_layer]
            received_goodput[fog_node][current_layer] = img_part.element_size() * img_part.nelement() # goodput
            received_data = received_goodput[fog_node][current_layer]
            total_data = (sent_data+received_data)/(1024*1024)

            if fog_node == 0:
                set_value(current_layer, (trans_user + trans_fog), sent_data/(1024*1024), layer_comp_time)
            elif fog_node == 1:
                set_value(current_layer, (trans_user + trans_fog), sent_data/(1024*1024), layer_comp_time)

            print("\tAfter: {}\n".format(img_part.size()))
            return img_part

        elif message[0] == "ff":
            """
            fc:
            0 = NN  1 = img_part  2 = row  3 = col  4 = total_time  5 = trans_diff  6 = trans_start 7 = layer_comp_time
            """
            trans_user = message[5]
            trans_fog_start = message[6]
            layer_comp_time = message[7] * 1000 # ms
            trans_fog = trans_fog_end - trans_fog_start
            transmission_time += trans_user + trans_fog
            img_part = np.frombuffer(message[1],dtype='float32')
            img_part = torch.from_numpy(img_part.reshape(message[2],message[3]))

            sent_data = sent_goodput[fog_node][current_layer]
            received_goodput[fog_node][current_layer] = img_part.element_size() * img_part.nelement() # goodput
            received_data = received_goodput[fog_node][current_layer]
            total_data = (sent_data+received_data)/(1024*1024)

            if fog_node == 0:
                set_value(current_layer, (trans_user + trans_fog), sent_data/(1024*1024), layer_comp_time)    
            elif fog_node == 1:
                set_value(current_layer, (trans_user + trans_fog), sent_data/(1024*1024), layer_comp_time)

            print("\tAfter ff: {}\n".format(img_part.size()))
            m = nn.ReLU()
            out = m(img_part).data > 0
            out = out.int()
            return out
            
    except Exception as e:
        print("received_message:ERROR")
        print(e)


def reset(worker):
    global transmission_latency, computation_latency, transmission_data_size, sent_goodput, received_goodput
    num_layers = 18
    for i in range(num_layers):
        transmission_latency[i] = 0
        computation_latency[i] = 0
        transmission_data_size[i] = 0
    sent_goodput = np.zeros((worker,18))
    received_goodput = np.zeros((worker,18))

def set_value(current_layer, transmission_time, transmission_data, computation_time):
    global transmission_latency, computation_latency, transmission_data_size

    transmission_latency[current_layer] += transmission_time
    computation_latency[current_layer] += computation_time
    transmission_data_size[current_layer] += transmission_data

def write_to_file(epochs,writer,worker):
    # workers: fog(2)
    global transmission_latency, computation_latency, transmission_data_size
    num_layers = 18
    total_trans_time = 0
    total_comp_time = 0
    for i in range(num_layers):
        total_trans_time += transmission_latency[i]
        total_comp_time += computation_latency[i]
        writer.writerow({'layer': i,'transmission_time':transmission_latency[i]/(worker*epochs),'transmission_data':transmission_data_size[i]/(worker*epochs),'computation_time':computation_latency[i]/(worker*epochs)})
    
    logging.info(f"Epochs: {epochs} Transmission time: {total_trans_time/worker} Computation time: {total_comp_time/worker}")
    reset(worker)

def adaptive_partitioning(img,partition_size):
    index = partition_size
    temp = img.detach().numpy()
    temp = torch.from_numpy(temp[:,:,index[0]:index[1],:])
    return temp

sent_goodput = np.zeros((2,18))
received_goodput = np.zeros((2,18))
def main():
    global transmission_time, sent_goodput, received_goodput, temp_trans_fog
    cfgs = [(64,6), (64,12), (64,12), (64,12), (128,18,2), (128,12), (128,12), (128,12), (256,18,2), (256,12), (256,12), (256,12), (256,12), (256,12), (512,18,2), (512,12), (512,12)]
    models = {1:"AlexNet",2:"ResNet34",3:"VGG16",4:"NiN"}
    m = input("Enter Model Number [1: AlexNet 2: ResNet34 3: VGG16 ,4: NiN]: ")
    pretrained_model = models[int(m)]
    MAX_MESSAGE_LENGTH = 104857600 # 100MB
    
    try:
        # write file in csv format
        csv_file = open('/home/arnab/Desktop/DNN/gRPC/logs/gRPC_time_state_resnet34_cpu_random_part.csv','a')
        fieldnames = ['layer', 'transmission_time', 'transmission_data', 'computation_time']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        # initialize variable
        current_layer = None
        kernel_filters = None
        img_part_list = []
        classify_list = []
        prev_input_units = 0
        total_transmission_time = 0
        final_out = None

        # create a channel and a stub (client)
        channel1 = grpc.insecure_channel('192.168.0.106:50051',
                                        options=[
                                        ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
                                        ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),],)
        channel2 = grpc.insecure_channel('192.168.0.107:50051',
                                        options=[
                                        ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
                                        ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),],)

        # create a stub (client)
        stub1 = message_pb2_grpc.CalculatorStub(channel1)
        stub2 = message_pb2_grpc.CalculatorStub(channel2)

        # load dataset
        CA = [float(x) for x in np.random.randint(1, 10, size=2)] # CA temporary size=worker
        print(f"CA: {CA}")
        userObject = user.User(pretrained_model=pretrained_model,CA=CA)
        BATCH_SIZE = userObject.BATCH_SIZE
        worker = userObject.worker
        loader = userObject.image_loader()
        reset(worker) # reset transmission_latency, computation_latency, transmission_data_size

        epoch = 0
        for img,level in loader:
            print(f"Epoch: {epoch}")

            for i in range(len(cfgs)):
                current_layer = i
                print(f"Current Conv Layer: {current_layer}\n")

                if i == 0:
                    input_ = img
                    channel = input_.shape[1]
                    kernel_filters = 7
                else:
                    input_ = final_out
                    channel = input_.shape[1]
                    kernel_filters = 3

                after_part = userObject.random_partition(input_,kernel_filters)
                for j in range(len(after_part)):
                    print(f"Conv: partition: {after_part[j]}")
                    img_part = adaptive_partitioning(input_,after_part[j]) # partitioned image
                    sent_goodput[j][current_layer] = img_part.element_size() * img_part.nelement() # goodput
                    msg = send_message("conv", img_part, after_part[j], BATCH_SIZE, current_layer, epoch, pretrained_model, 0)
                    out = message_pb2.Request(message=msg)

                    if j == 0:
                        r = stub1.Node(out)
                        img_part = received_message(r.message,current_layer,j) 
                        img_part_list.append(img_part)
                    elif j == 1:
                        r = stub2.Node(out)
                        img_part = received_message(r.message,current_layer,j)
                        img_part_list.append(img_part)

                    if len(img_part_list) == 1:
                        final_out = img_part
                    else:
                        final_out = torch.cat((final_out,img_part),2)
    
                if len(img_part_list) == worker:
                    img_part_list = []
                    print("\tAfter Marge: " + str(final_out.size()))
                    total_transmission_time += transmission_time/worker
                    print(f"total_transmission_time: {total_transmission_time}")
                    transmission_time = 0
            
            
            # Adaptive average Pool
            avgpool = nn.AdaptiveAvgPool2d((1, 1))
            out = avgpool(final_out)
            input_ = out
            channel = input_.shape[1]
            current_layer += 1

            print(f"\nCurrent Fc layer: {current_layer}")
            for j in range(worker):
                img_part = input_
                sent_goodput[j][current_layer] = img_part.element_size() * img_part.nelement() # goodput
                msg = send_message("ff", img_part, 0, BATCH_SIZE, current_layer, epoch, pretrained_model, 0)
                out = message_pb2.Request(message=msg)
                if j == 0:
                    r = stub1.Node(out)
                    img_part = received_message(r.message,current_layer,j)
                    classify_list.append(img_part)
                elif j == 1:
                    r = stub2.Node(out)
                    img_part = received_message(r.message,current_layer,j)
                    classify_list.append(img_part)

            if len(classify_list) == worker:
                classify_final = None
                for i in range(len(classify_list)-1):
                    if i == 0:
                        classify_final = np.bitwise_or(classify_list[i].numpy()[:], classify_list[i+1].numpy()[:]) 
                    else:
                        classify_final = np.bitwise_or(classify_final,classify_list[i+1].numpy()[:])
                classify_list = []
                total_transmission_time += transmission_time/worker
                print(f"total_transmission_time: {total_transmission_time}")

            print(f"Transmission time (one node): {total_transmission_time}")
            transmission_time = 0
            total_transmission_time = 0
            
            epoch += 1
            if epoch == 1:
                write_to_file(epoch, writer, worker)
                break
            
    except Exception as e:
        print("main:ERROR")
        print(e)


if __name__ == '__main__':
    main()