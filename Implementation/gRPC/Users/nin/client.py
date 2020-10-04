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
#logging.basicConfig(filename= "/home/arnab/Desktop/DNN/gRPC/logs/gRPC_user_vgg16_cpu.log" , filemode="a", level= logging.INFO, format=custom_logging_format)
logging.basicConfig(filename= "/home/arnab/Desktop/DNN/gRPC/logs/gRPC_user_vgg16_cpu_msg_size.log" , filemode="a", level= logging.INFO, format=custom_logging_format)

def send_message(NN, img_part, part_inx, BATCH_SIZE, current_layer, epoch, pretrained_model, prev_input_units):
    try:
        trans_start = time.time()
        channel = img_part.shape[1]
        row = img_part.shape[2]
        col = img_part.shape[3]
        img_part = img_part.numpy().tobytes()
        if NN == "conv":
            msg = [NN, img_part, part_inx, BATCH_SIZE, channel, row, col, current_layer, trans_start, epoch, pretrained_model]
            
        return str(msg)
    except Exception as e:
        print("send_message:ERROR")
        print(e)

transmission_time = 0
def received_message(msg, current_layer, fog_node, writer):
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
                writer.writerow({'layer': current_layer,'transmission_time':(trans_user + trans_fog),'transmission_data':sent_data/(1024*1024),'computation_time':layer_comp_time})
            elif fog_node == 1:
                writer.writerow({'layer': current_layer,'transmission_time':(trans_user + trans_fog),'transmission_data':sent_data/(1024*1024),'computation_time':layer_comp_time})

            print("\tAfter: {}\n".format(img_part.size()))
            return img_part
            
    except Exception as e:
        print("received_message:ERROR")
        print(e)

def adaptive_partitioning(img,partition_size):
    index = partition_size
    temp = img.detach().numpy()
    temp = torch.from_numpy(temp[:,:,index[0]:index[1],:])
    return temp

sent_goodput = np.zeros((2,19))
received_goodput = np.zeros((2,19))
def main():
    global transmission_time, sent_goodput, received_goodput, temp_trans_fog
    cfgs = [192, 160, 96, 'M', 192, 192, 192,'A', 192, 192, 12, 'A']
    kernel_filters = [5,1,1,3,5,1,1,3,3,1,1,8]
    models = {1:"AlexNet",2:"ResNet34",3:"VGG16",4:"NiN"}
    m = input("Enter Model Number [1: AlexNet 2: ResNet34 3: VGG16 4: NiN]: ")
    pretrained_model = models[int(m)]
    CA = [4.0, 6.0] # capability temporary
    MAX_MESSAGE_LENGTH = 104857600 # 100MB
    # open a gRPC channel
    try:
        # write file in csv format
        #csv_file = open('/home/arnab/Desktop/DNN/gRPC/logs/gRPC_time_state_vgg16_cpu.csv','a')
        csv_file1 = open('/home/arnab/Desktop/DNN/gRPC/logs/gRPC_time_state_nin_cpu_node_1.csv','a')
        csv_file2 = open('/home/arnab/Desktop/DNN/gRPC/logs/gRPC_time_state_nin_cpu_node_2.csv','a')
        fieldnames = ['layer', 'transmission_time', 'transmission_data', 'computation_time']
        writer1 = csv.DictWriter(csv_file1, fieldnames=fieldnames)
        writer1.writeheader()
        writer2 = csv.DictWriter(csv_file2, fieldnames=fieldnames)
        writer2.writeheader()

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

        current_layer = None
        img_part_list = []
        classify_list = []
        prev_input_units = 0
        total_transmission_time = 0
        final_out = None
        userObject = user.User(pretrained_model=pretrained_model,CA=CA)
        BATCH_SIZE = userObject.BATCH_SIZE
        worker = userObject.worker
        loader = userObject.image_loader()

        epoch = 0
        for img,level in loader:
            print(f"Epoch: {epoch}")

            for i in range(len(cfgs)):
                current_layer = i
                print(f"Current Conv Layer: {current_layer}\n")
                if i == 0:
                    input_ = img
                    channel = input_.shape[1]
                else:
                    input_ = final_out
                    channel = input_.shape[1]

                after_part = userObject.partition_algorithm("conv", input_, f=kernel_filters[current_layer])
                for j in range(len(after_part)):
                    img_part = adaptive_partitioning(input_,after_part[j]) # partitioned image
                    sent_goodput[j][current_layer] = img_part.element_size() * img_part.nelement() # goodput
                    msg = send_message("conv", img_part, after_part[j], BATCH_SIZE, current_layer, epoch, pretrained_model, 0)
                    out = message_pb2.Request(message=msg)
                    if j == 0:
                        r = stub1.Node(out)
                        img_part = received_message(r.message,current_layer,j,writer1) # goodput
                        img_part_list.append(img_part)
                    elif j == 1:
                        r = stub2.Node(out)
                        img_part = received_message(r.message,current_layer,j,writer2) # goodput
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
                
                #break # first conv layer
            
            # Adaptive average Pool
            avgpool = nn.AdaptiveAvgPool2d((1, 1))
            final_out = avgpool(final_out)
            final_out = torch.flatten(final_out,1)
            m = nn.ReLU()
            final_out = m(final_out).data > 0
            final_out = final_out.int()

            print(f"Final Result: {final_out}")
            print(f"Transmission time (one node): {total_transmission_time}")
            transmission_time = 0
            total_transmission_time = 0

            epoch += 1
            if epoch == 1:
                break
            
            #break # one epoch
    except Exception as e:
        print("main:ERROR")
        print(e)


    """
    msg = calculator_pb2.Request(msg="Value: ")
    number = calculator_pb2.Request(value=16)
    response_value = stub1.SquareRoot(number)
    response_msg = stub1.MessagePrint(msg)
    print(response_msg.msg,response_value.value)
    """

if __name__ == '__main__':
    main()