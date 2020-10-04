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
spec = importlib.util.spec_from_file_location("module.name", "/home/arnab/Desktop/DNN/gRPC/matching.py")
match = importlib.util.module_from_spec(spec)
spec.loader.exec_module(match)
spec = importlib.util.spec_from_file_location("module.name", "/home/arnab/Desktop/DNN/gRPC/Users/nin/nin.py")
nin = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nin)

custom_logging_format = '%(asctime)s : [%(levelname)s] - %(message)s'
logging.basicConfig(filename= "/home/arnab/Desktop/DNN/gRPC/logs/gRPC_time_state_nin_cpu_greedy_part.log" , filemode="a", level= logging.INFO, format=custom_logging_format)

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
transmission_latency  = {}
computation_latency = {}
transmission_data_size = {}
def received_message(msg, current_layer):
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

            sent_data = sent_goodput[current_layer]
            received_goodput[current_layer] = img_part.element_size() * img_part.nelement() # goodput
            received_data = received_goodput[current_layer]
            total_data = (sent_data+received_data)/(1024*1024)

            set_value(current_layer, (trans_user + trans_fog), sent_data/(1024*1024), layer_comp_time)

            print("\tAfter: {}\n".format(img_part.size()))
            return img_part
            
    except Exception as e:
        print("received_message:ERROR")
        print(e)

def reset():
    global transmission_latency, computation_latency, transmission_data_size, sent_goodput, received_goodput
    num_layers = 12
    for i in range(num_layers):
        transmission_latency[i] = 0
        computation_latency[i] = 0
        transmission_data_size[i] = 0
    sent_goodput = np.zeros(12)
    received_goodput = np.zeros(12)

def set_value(current_layer, transmission_time, transmission_data, computation_time):
    global transmission_latency, computation_latency, transmission_data_size

    transmission_latency[current_layer] += transmission_time
    computation_latency[current_layer] += computation_time
    transmission_data_size[current_layer] += transmission_data

def write_to_file(epochs,writer,worker):
    # workers: fog(1)
    global transmission_latency, computation_latency, transmission_data_size
    num_layers = 12
    total_trans_time = 0
    total_comp_time = 0
    for i in range(num_layers):
        total_trans_time += transmission_latency[i]
        total_comp_time += computation_latency[i]
        writer.writerow({'layer': i,'transmission_time':transmission_latency[i]/(worker*epochs),'transmission_data':transmission_data_size[i]/(worker*epochs),'computation_time':computation_latency[i]/(worker*epochs)})
    
    logging.info(f"Epochs: {epochs} Transmission time: {total_trans_time/worker} Computation time: {total_comp_time/worker}")
    reset()


def adaptive_partitioning(img,partition_size):
    index = partition_size
    temp = img.detach().numpy()
    temp = torch.from_numpy(temp[:,:,index[0]:index[1],:])
    return temp

sent_goodput = np.zeros(12)
received_goodput = np.zeros(12)
def main():
    global transmission_time, sent_goodput, received_goodput, temp_trans_fog
    models = {1:"AlexNet",2:"ResNet34",3:"VGG16",4:"NiN"}
    m = input("Enter Model Number [1: AlexNet 2: ResNet34 3: VGG16 4: NiN]: ")
    pretrained_model = models[int(m)]
    MAX_MESSAGE_LENGTH = 104857600 # 100MB
    
    try:
        # write file in csv format
        csv_file = open('/home/arnab/Desktop/DNN/gRPC/logs/gRPC_time_state_nin_cpu_greedy_part.csv','a')
        fieldnames = ['layer', 'transmission_time', 'transmission_data', 'computation_time']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        # initialize variable
        current_layer = None
        img_part_list = []
        classify_list = []
        prev_input_units = 0
        total_transmission_time = 0
        final_out = None
        num_layers = 12 # 9 conv layers and 3 maxpool layers

        # swap matching algorithm
        matching = match.Matching()
        greedy_random_match = matching.greedy_random_match()
        print(f"Greedy Random Match: {greedy_random_match}")
        matched_fog_node = greedy_random_match[0][1] # one user matched with one best fog node.
        matched_fog_node = matched_fog_node + ":50051"

        # create a channel and a stub (client)
        channel = grpc.insecure_channel(matched_fog_node,
                                        options=[
                                        ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
                                        ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),],)
        stub = message_pb2_grpc.CalculatorStub(channel)

        
        # load dataset
        userObject = user.User(pretrained_model=pretrained_model,CA=matching.F_CA)
        BATCH_SIZE = userObject.BATCH_SIZE
        worker = userObject.worker
        loader = userObject.image_loader()
        reset() # reset transmission_latency, computation_latency, transmission_data_size

        epoch = 0
        for img,level in loader:
            print(f"Epoch: {epoch}")

            # One user matches with one fog node (NO PARTITION)
            for layer in range(num_layers):
                current_layer = layer
                if current_layer == 0:
                    input_ = img
                    channel = input_.shape[1]
                else:
                    input_ = final_out
                    channel = input_.shape[1]

                sent_goodput[current_layer] = input_.element_size() * input_.nelement() # goodput
                msg = send_message("conv", input_ , 0 , BATCH_SIZE, current_layer, epoch, pretrained_model, 0)
                out = message_pb2.Request(message=msg)
                r = stub.Node(out)
                img_part = received_message(r.message, current_layer)
                final_out = img_part
                total_transmission_time += transmission_time
                transmission_time = 0

            # Adaptive average Pool
            avgpool = nn.AdaptiveAvgPool2d((1, 1))
            final_out = avgpool(final_out)
            final_out = torch.flatten(final_out,1)
            m = nn.ReLU()
            final_out = m(final_out).data > 0
            final_out = final_out.int()
            print(f"Final_out: {final_out}")
            print(f"total_transmission_time: {total_transmission_time}")
            
            transmission_time = 0
            total_transmission_time = 0

            epoch += 1
            if epoch == 1:
                write_to_file(epoch, writer, 1)
                break
            
    except Exception as e:
        print("main:ERROR")
        print(e)


if __name__ == '__main__':
    main()