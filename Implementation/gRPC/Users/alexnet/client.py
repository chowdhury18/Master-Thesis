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
logging.basicConfig(filename= "/home/arnab/Desktop/DNN/gRPC/logs/gRPC_user_alexnet_cpu.log" , filemode="a", level= logging.INFO, format=custom_logging_format)

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
total_computation_time = 0
def received_message(msg, current_layer, fog_node, writer):
    global transmission_time, total_computation_time
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

            sent_data = sent_goodput[fog_node][current_layer] # goodput
            received_goodput[fog_node][current_layer] = img_part.element_size() * img_part.nelement() # goodput
            received_data = received_goodput[fog_node][current_layer] # goodput
            total_data = (sent_data+received_data)/(1024*1024)
            if fog_node == 0:
                writer.writerow({'layer': current_layer,'transmission_time':(trans_user + trans_fog),'transmission_data':total_data,'computation_time':layer_comp_time})
            elif fog_node == 1:
                writer.writerow({'layer': current_layer,'transmission_time':(trans_user + trans_fog),'transmission_data':total_data,'computation_time':layer_comp_time})
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
            total_computation_time = total_computation_time + message[4]
            img_part = np.frombuffer(message[1],dtype='float32')
            img_part = torch.from_numpy(img_part.reshape(message[2],message[3]))

            sent_data = sent_goodput[fog_node][current_layer] # goodput
            received_goodput[fog_node][current_layer] = img_part.element_size() * img_part.nelement() # goodput
            received_data = received_goodput[fog_node][current_layer] # goodput
            total_data = (sent_data+received_data)/(1024*1024)
            if fog_node == 0:
                writer.writerow({'layer': current_layer,'transmission_time':(trans_user + trans_fog),'transmission_data':total_data,'computation_time':layer_comp_time})
            elif fog_node == 1:
                writer.writerow({'layer': current_layer,'transmission_time':(trans_user + trans_fog),'transmission_data':total_data,'computation_time':layer_comp_time})
            print("\tAfter ff: {}\n".format(img_part.size()))
            m = nn.ReLU()
            out = m(img_part).data > 0
            out = out.int()
            return out
            
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
    global transmission_time, total_computation_time
    models = {1:"AlexNet",2:"ResNet34",3:"VGG16"}
    m = input("Enter Model Number [1: AlexNet 2: ResNet34 3: VGG16]: ")
    pretrained_model = models[int(m)]
    # open a gRPC channel
    try:
        # write file in csv format
        #csv_file = open('/home/arnab/Desktop/DNN/gRPC/logs/gRPC_time_state_alexnet_cpu.csv','a')
        #fieldnames = ['computation_time', 'transmission_time']
        #writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        #writer.writeheader()
        csv_file1 = open('/home/arnab/Desktop/DNN/gRPC/logs/gRPC_time_state_alexnet_cpu_node_1.csv','a')
        csv_file2 = open('/home/arnab/Desktop/DNN/gRPC/logs/gRPC_time_state_alexnet_cpu_node_2.csv','a')
        fieldnames = ['layer', 'transmission_time', 'transmission_data', 'computation_time']
        writer1 = csv.DictWriter(csv_file1, fieldnames=fieldnames)
        writer1.writeheader()
        writer2 = csv.DictWriter(csv_file2, fieldnames=fieldnames)
        writer2.writeheader()

        channel1 = grpc.insecure_channel('192.168.0.106:50051')
        channel2 = grpc.insecure_channel('192.168.0.107:50051')

        # create a stub (client)
        stub1 = message_pb2_grpc.CalculatorStub(channel1)
        stub2 = message_pb2_grpc.CalculatorStub(channel2)

        current_layer = None
        img_part_list = []
        classify_list = []
        prev_input_units = 0
        total_transmission_time = 0
        final_out = None
        userObject = user.User(pretrained_model=pretrained_model)
        BATCH_SIZE = userObject.BATCH_SIZE
        worker = userObject.worker
        loader = userObject.image_loader()

        epoch = 0
        for img,level in loader:
            print(f"Epoch: {epoch}")
            for i,k in enumerate(userObject.kernel_filters):
                current_layer = i
                if i == 0:
                    input_ = img
                    channel = input_.shape[1]
                else:
                    input_ = final_out
                    channel = input_.shape[1]

                after_part = userObject.partition_algorithm("conv", input_, f=k)
                for j in range(len(after_part)):
                    img_part = adaptive_partitioning(input_, after_part[j]) # partitioned image
                    sent_goodput[j][current_layer] = img_part.element_size() * img_part.nelement() # goodput
                    msg = send_message("conv", img_part, after_part[j], BATCH_SIZE, current_layer, epoch, pretrained_model, 0)
                    out = message_pb2.Request(message=msg)
                    if j == 0:
                        r = stub1.Node(out)
                        img_part = received_message(r.message, current_layer, j, writer1) # goodput
                        img_part_list.append(img_part)
                    elif j == 1:
                        r = stub2.Node(out)
                        img_part = received_message(r.message, current_layer, j, writer2) # goodput
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
            avgpool = nn.AdaptiveAvgPool2d((6, 6))
            out = avgpool(final_out)
            input_ = out
            channel = input_.shape[1]
            current_layer += 1
            after_part = userObject.partition_algorithm("ff", input_, f=0)
            for j in range(len(after_part)):
                img_part = adaptive_partitioning(input_,after_part[j])
                flat_img = torch.flatten(img_part,1)

                sent_goodput[j][current_layer] = img_part.element_size() * img_part.nelement() # goodput
                msg = send_message("ff", img_part, after_part[j], BATCH_SIZE, current_layer, epoch, pretrained_model, prev_input_units)
                prev_input_units = flat_img.shape[1]
                out = message_pb2.Request(message=msg)
                if j == 0:
                    r = stub1.Node(out)
                    img_part = received_message(r.message, current_layer, j, writer1) # goodput
                    classify_list.append(img_part)
                elif j == 1:
                    r = stub2.Node(out)
                    img_part = received_message(r.message, current_layer, j, writer2) # goodput
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
                #break # first conv layer
            logging.info(f"Epoch: {epoch}")
            logging.info(f"Computation time (one node): {total_computation_time/worker}")
            print(f"Computation time (one node): {total_computation_time/worker}")
            logging.info(f"Transmission time (one node): {total_transmission_time}\n")
            print(f"Transmission time (one node): {total_transmission_time}")
            #writer.writerow({'computation_time':total_computation_time/worker,'transmission_time':total_transmission_time})
            transmission_time = 0
            total_computation_time = 0
            total_transmission_time = 0
            epoch += 1
            
            if epoch == 1:
                break
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