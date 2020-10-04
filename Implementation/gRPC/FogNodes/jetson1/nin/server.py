import grpc
from concurrent import futures
import time

import message_pb2
import message_pb2_grpc

import node
#import node_GPU

class CalculatorService(message_pb2_grpc.CalculatorServicer):
    def Node(self, request, context):
        response = message_pb2.Response()
        response.message = node.fog_node(request.message)
        return response
    """
    def Node_GPU(self, request, context):
        response = message_pb2.Response()
        response.message = node_GPU.fog_node(request.message)
        return response
    """
    
MAX_MESSAGE_LENGTH = 104857600 # 100MB
executor = futures.ThreadPoolExecutor()
server = grpc.server(executor, 
                    options=[
                    ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
                    ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),],)

message_pb2_grpc.add_CalculatorServicer_to_server(CalculatorService(), server)

print('Starting server. Listening on port 50051')
server.add_insecure_port('[::]:50051')
server.start()

try:
    while True:
        time.sleep(86400)
except KeyboardInterrupt:
    server.stop(0)