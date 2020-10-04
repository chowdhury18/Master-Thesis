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
    
server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

message_pb2_grpc.add_CalculatorServicer_to_server(CalculatorService(), server)

print('Starting server. Listening on port 50051')
server.add_insecure_port('[::]:50051')
server.start()

try:
    while True:
        time.sleep(86400)
except KeyboardInterrupt:
    server.stop(0)