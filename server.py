import os
import argparse
from datetime import datetime
import logging
from concurrent import futures

import cv2
import grpc
import numpy as np
from pytz import timezone
import proto_sample_pb2, proto_sample_pb2_grpc

from gaze_wrapper import GazeTracker

class MyAI_Model(proto_sample_pb2_grpc.RemoteControlService):
    def __init__(self, args):
        super(MyAI_Model, self).__init__()
        
        # logging
        if not os.path.exists(args.logdir):
            os.makedirs(args.logdir)
            
        self.logger = logging.getLogger(__class__.__name__)
        self.logger.setLevel(logging.INFO)
        
        self.stream_handler = logging.StreamHandler()
        self.file_handler = logging.FileHandler(os.path.join(args.logdir, __class__.__name__+'.log'))
        self.logger.addHandler(self.stream_handler)
        self.logger.addHandler(self.file_handler)
        
        self.fmt = "%Y-%m-%d %H:%M:%S"
        self.logger.info('{} - MyAI_Model Initialization Finished!'.format(
            datetime.now(timezone('Australia/Sydney')).strftime(self.fmt)))
        self.model = GazeTracker()
        
    def process(self, input, context):
        # recover input img to 2d
        print('debug mode on')
        image = np.array(list(input.img_bytes))
        image = image.reshape((input.height, input.width, input.channel))
        image = np.array(image, dtype=np.uint8)
        
        self.height, self.width, _ = image.shape
        
        print("self width is ", self.width)
        print("self height is ", self.height)
        
        inference_result = None
        reply_data = proto_sample_pb2.ServerReply()
        reply_data.distance = -1
        reply_data.is_abscent = True


        print('debug mode ---')
        try:
            print('debug mode +++')
            inference_result = self.model(image)
        except Exception as e:
            self.logger.info('{0} - Error Occued: {1}'.format(datetime.now(timezone('Australia/Sydney')).strftime(self.fmt), repr(e)))
            pass
        except KeyboardInterrupt:
            print('--------------> Finish Model Running')

        if inference_result:
            print('debug mode zzz')
            print(inference_result.face_distance)
            if inference_result.face_distance == None:
                reply_data.distance = -1
            else:
                reply_data.distance = inference_result.face_distance
            print('debug mode ooo')
            if reply_data.distance != -1:
                reply_data.is_abscent = False
        print('debug mode xxx')
        
        self.logger.info('{} - Reply to request'.format(datetime.now(timezone('Australia/Sydney')).strftime(self.fmt)))
        print('debug mode off')
        return reply_data
    
    
    
def opt():
    parser = argparse.ArgumentParser()
    #? grpc option
    parser.add_argument('--port', type=str, default='16023', help='Port number, default: 16023')
    parser.add_argument('--num_worker', type=int, default=8, help='the number of threads. default: 8')
    parser.add_argument('--logdir', type=str, default='./service_log', help='directory where the logging file is saved.')
    return parser.parse_args()
    

def serve():
    args = opt()
    
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=args.num_worker))
    proto_sample_pb2_grpc.add_RemoteControlServiceServicer_to_server(MyAI_Model(args), server)
    
    server.add_insecure_port('[::]:%s' % (args.port))
    server.start()
    server.wait_for_termination()
    


if __name__ == '__main__':
    serve()