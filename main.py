from tkinter import ON
from urllib import response
import grpc
import proto_sample_pb2, proto_sample_pb2_grpc

import os
import glob
import cv2
import argparse

class OnlineClassMonitoring:
    def __init__(self, file_path):
        #TODO save logging
        self.file_path = file_path
        
        #TODO hueristically determined
        self.thres_max = 100
        self.thres_min = 0
        
        self.message1 = 'leaving'
        self.message2 = 'please away your head from monitor'
        
    #magic method
    def __call__(self, distance):
        if distance > self.thres_max:
            os.system('say {}'.format(self.message1))
        elif distance < self.thres_min:
            os.system('say {}'.format(self.message2))
            pass
    # def __len__(self):
    #     return 200
    # def __repr__(self):
    #     return 'this is online monitoring system'

def request(ip, port, frame):
    print('request: client -> server : {}{}'.format(ip, port))
    with grpc.insecure_channel(ip+port) as channel:
        stub = proto_sample_pb2_grpc.RemoteControlServiceStub(channel)
        response = stub.process(
            proto_sample_pb2.UserRequest(
                img_bytes=bytes(frame),
                width=frame.shape[1],
                height=frame.shape[0],
                channel=frame.shape[2]
            )
        )
        return response


def opt():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--port', type=str, default='50051', help='port number (defualt: 50051)')
    parser.add_argument('--ip', type=str, default='localhost:', help='ip address (default: localhost:')
    return parser.parse_args()
    
    
def main():
    args = opt()    
    monitor = OnlineClassMonitoring("./")
    
    #webcam obj
    webcam = cv2.VideoCapture(0)
    webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    #! debug
    distance = 50
    while True:
        _ , frame = webcam.read()
        frame = cv2.flip(frame, 1)
        
        # TODO implement
        # response = request(frame, ip, port)
        # distance = response.distance
        
        #! Fake signal
        monitor(distance)
        
        
        # image display
        cv2.imshow('window', frame) 
        key = cv2.waitKey(33) # 33ms
        
        if key == ord('q'):
            break
        elif key == ord('c'):
            # cv2.imwrite('screen_shot.png', frame) #image write
            # print('Image has been saved!')
            distance = -5
        elif key == ord('f'):
            distance = 105
        else:
            distance = 50




if __name__ == '__main__':
    main()