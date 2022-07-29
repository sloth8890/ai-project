from urllib import response
import grpc
import proto_sample_pb2, proto_sample_pb2_grpc

import os
import glob
import cv2
import argparse

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
    
    parser.add_argument('--thres', type=int, default=150, help='threashold value for to oclose')
    parser.add_argument('--debug', action='store_true', default=False, help='debug mode (default: False)')
    return parser.parse_args()
    
    
    
def main():
    args = opt()    

    webcam = cv2.VideoCapture(0)

    cnt = 0
    
    while True:
        # if cnt == 1818:
        _ , frame_bgr = webcam.read()
        if frame_bgr is None:
            print('can not connect to the webcam!')
            break
        
        if cnt % 30 == 0:
            cnt = 0
            
            response = request(args.ip, args.port, frame_bgr)
            if response.is_abscent: 
                #warn
                if args.debug:
                    frame_bgr = cv2.putText(frame_bgr, 'client is abscent', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                else:
                    os.system('say {}'.format('client is abscent'))
            else:
                if response.distance < args.thres:
                    if args.debug:
                        frame_bgr = cv2.putText(frame_bgr, 'client too close to the screen', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                    else:
                        os.system('say {}'.format('client is too close to the screen'))

        cv2.imshow('window', frame_bgr)
        key = cv2.waitKey(33)
        if key == ord('q'):
            break
        cnt += 1 
    





if __name__ == '__main__':
    main()