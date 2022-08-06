# Eyesight Protector
![](eyesight_protector.jpeg)
## Introduction

Due to pandemic children takes remote classes without acknowledging impact of screen on their eyesight. This program helps them to keep a distance from the screen.

This ai-project utilise dlib open source library to create interactive program that warn to clients with a message if they are too close to the webcam. 

Server client is created by gRPC framework to communicate between client and server


## Installation
1. Prerequisties 
    Follow the command below.
    1. Optional: Recommanded to use Anaconda Virtual Environment
        ```
        conda create -n <name of environment> python=3.8
        conda activate <name>
        pip install -r requirements.txt
        ```
    2. without anaconda


2. Build Faceboxes
```
cd utils
python3 build.py build_ext --inplace
cd ..
```

3. Build gRPC protobuf
```
sh build_grpc_pb.sh
```

## How to Run
To activate server
```
python server.py --port <port number>
```
To activate client
```
python client.py --ip <server ip> --port <port number>
```

## Acknowledgement
[Dlib libary](https://github.com/davisking/dlib)
