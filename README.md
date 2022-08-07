# Eyesight Protector
![](eyesight_protector.jpeg)
## Abstract
- My cousin starts remote classes due to pandemic.  
- However, he could not foucs in class and stay too close on the screen.
- Thus these activaties inspired me to create a program to prevent eyesight loss.


## Introduction
Due to pandemic children takes remote classes without acknowledging impact of screen on their eyesight. This program helps them to keep a distance from the screen.

This project utilise dlib open source library to create interactive program that warn to clients with a message if they are too close to the webcam. 

Server client is created by gRPC framework to communicate between client and server

- Scan webcam image per second, and send frame to the server.
- Once server receive the frame, it detects whether client is on appropriate distance from the webcam by using ai module. 
- If client is too close on screen or abscent, text will be written on the frame and visible to the client.

## Installation
1. Prerequisties 
    
    Follow the command below.
    
- Optional: Recommanded to use Anaconda Virtual Environment

    ```bash
    conda create -n <name of environment> python=3.8
    conda activate <name>
    pip install -r requirements.txt
    ```
- Without anaconda
    
    ```bash
    pip install -r requirements.txt
    ```

2. Build Faceboxes
    ```bash
    cd utils
    python3 build.py build_ext --inplace
    cd ..
    ```

3. Build gRPC protobuf

    ```bash
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
