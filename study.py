# from ast import parse
# import grpc
# import proto_sample_pb2, proto_sample_pb2_grpc
# from concurrent import futures

# import os
# import cv2
# import numpy as np
# import argparse

# from datetime import datetime
# from pytz import timezone
# import logging

# [1] decorator 
'''
    함수를 인자로 받아서, 함수를 리턴하는 기능
'''
import time
def wrapper(func):
    def inner(*args, **kwargs):
        start = time.perf_counter()

        ret = func(*args, **kwargs)

        finish = time.perf_counter()
        print('Tile elpased: {}'.format(finish-start))
        return ret
    return inner


@wrapper
def func1():
    time.sleep(1.5)

@wrapper
def func2(a, b, c):
    return a+b+c

@wrapper
def func3(a, b, c=10, d=5):
    return a+b+c+d

print( func3(1, 1, c=5) )


from dataclasses import dataclass

class A:
    def __init__(self, a, b):
        self._a = a 
        self._b = b
    
    @staticmethod
    def function1():
        print("this is static method")
        
    @property
    def a(self):
        return self._a
    
    @a.setter
    def a(self, value):
        self._a = value


def main():
    obj1 = A(2,4)
    obj1.function1()
    print(obj1.a)
    obj1.a = 5
    print(obj1.a)
    pass

if __name__ == '__main__':
    main()