# encoding:utf-8
import os
import time
import cv2

READFILE = "/home/faliks/Desktop/fifoCppToPython.tmp"
WRITEFILE = "/home/faliks/Desktop/fifoPythonToCpp.tmp"

while not os.path.exists(READFILE) or not os.path.exists(WRITEFILE):
    print("no file")
    time.sleep(1)

rf = os.open(READFILE, os.O_RDWR | os.O_NONBLOCK)
wf = os.open(WRITEFILE, os.O_RDWR | os.O_NONBLOCK)


def read():
    try:
        msg = os.read(rf, 1)
    except Exception as e:
        msg = "0"
    if msg != "0":
        print(msg)
    return msg


def write(s):
    os.write(wf, s.encode())


def close():
    os.close(rf)
    os.close(wf)

if __name__ == '__main__':
    while True:
        print('write')
        write('r')
        time.sleep(1000)