from models.experimental import attempt_load
from utils.datasets import LoadImages,letterbox
from utils.torch_utils import select_device,time_synchronized
from utils.general import non_max_suppression,increment_path,scale_coords
import torch
torch.set_grad_enabled(False)
import time
from pathlib import Path
from utils.plots import plot_one_box
import cv2
from numpy import random
import numpy as np
import gc
import uvicorn
import pandas as pd
import threading
import psutil
import base64
import time
import json
import os
from fastapi import FastAPI
from pydantic import BaseModel
from io import BytesIO,StringIO
from pandas import Series, DataFrame
import socket
import argparse
import sys

def yolov7_startup_event(weightsPath,input_size,iou_thres,conf_thres,stride,STRdevice="",classes=None,agnostic_nms=False):
    print('---------------------------------------------------------------------------')
    print('init yolov7 model')
    print(f'load weights from {weightsPath}')
    print('---------------------------------------------------------------------------')
    global yolov7Model
    yolov7Model = yolov7DetectClass( weightsPath, input_size, iou_thres, conf_thres,stride,STRdevice,classes,agnostic_nms)
    print('---------------------------------------------------------------------------')
    print('yolov7 model create ok')
    print('---------------------------------------------------------------------------')
    return yolov7Model

class yolov7DetectClass():
    def __init__(self, weightsPath, input_size, iou_thres, conf_thres,stride,STRdevice,classes,agnostic_nms):
        self.weightsPath = weightsPath
        self.inputsize = input_size
        self.iou = iou_thres
        self.score = conf_thres
        self.stride = stride
        self.device = select_device(STRdevice)        
        self.classes = classes
        self.agnostic_nms = agnostic_nms
        self.half = self.device.type != 'cpu'
        self.infer = attempt_load(weightsPath, map_location=self.device)
        if self.half:
            self.infer.half()
        self.names = self.infer.module.names if hasattr(self.infer, 'module') else self.infer.names
    def predict(self,img_cv2):
        result_list=[]
        # Padded resize
        img = letterbox(img_cv2, self.inputsize, stride=self.stride)[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        pred = self.infer(img, augment=False)[0]
        pred = non_max_suppression(pred, self.score, self.iou, classes=self.classes, agnostic=self.agnostic_nms)
        print(pred)
        for i,det in enumerate(pred):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img_cv2.shape).round()
        for *xyxy, conf, cls in reversed(det):    
            x1 = int(xyxy[0])
            y1 = int(xyxy[1])
            x2 = int(xyxy[2])
            y2 = int(xyxy[3])
            obj_dict = {
                "Object":self.names[int(cls)],
                "Score":int(conf*100),
                "X1":int(xyxy[0]),
                "Y1":int(xyxy[1]),
                "X2":int(xyxy[2]),
                "Y2":int(xyxy[3]),
            }
            result_list.append(obj_dict)
        return result_list
    def delete(self):
        self.infer.cpu()
        del self.infer
        gc.collect()
        torch.cuda.empty_cache()

# ---api input json format config---s
class objectDetection(BaseModel):
	Item:str
	Base64img:str
# ---api input json format config---e

def decodeBase64ToImg(imgB64String):
	jpg_original = base64.b64decode(imgB64String)
	jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
	img_cv2 = cv2.imdecode(jpg_as_np, flags=1)
	return img_cv2

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./checkpoints/pcakage-211208-v1', help='model path')
    parser.add_argument('--islocal', type=bool, default=False, help='whether to use loacl ip')
    opt = parser.parse_args()
    return opt.weights, opt.islocal

weightsPath, islocal = parse_opt()

app = FastAPI()

# startup event
@app.on_event("startup")
async def startup_event():
    ##init model
    input_size = 320
    iou_thres = 0.3
    conf_thres = 0.3
    stride = 32
    STRdevice = ""
    classes = None
    agnostic_nms = False
    

    print('---------------------------------------------------------------------------')
    print('init yolov7 model')
    print(f'load weights from {weightsPath}')
    print('---------------------------------------------------------------------------')
    global yolov7Model
    yolov7Model = yolov7DetectClass( weightsPath, input_size, iou_thres, conf_thres,stride,STRdevice,classes,agnostic_nms)
    print('---------------------------------------------------------------------------')
    print('yolov7 model create ok')
    print('---------------------------------------------------------------------------')

# shutdwon event
@app.on_event("shutdown")
def shutdown_event():
    with open("log.txt",mode="a") as log:
        log.write("Application shutdown\n")


@app.post("/predict")
async def predict( objectDetection:objectDetection):
    Item = objectDetection.Item
    Base64img = objectDetection.Base64img
    
    img_cv2 = decodeBase64ToImg(Base64img)

    # cv2.imwrite( Item , img_cv2)
    predictResult = yolov7Model.predict(img_cv2)
    
    reponse = dict()
    reponse['Status'] = 200
    reponse['Item'] = Item
    reponse['PredictModelName'] = 'temp'
    reponse['ObjectList'] = predictResult
    
    # reponse['ObjectList'] = [
            # {"Object": "obj1", "Score": 99.18, "X1": 48, "Y1": 133, "X2": 188, "Y2": 186},
            # {"Object": "obj2", "Score": 88.32, "X1": 48, "Y1": 133, "X2": 188, "Y2": 186}
    # ]
    
    return json.dumps(reponse)

def Get_local_ip(isLocal = False):
    if isLocal: return "127.0.0.1"
    try:
        csock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        csock.connect(('8.8.8.8', 80))
        (addr, port) = csock.getsockname()
        csock.close()
        return addr
    except socket.error:
        return "127.0.0.1"

if __name__ =="__main__":
    log_config = uvicorn.config.LOGGING_CONFIG
    log_config["formatters"]["access"]["fmt"] = "%(asctime)s - %(levelname)s - %(message)s"
    log_config["formatters"]["default"]["fmt"] = "%(asctime)s - %(levelname)s - %(message)s"
    host_id = Get_local_ip(islocal)
    print(f';start api on {host_id}:1234;')
    uvicorn.run("ObjectDetectionFastAPI:app",host=f'{host_id}',port=1234,log_config=log_config,reload=True)
'''
if __name__ == "__main__":
    model = yolov7_startup_event("./yolov7-tiny_20220919_CP39J6736035A_best.pt",640,0.45,0.25,32,"",None,False)
    # model2 = yolov7_startup_event("./yolov7.pt",640,0.45,0.25,32,"",None,False)
    source = "/root/backup/Accton/S1_0001939.jpg"
    predict_img = cv2.imread(source)
    result = model.predict(predict_img)
    print(result)
    # result = model2.predict(predict_img)
    # logger.info(result)
    # Model_dict = dict()
    # for i in range(50):
    #     logger.info(f"No.{i}")
    #     Model_dict[f"Test-{i}"] = yolov7_startup_event("./yolov7-tiny_20220919_CP39J6736035A_best.pt",640,0.45,0.25,32,"",None,False)
    # logger.info(len(Model_dict))
    # logger.info(Model_dict)
'''