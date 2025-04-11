import numpy as np
import sys
from dx_engine import InferenceEngine

import threading
import queue
from threading import Thread

import cv2
import argparse
import json
import torch
import torchvision
from ultralytics.utils import ops

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--model', type=str, help='Path to the model file')
arg_parser.add_argument('--video', type=str, help='Path to the video file')
arg_parser.add_argument('--config', type=str, help='Path to the config file')
args = arg_parser.parse_args()
model_path = args.model
video_path = args.video
config_path = args.config



def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def all_decode(ie_outputs, layer_config):
    ''' slice outputs'''
    outputs = []
    outputs.append(ie_outputs[0][...,:255])
    outputs.append(ie_outputs[1][...,:255])
    outputs.append(ie_outputs[2][...,:255])
    
    decoded_tensor = []
    
    for i, output in enumerate(outputs):
        output[...,4] = sigmoid(output[...,4]) # obj confidence
        for l in range(len(layer_config[i]["anchor_width"])):
            layer = layer_config[i]
            stride = layer["stride"]
            grid_size = output.shape[2]
            meshgrid_x = np.arange(0, grid_size)
            meshgrid_y = np.arange(0, grid_size)
            grid = np.stack([np.meshgrid(meshgrid_y, meshgrid_x)], axis=-1)[...,0]
            cxcy = output[...,(l*85)+0:(l*85)+2]
            wh = output[...,(l*85)+2:(l*85)+4]
            cxcy[...,0] = (sigmoid(cxcy[...,0]) * 2 - 0.5 + grid[0]) * stride
            cxcy[...,1] = (sigmoid(cxcy[...,1]) * 2 - 0.5 + grid[1]) * stride
            wh[...,0] = ((sigmoid(wh[...,0]) * 2) ** 2) * layer["anchor_width"][l]
            wh[...,1] = ((sigmoid(wh[...,1]) * 2) ** 2) * layer["anchor_height"][l]
            decoded_tensor.append(output[...,(l*85)+0:(l*85)+85].reshape(-1, 85))
            
    decoded_output = np.concatenate(decoded_tensor, axis=0)
    
    return decoded_output

def post_process(decoded_tensor, image_input, i, config):
    model_path = config["model"]["path"]
    classes = config["output"]["classes"]
    score_threshold = config["model"]["param"]["score_threshold"]
    iou_threshold = config["model"]["param"]["iou_threshold"]
    layers = config["model"]["param"]["layer"]

    ''' post Processing '''
    x = torch.Tensor(decoded_tensor)
    x = x[x[..., 4] > score_threshold]
    box = ops.xywh2xyxy(x[:, :4])
    x[:, 5:] *= x[:, 4:5]
    conf, j = x[:, 5:].max(1, keepdims=True)
    x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > score_threshold]
    x = x[x[:, 4].argsort(descending=True)]
    x = x[torchvision.ops.nms(x[:,:4], x[:, 4], iou_threshold)]
    x = x[x[:,4] > 0]
    print("[Result] Detected {} Boxes.".format(len(x)))
    ''' save result and print detected info '''
    image = cv2.cvtColor(image_input, cv2.COLOR_RGB2BGR)
    colors = np.random.randint(0, 256, [80, 3], np.uint8).tolist()
    for idx, r in enumerate(x.numpy()):
        
        pt1, pt2, conf, label = r[0:2].astype(int), r[2:4].astype(int), r[4], r[5].astype(int)
        print("[{}] conf, classID, x1, y1, x2, y2, : {:.4f}, {}({}), {}, {}, {}, {}"
              .format(idx, conf, classes[label], label, pt1[0], pt1[1], pt2[0], pt2[1]))
        image = cv2.rectangle(image, pt1, pt2, colors[label], 2)
    cv2.imwrite(f"{i}.jpg", image)
    print(f"save file : {i}.jpg ")    

if __name__ == "__main__":

    f = open(config_path, "r")
    json_config = json.load(f)
    # create inference engine instance with model
    ie = InferenceEngine(model_path)
    cap = cv2.VideoCapture(video_path)
    # register call back function
    loop_count = 1
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # run inference
            output = ie.Run(frame)
            layers = json_config["model"]["param"]["layer"]
            decoded_a = all_decode(output, layers)
            post_process(decoded_a, frame, loop_count, json_config) 
            # increment loop count
            loop_count += 1
            print(f"Loop count: {loop_count}")
        else:
            print("End of video stream: Breaking")
            break
    print("out of while loop")
    f.close()
    exit(0)