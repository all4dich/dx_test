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
    # 1. Initial Slicing:
    #    - It assumes ie_outputs contains results from multiple detection layers (often 3 in YOLO).
    #    - It takes the first three output tensors from the inference engine.
    #    - For each tensor, it keeps only the first 255 channels along the last dimension ([..., :255]).
    #      This number (255) often corresponds to (num_classes + 5) * num_anchors per layer.
    #      For example, with 80 classes and 3 anchors: (80 + 4 box coords + 1 obj confidence) * 3 = 85 * 3 = 255.
    outputs = []
    outputs.append(ie_outputs[0][...,:255])
    outputs.append(ie_outputs[1][...,:255])
    outputs.append(ie_outputs[2][...,:255])

    # This list will store the decoded bounding box information from all layers and anchors.
    decoded_tensor = []

    # 2. Iterate Through Each Output Layer:
    #    - 'i' is the index of the layer (0, 1, 2).
    #    - 'output' is the sliced tensor for that layer.
    for i, output in enumerate(outputs):
        # 3. Apply Sigmoid to Object Confidence:
        #    - The 5th channel (index 4) usually represents the object confidence score.
        #    - The sigmoid function squashes this value between 0 and 1, making it a probability.
        #    - This modification happens *in-place* on the 'output' array.
        output[...,4] = sigmoid(output[...,4]) # obj confidence

        # 4. Iterate Through Anchors for the Current Layer:
        #    - 'l' is the index of the anchor box for this layer (e.g., 0, 1, 2 if there are 3 anchors).
        #    - layer_config[i]["anchor_width"] gives the number of anchors for layer 'i'.
        for l in range(len(layer_config[i]["anchor_width"])):
            # Get configuration specific to this layer 'i'.
            layer = layer_config[i]
            # Get the stride for this layer (how much the feature map is downscaled from the input).
            stride = layer["stride"]
            # Get the spatial dimension (width/height) of the feature map for this layer.
            grid_size = output.shape[2]

            # 5. Create Coordinate Grid:
            #    - Creates meshgrids representing the x and y coordinates for each cell in the feature map grid.
            #    - 'grid' will have shape (grid_size, grid_size, 2), storing (x, y) for each cell.
            meshgrid_x = np.arange(0, grid_size)
            meshgrid_y = np.arange(0, grid_size)
            # Note: The np.stack and [...,0] seem a bit convoluted way to create the grid,
            #       np.meshgrid directly gives the desired structure usually.
            #       Let's assume it correctly produces a grid where grid[0] is x-coords and grid[1] is y-coords.
            grid = np.stack([np.meshgrid(meshgrid_y, meshgrid_x)], axis=-1)[...,0]

            # 6. Extract Raw Box Predictions for Anchor 'l':
            #    - Extracts the raw center coordinates (cx, cy) and width/height (w, h)
            #      for the current anchor 'l' from the 'output' tensor.
            #    - Assumes each anchor's data occupies 85 channels (cx, cy, w, h, conf, 80 classes).
            cxcy = output[...,(l*85)+0:(l*85)+2]
            wh = output[...,(l*85)+2:(l*85)+4]

            # 7. Decode Center Coordinates (cx, cy):
            #    - Applies the YOLO decoding formula for center coordinates:
            #      - sigmoid(raw_cx) * 2 - 0.5: Scales the prediction relative to the grid cell center.
            #      - + grid[0] / grid[1]: Adds the grid cell's top-left corner coordinate.
            #      - * stride: Scales the coordinate from the feature map scale to the input image scale.
            #    - This calculation modifies the 'cxcy' slice (and thus 'output') *in-place*.
            cxcy[...,0] = (sigmoid(cxcy[...,0]) * 2 - 0.5 + grid[0]) * stride # Decode cx
            cxcy[...,1] = (sigmoid(cxcy[...,1]) * 2 - 0.5 + grid[1]) * stride # Decode cy

            # 8. Decode Width and Height (w, h):
            #    - Applies the YOLO decoding formula for width/height:
            #      - (sigmoid(raw_wh) * 2) ** 2: Scales the prediction relative to the anchor size.
            #      - * layer["anchor_width/height"][l]: Multiplies by the predefined anchor dimension for this anchor 'l'.
            #    - This calculation modifies the 'wh' slice (and thus 'output') *in-place*.
            wh[...,0] = ((sigmoid(wh[...,0]) * 2) ** 2) * layer["anchor_width"][l]  # Decode w
            wh[...,1] = ((sigmoid(wh[...,1]) * 2) ** 2) * layer["anchor_height"][l] # Decode h

            # 9. Store Decoded Anchor Data:
            #    - Takes the entire block of 85 channels for the current anchor 'l'.
            #    - Since cx, cy, w, h, and confidence were modified in-place, this block now contains
            #      decoded coordinates, confidence, and raw class scores.
            #    - Reshapes it into a 2D array where each row is a potential detection (-1 rows, 85 columns).
            #    - Appends this array to the `decoded_tensor` list.
            decoded_tensor.append(output[...,(l*85)+0:(l*85)+85].reshape(-1, 85))

    # 10. Concatenate All Decoded Tensors:
    #     - Combines all the decoded tensors (from all layers and all anchors) into a single large NumPy array.
    #     - The concatenation happens along axis 0 (stacking the rows).
    #     - The final array contains all potential detections from the model.
    decoded_output = np.concatenate(decoded_tensor, axis=0)

    # 11. Return Result:
    #     - Returns the single NumPy array containing decoded bounding boxes, confidences, and class scores.
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