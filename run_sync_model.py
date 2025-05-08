import numpy as np
import sys
from dx_engine import InferenceEngine

import threading
import queue
from threading import Thread
import os

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

def letter_box(image_src, new_shape=(512, 512), fill_color=(114, 114, 114), format=None):
    src_shape = image_src.shape[:2] # height, width
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / src_shape[0], new_shape[1] / src_shape[1])

    ratio = r, r
    new_unpad = int(round(src_shape[1] * r)), int(round(src_shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

    dw /= 2
    dh /= 2

    if src_shape[::-1] != new_unpad:
        image_src = cv2.resize(image_src, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    image_new = cv2.copyMakeBorder(image_src, top, bottom, left, right, cv2.BORDER_CONSTANT, value=fill_color)  # add border
    if format is not None:
        image_new = cv2.cvtColor(image_new, format)

    return image_new, ratio, (dw, dh)

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
    # 1. Extract Configuration Parameters:
    #    - Retrieves necessary settings from the 'config' dictionary.
    #    - model_path and layers seem unused within this specific function.
    model_path = config["model"]["path"] # Unused here
    classes = config["output"]["classes"] # List of class names (e.g., ['person', 'car', ...])
    score_threshold = config["model"]["param"]["score_threshold"] # Minimum confidence score to consider a detection
    iou_threshold = config["model"]["param"]["iou_threshold"] # Overlap threshold for NMS
    layers = config["model"]["param"]["layer"] # Unused here

    ''' post Processing '''
    # 2. Convert to PyTorch Tensor:
    #    - Converts the NumPy array 'decoded_tensor' (output from all_decode)
    #      into a PyTorch tensor for efficient processing with PyTorch/Torchvision functions.
    #    - decoded_tensor shape is likely (N, 85), where N is the total number of potential
    #      detections across all grid cells and anchors, and 85 = cx, cy, w, h, obj_conf, 80 class_scores.
    x = torch.Tensor(decoded_tensor)

    # 3. Initial Confidence Filtering:
    #    - Filters the tensor 'x', keeping only rows (detections) where the object confidence
    #      score (at index 4) is greater than the specified 'score_threshold'.
    #    - This removes low-confidence predictions early on.
    x = x[x[..., 4] > score_threshold]

    # 4. Convert Box Format (xywh -> xyxy):
    #    - Takes the bounding box coordinates [cx, cy, w, h] (center x, center y, width, height)
    #      from the first 4 columns of 'x'.
    #    - Converts them to [x1, y1, x2, y2] format (top-left x, top-left y, bottom-right x, bottom-right y).
    #    - This uses a utility function `ops.xywh2xyxy` likely from the 'ultralytics' library.
    box = ops.xywh2xyxy(x[:, :4])

    # 5. Calculate Class Confidences:
    #    - Multiplies the class scores (columns 5 onwards) by the object confidence score (column 4).
    #    - This gives the final confidence for each class: P(Class_i|Object) * P(Object).
    x[:, 5:] *= x[:, 4:5]

    # 6. Find Best Class and Score per Box:
    #    - For each detection, finds the maximum confidence score among all class scores (columns 5 onwards).
    #    - 'conf' will store the maximum confidence score.
    #    - 'j' will store the index (the class ID) of that maximum score.
    conf, j = x[:, 5:].max(1, keepdims=True)

    # 7. Combine and Filter Again:
    #    - Concatenates the converted boxes ('box'), the maximum confidence ('conf'),
    #      and the class ID ('j') into a new tensor. Shape: (N_filtered, 6) -> [x1, y1, x2, y2, final_conf, class_id].
    #    - Filters this tensor *again* based on the final class confidence ('conf'), ensuring it's above the 'score_threshold'.
    #      (This might seem redundant with step 3, but it ensures the *final* class confidence meets the threshold).
    x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > score_threshold]

    # 8. Sort by Confidence:
    #    - Sorts the remaining detections in descending order based on their confidence score (column 4).
    #    - NMS algorithms often work best with sorted inputs.
    x = x[x[:, 4].argsort(descending=True)]

    # 9. Apply Non-Maximum Suppression (NMS):
    #    - Uses `torchvision.ops.nms` to remove overlapping bounding boxes for the same object.
    #    - It takes the boxes ([x1, y1, x2, y2]), the confidence scores, and the 'iou_threshold'.
    #    - If two boxes have an Intersection over Union (IoU) greater than 'iou_threshold',
    #      the one with the lower confidence score is suppressed (removed).
    #    - 'x' now contains only the indices of the boxes kept after NMS.
    x = x[torchvision.ops.nms(x[:,:4], x[:, 4], iou_threshold)]

    # 10. Final Check (Optional but safe):
    #     - Ensures that all remaining boxes still have a positive confidence score.
    x = x[x[:,4] > 0]

    # 11. Print Detection Count:
    print("[Result] Detected {} Boxes.".format(len(x)))

    ''' save result and print detected info '''
    # 12. Prepare Image for Drawing:
    #     - Converts the input image (assumed RGB) to BGR format, which OpenCV uses for drawing and saving.
    image = cv2.cvtColor(image_input, cv2.COLOR_RGB2BGR)

    # 13. Generate Colors:
    #     - Creates a list of random colors, one for each potential class (hardcoded to 80 here).
    #     - It's generally better to generate this once outside the loop or use a fixed color map.
    colors = np.random.randint(0, 256, [80, 3], np.uint8).tolist()

    # 14. Draw Bounding Boxes and Print Info:
    #     - Iterates through the final detections in the tensor 'x' (converted back to NumPy).
    for idx, r in enumerate(x.numpy()):
        # Extract box coordinates, confidence, and class label.
        pt1, pt2, conf, label = r[0:2].astype(int), r[2:4].astype(int), r[4], r[5].astype(int)
        # Print detailed information about the detection.
        print("[{}] conf, classID, x1, y1, x2, y2, : {:.4f}, {}({}), {}, {}, {}, {}"
              .format(idx, conf, classes[label], label, pt1[0], pt1[1], pt2[0], pt2[1]))
        # Draw the rectangle on the image using the class-specific color.
        image = cv2.rectangle(image, pt1, pt2, colors[label], 2)

    # 15. Save Output Image:
    #     - Saves the image with the drawn bounding boxes to a file named using the counter 'i'.
    cv2.imwrite(f"{i}.jpg", image)
    print(f"save file : {i}.jpg ")


if __name__ == "__main__":

    f = open(config_path, "r")
    json_config = json.load(f)
    # create inference engine instance with model
    ie = InferenceEngine(model_path)
    input_size = np.sqrt(ie.input_size() / 3)
    cap = cv2.VideoCapture(video_path)
    # register call back function
    loop_count = 1
    model_path = json_config["model"]["path"]
    classes = json_config["output"]["classes"]
    score_threshold = json_config["model"]["param"]["score_threshold"]
    iou_threshold = json_config["model"]["param"]["iou_threshold"]
    layers = json_config["model"]["param"]["layer"]   
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # run inference
            image_input, _, _ = letter_box(frame, new_shape=(int(input_size), int(input_size)), fill_color=(114, 114, 114), format=cv2.COLOR_BGR2RGB)
            ie_output = ie.Run(image_input)
            layers = json_config["model"]["param"]["layer"]
            decoded_tensor = []
            if ie.output_dtype()[0] == "BBOX":
                decoded_tensor = ppu_decode(ie_output, layers)
            elif len(ie_output) > 1:
                cpu_model_path = os.path.join(os.path.split(model_path)[0], "cpu_0.onnx")
                if os.path.exists(cpu_model_path):
                    decoded_tensor = onnx_decode(ie_output, cpu_model_path)
                else:
                    decoded_tensor = all_decode(ie_output, layers)
            else:
                decoded_tensor = ie_output[0]
            print("decoding output Done! ")

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
                print(f"pt1: {pt1}, pt2: {pt2}, conf: {conf}, label: {label}")
            cv2.imwrite(f"{loop_count}.jpg", image)    
            print(f"save file : {loop_count}.jpg ")
            loop_count += 1
###            decoded_output = all_decode(output, layers)
###            post_process(decoded_output, frame, loop_count, json_config) 
###            # increment loop count
###            loop_count += 1
###            print(f"Loop count: {loop_count}")
            
        else:
            print("End of video stream: Breaking")
            break
    print("out of while loop")
    f.close()
    exit(0)