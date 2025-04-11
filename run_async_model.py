import numpy as np
import sys
from dx_engine import InferenceEngine

import threading
import queue
from threading import Thread

import cv2

model_path = sys.argv[1]
video_path = sys.argv[2]


q = queue.Queue()
gLoopCount = 0

lock = threading.Lock()

def onInferenceCallbackFunc(outputs, user_arg):
    #with lock:
    #    print(f"onInferenceCallbackFunc: {len(outputs)} ")
    #    gLoopCount += 1
    #    loop_count = user_arg.value
    #    print(f"onInferenceCallbackFunc: {gLoopCount} {loop_count}")
    #    if ( gLoopCount == loop_count ) :
    #        print("Complete Callback")
    #p        q.put(0)
    print(f"onInferenceCallbackFunc: {len(outputs)} ")
    return


if __name__ == "__main__":
    DEFAULT_LOOP_COUNT = 1
    loop_count = DEFAULT_LOOP_COUNT

    # create inference engine instance with model
    ie = InferenceEngine(model_path)
    cap = cv2.VideoCapture(video_path)
    # register call back function
    ie.RegisterCallBack(onInferenceCallbackFunc)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # run inference
            req_id = ie.RunAsync(frame, [loop_count])
            print(f"Inference request ID: {req_id} submitted")
            # increment loop count
            loop_count += 1
        else:
            print("End of video stream: Breaking")
            break
    print("out of while loop")
    exit(0)