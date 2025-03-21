from dx_engine import InferenceEngine
import sys
import cv2
import queue
import threading

result_queue = queue.Queue()

# Get model path from an argument
model_path = sys.argv[1]
video_path = sys.argv[2]

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

def wait_for_requests(ie):
    while not result_queue.empty():
        try:
            req_id = result_queue.get(timeout=5)
            print(f"Waiting for request {req_id}")
            outputs = ie.Wait(req_id)
            print(f"Outputs: {len(outputs)}")
            print(f"Outputs: {outputs[0].shape}")
            outputs_checker = []
            for output in outputs:
                outputs_checker.append(output[...,:255])
            print(f"Request {req_id} completed")
            #print(f"Output shape: {len(outputs)}")
            #print(f"Output shape 0: {outputs[0].shape}")
            #print(f"Output shape 1: {outputs[1].shape}")
            #print(f"Output shape 2: {outputs[2].shape}")
        except queue.Empty:
            print("Queue is empty")
            break

if __name__ == "__main__":
    ie = InferenceEngine(model_path)
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            req_id = ie.RunAsync(frame,None)
            print(f"Inference request ID: {req_id} submitted")
            result_queue.put(req_id)
        else:
            break 

    wait_thread = threading.Thread(target=wait_for_requests, args=(ie,))
    wait_thread.start()
    result_queue.join()
    wait_thread.join()
    
    print(ie)
