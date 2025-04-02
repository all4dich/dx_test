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
            print(f"Request {req_id} completed")
            print(f"Output shape: {len(outputs)}")
            #print(f"Output shape: {outputs[0]}")
            print(f"Output shape: {outputs[0].shape}")
            print(f"Output shape: {outputs[1].shape}")
            print(f"Output shape 2: {outputs[2].shape}")
            cv2.imshow("result", outputs[2])
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except queue.Empty:
            print("Queue is empty")
            break

        #status = ie.Wait(req_id)
        #print(f"Request {req_id}dfcompleted with status {status}")
        #if status == 0:
        #    result = ie.GetResult(req_id)
        #    print(f"Result shape: {result.shape}")
        #    cv2.imshow("result", result)
        #    if cv2.waitKey(1) & 0xFF == ord('q'):
        #        break
        #else:
        #    print(f"Request {req_id} failed with status {status}")
        #    #break

if __name__ == "__main__":
    engine = InferenceEngine(model_path)
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            #frame, ratio, pad = letter_box(frame, new_shape=(512, 512), fill_color=(114, 114, 114), format=None)
            req_id = engine.RunAsync(frame, frame)
            print(f"Inference request ID: {req_id} submitted")
            result_queue.put(req_id)
            #cv2.imshow("result", result)
            #cv2.imshow("frame", frame)
            #print(frame.shape)
            #if cv2.waitKey(1) & 0xFF == ord('q'):
            #    break
        else:
            break 

    wait_thread = threading.Thread(target=wait_for_requests, args=(engine,))
    wait_thread.start()
    result_queue.join
    wait_thread.join()
    
    print(engine)
