
from ctypes import *
import pathlib
import cv2
import numpy as np
import vart
import xir
import os
import sys
import threading
import queue
import time
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

global isCapturing
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def streamCapture(path, queueIn,input_scale):
    global isCapturing
    print('Capture stream from {}'.format(path))
    cap = cv2.VideoCapture(path)
    frame_id = 0
    while cap.isOpened():
        ret, frame = cap.read()
        converted = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

        frame = Image.fromarray(converted)

        frame = frame.resize((224,224),Image.NEAREST)

        isCapturing = True

        if not ret:
            isCapturing = False
            break
        queueIn.put((frame_id, frame))
        frame_id = frame_id + 1

    cap.release()


def outputStream(queueOut):
    global isCapturing
    count = 0
    width, height = 224, 224

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('output.avi',fourcc, 20.0, (224,224))

    tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()
        ])

    count = 0

    while isCapturing:
        img_output = 'output/' + str(count) + '.jpg'
        frame_id, (original_frame, pred_labels) = queueOut.get()
        pred_labels = pred_labels.astype('float32')
        output = pred_labels
        torch_array = torch.from_numpy(pred_labels)
        torch_array = torch_array.unsqueeze(0)
        torch_array = torch_array.permute(0,3,1,2)
        probs = F.softmax(torch_array, dim=1)[0]

        full_mask = tf(probs.cpu()).squeeze()

        final = F.one_hot(full_mask.argmax(dim=0), 2).permute(2, 0, 1).numpy()
        final = Image.fromarray((np.argmax(final, axis=0) * 255 / final.shape[0]).astype(np.uint8))

        seg_mask = np.array(final)

        original_frame = np.array(original_frame)


        altered_mask = cv2.cvtColor(seg_mask, cv2.COLOR_GRAY2BGR)
        img_out = cv2.addWeighted(original_frame, 0.4, altered_mask, 0.6, 0)
      

        out.write(img_out)

        cv2.imshow('segmented_mask_ouput',seg_mask)
        cv2.imshow('output_on_face', img_out)
        cv2.waitKey(1)
        count += 1
        img_out = seg_mask
        prev = frame_id 

    out.release()

def runSegmentation(worker, dpu, queueIn, queueOut,input_scale):
    global isCapturing
    print('Worker: {}'.format(worker))
    inputTensors  = dpu.get_input_tensors()
    outputTensors = dpu.get_output_tensors()

    output_fixpos = outputTensors[0].get_attr("fix_point")
    output_scale = 1 / (2**output_fixpos)

    internal_list = []
    frame_id_list = []

    while isCapturing:
        if queueIn.empty():
            time.sleep(0.2)
            continue
        
        frame_id, original_frame = queueIn.get() 
        img = original_frame
        img = np.array(img)
        img_bgr = img[:,:,::-1]
        img_ndarray = img * (1/255) * input_scale

        outputData = []
        inputData  = []
        outputData.append(np.empty(tuple(outputTensors[0].dims), dtype=np.int8, order = 'C'))
        inputData.append(np.empty(tuple(inputTensors[0].dims), dtype = np.int8, order = 'C'))

        imageRun = inputData[0]
        imageRun[0, ...] = img_ndarray.reshape(inputTensors[0].dims[1:])

        job_id = dpu.execute_async(inputData, outputData)
        dpu.wait(job_id)

        queueOut.put((frame_id, (img_bgr, outputData[0][0]*output_scale)))

def get_subgraph(g):
    sub = []
    root = g.get_root_subgraph()
    sub = [ s for s in root.children if s.metadata.get_attr_str ("device") == "DPU"]
    return sub

if __name__ == "__main__":
    # Change variables below for your setup
    threads = 1
    path = 'resize.avi'
    model   = '../vck_5000_class_weight/YNET_FFT.xmodel' 

    g = xir.Graph.deserialize(model)
    subgraphs = g.get_root_subgraph().toposort_child_subgraph()

    dpu_runners = []
    for i in range(int(threads)):
        dpu_runners.append(vart.Runner.create_runner(subgraphs[1], "run"))

    input_fixpos = dpu_runners[0].get_input_tensors()[0].get_attr("fix_point")
    input_scale = 2**input_fixpos

    # Init synchronous queues for inter-thread communication
    queueIn  = queue.Queue()
    queueOut = queue.PriorityQueue()

    # Launch threads
    threadAll = []
    taskCapture = threading.Thread(target=streamCapture, args=(path, queueIn,input_scale))
    threadAll.append(taskCapture)

    for i in range(threads):
        taskPrediction = threading.Thread(target=runSegmentation, args=(i, dpu_runners[i], queueIn, queueOut,input_scale))
        threadAll.append(taskPrediction)

    taskDisplay = threading.Thread(target=outputStream, args=(queueOut,))
    threadAll.append(taskDisplay)

    global isCapturing
    isCapturing = True

    for t in threadAll:
        t.start()

    # Wait for all threads to stop
    for t in threadAll:
        t.join()

    # clean up resources
    for runner in dpu_runners:
        del runner