from ctypes import *
from typing import List
import cv2
import numpy as np
import vart
import xir
import os
import pathlib
import xir
import threading
import time
import sys
import argparse
import math
from PIL import Image
import json

def Sigmoid1(xx):
    x = np.asarray( xx, dtype="float128")
    t = 1 / (1 + np.exp(-x))
    return t

def ResizeImageArr( path ):
    #path1 = 'dataset/img1.png'
    img = Image.open(path)
    newW,newH = 224,224
    pil_img = img.resize((newW, newH), resample=Image.BICUBIC)
    img_ndarray = np.asarray(pil_img)
    img_ndarray = img_ndarray / 255
    return img_ndarray




def get_child_subgraph_dpu(graph: "Graph") -> List["Subgraph"]:
    assert graph is not None, "'graph' should not be None."
    root_subgraph = graph.get_root_subgraph()
    assert (
        root_subgraph is not None
    ), "Failed to get root subgraph of input Graph object."
    if root_subgraph.is_leaf:
        return []
    child_subgraphs = root_subgraph.toposort_child_subgraph()
    assert child_subgraphs is not None and len(child_subgraphs) > 0
    return [
        cs
        for cs in child_subgraphs
        if cs.has_attr("device") and cs.get_attr("device").upper() == "DPU" or "CPU"
    ]

def execute_async(dpu, tensor_buffers_dict):
    input_tensor_buffers = [
        tensor_buffers_dict[t.name] for t in dpu.get_input_tensors()
    ]
    output_tensor_buffers = [
        tensor_buffers_dict[dpu.get_output_tensors()[1]]
    ]
    jid = dpu.execute_async(input_tensor_buffers, output_tensor_buffers)
    return dpu.wait(jid)

def runDPU(id,start,dpu,img):

    '''get tensor'''
    inputTensors = dpu.get_input_tensors()
    outputTensors = dpu.get_output_tensors()
    input_ndim = tuple(inputTensors[0].dims)
    output_ndim = tuple(outputTensors[0].dims)
    output_ndim_1 = tuple(outputTensors[1].dims)
    print('output tensor = ',output_ndim)
    


    print("\nrunDPU-  INPUT DIM: ",  input_ndim)
    print("\nrunDPU- OUTPUT DIM: ", output_ndim)

    batchSize = input_ndim[0]
    print()
    n_of_images = len(img)
    count = 0
    write_index = start
    print("\nrunDPU- batchSize: ", batchSize)
    print("\nrunDPU- # images : ", n_of_images)
    

    while count < n_of_images:
        if (count+batchSize<=n_of_images):
            runSize = batchSize
        else:
            runSize=n_of_images-count
        if count==0:
            print("\nrunDPU- # runSize : ", runSize)

        '''prepare batch input/output '''
        outputData = []
        inputData = []
        inputData = [np.empty(input_ndim, dtype=np.float32, order="C")]
        #outputData = [np.empty(output_ndim,dtype=np.float32,order="c")]
        outputData = [np.empty(output_ndim, dtype=np.float32, order="C"),np.empty(output_ndim_1,dtype=np.int8,order="C")]

        '''init input image to input buffer '''
        for j in range(runSize):
            imageRun = inputData[0]
            imageRun[j, ...] = img[(count + j) % n_of_images].reshape(input_ndim[1:])

        '''run with batch '''
        job_id = dpu.execute_async(inputData,outputData)
        dpu.wait(job_id)
       
        '''store output vectors '''
        for j in range(runSize):
            out_q[write_index] = outputData[0][j]
            class_q[write_index] = outputData[1][j]
            write_index += 1


        count = count + runSize

    
    print("\nrunDPU: write_index : ", write_index)

def app(image_dir,threads,model):

    # load testing images
    print("\nAPP- loading segmentation images and preprocessing test images")
    test_images = os.listdir(image_dir)
    test_images.sort()
    print("\nAPP- TEST IMAGES = ", test_images)
    print("\nAPP- LEN TEST IMAGES = ", len(test_images))

    runTotal = len(test_images)
    dir_test_seg = image_dir + "dataset/"
    test_segmentations  = os.listdir(image_dir)
    test_segmentations.sort()

    X_test = []
    pre_process_time_0 = time.time()
    for im  in (test_images) :
        X_test.append(ResizeImageArr(  os.path.join(image_dir, im)) )
    X_test = np.array(X_test)
    pre_process_time_1 = time.time()
    print('total time take for resizing and appending to list = ',pre_process_time_1-pre_process_time_0)
    global out_q
    global class_q
    out_q = [None] * runTotal
    class_q = [None]* runTotal

    all_dpu_runners = []


    g = xir.Graph.deserialize(model)
    subgraphs = g.get_root_subgraph().toposort_child_subgraph()
    


    #assert len(subgraphs) == 1  # only one DPU kernel
    for i in range(threads):
        all_dpu_runners.append(vart.Runner.create_runner(subgraphs[1], "run"))

    '''run threads '''
    print('\nAPP- Starting',threads,'threads...')
    threadAll = []
    start=0
    for i in range(threads):
        if (i==threads-1):
            end = len(X_test)
        else:
            end = start+(len(X_test)//threads)
        in_q = X_test[start:end]

        t1 = threading.Thread(target=runDPU, args=(i,start,all_dpu_runners[i], in_q))
        threadAll.append(t1)
        start=end

    time1 = time.time()
    for x in threadAll:
        x.start()
    for x in threadAll:
        x.join()
    time2 = time.time()
    timetotal = time2 - time1


    fps = float(runTotal / timetotal)
    print("\nAPP- FPS=%.2f, total frames = %.0f , time=%.4f seconds\n" %(fps,runTotal, timetotal))
    print('saving npy file')
    np.save("vck_benchmark.npy", out_q)
    print('saving done')

    print('starting validation for classification part')
    final = np.array(class_q)
    test_dir = os.listdir('unseen_data/masks')
    test_dir.sort()
    file = open('metadata.json')
    file = json.load(file)
    test_dir_crops = os.listdir('unseen_data/crops')
    test_dir_crops.sort()
    count = 0
    true = 0
    false = 0
    actual_real=0
    for idx,(i,mask,crops) in enumerate(zip(final,test_dir,test_dir_crops)):
        filename = 'unseen_data/masks/'+mask
        filename_crops = 'unseen_data/crops/'+crops
        
        mask_img = Image.open(filename)
        crop_img = Image.open(filename_crops)
        output = Sigmoid1(i)
            
        name = crops.split('_')[0]

        if file[name+'.mp4']['label'] == 'REAL':
            true_out = 1.0
            actual_real+=1
        else:
            true_out = 0.0
            
        if output>0.5:
            output = 1.0
        else:
            output = 0.0
            
        if output == true_out:
            true+=1
        else:
            false+=1
        if output>0.5:
            mask_img.save('prediction/'+mask)
            count+=1
    print('total real images predicted by the model = ',count)
    print('Correct {}, Wrong {}'.format(true, false))

   


# only used if script is run as 'main' from command line
def main():

  # construct the argument parser and parse the arguments
  ap = argparse.ArgumentParser()
  ap.add_argument('-d', '--images',  type=str,default='dataset',            help='Path to folder of images.')
  ap.add_argument('-t', '--threads', type=int, default=1, help='Number of threads. Default is 1')
  ap.add_argument('-m', '--model',   type=str,default='unet_deploy.xmodel',            help='Path of xmodel')

  args = ap.parse_args()

  print ('Command line options:')
  print (' --images  : ', args.images)
  print (' --threads : ', args.threads)
  print (' --model   : ', args.model)


  app(args.images,args.threads,args.model)

main()


