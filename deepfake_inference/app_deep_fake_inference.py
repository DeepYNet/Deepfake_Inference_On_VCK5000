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

def Sigmoid(x):
    t = 1 / (1 + np.exp(-x))
    return t

def ResizeImageArr( path, fix_scale ):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    newW,newH = 224,224
    image = cv2.resize(img, (newW, newH), interpolation=cv2.INTER_LINEAR)
    image = image * (1/255.0) * fix_scale
    image_data = image.astype(np.int8)
    return image_data


def runDPU(id,start,dpu,img):
    inputTensors = dpu.get_input_tensors()
    outputTensors = dpu.get_output_tensors()
    output_fixpos = outputTensors[0].get_attr("fix_point")
    output_fixpos_2 = outputTensors[1].get_attr("fix_point")
    print('fixpos for output = ',output_fixpos)
    print('fixpos for output 2 = ',output_fixpos_2)
    
    output_scale = 1 / (2**output_fixpos)
    output_scale_2 = 1 / (2**output_fixpos_2)

    input_ndim = tuple(inputTensors[0].dims)
    output_ndim0 = tuple(outputTensors[0].dims)
    output_ndim1 = tuple(outputTensors[1].dims)

    print('output tensor = ',output_ndim0)
    print("[run_DPU]: INPUT DIM: ",  input_ndim)
    print("[run_DPU]: OUTPUT 0 DIM: ", output_ndim0)
    print("[run_DPU]: OUTPUT 1 DIM: ", output_ndim1)

    batchSize = input_ndim[0]
    print()
    n_of_images = len(img)
    count = 0
    write_index = start
    print("[run_DPU]: batchSize: ", batchSize)
    print("[run_DPU]: # images : ", n_of_images)
    

    while count < n_of_images:
        if (count+batchSize<=n_of_images):
            runSize = batchSize
        else:
            runSize=n_of_images-count
        if count==0:
            print("[run_DPU]: # runSize : ", runSize)
        '''prepare batch input/output '''
        outputData = []
        inputData = []
        inputData = [np.empty(input_ndim, dtype=np.int8, order="C")]
        outputData = [np.empty(output_ndim0, dtype=np.int8, order="C"),np.empty(output_ndim1, dtype=np.int8, order="C")]

        '''init input image to input buffer '''
        for j in range(runSize):
            imageRun = inputData[0]
            imageRun[j, ...] = img[(count + j) % n_of_images].reshape(input_ndim[1:])

        '''run with batch '''
        job_id = dpu.execute_async(inputData,outputData)
        dpu.wait(job_id)
        
       
        '''store output vectors '''
        for j in range(runSize):
            out_q[write_index] = (outputData[0][j]*output_scale)
            class_q[write_index] = (outputData[1][j]*output_scale_2)
            write_index += 1
        
        count = count + runSize

    print("\nrunDPU: write_index : ", write_index)

def get_child_subgraph_dpu(graph: "Graph") -> List["Subgraph"]:
    assert graph is not None, "'graph' should not be None."
    root_subgraph = graph.get_root_subgraph()
    assert (root_subgraph
            is not None), "Failed to get root subgraph of input Graph object."
    if root_subgraph.is_leaf:
        return []
    child_subgraphs = root_subgraph.toposort_child_subgraph()
    assert child_subgraphs is not None and len(child_subgraphs) > 0
    return [
        cs for cs in child_subgraphs
        if cs.has_attr("device") and cs.get_attr("device").upper() == "DPU"
    ]

def app(image_dir,threads,model):
    # load testing images
    print("[APP]: loading segmentation images and preprocessing test images")
    test_images = os.listdir(image_dir)
    test_images.sort()
    print("[APP]: LEN TEST IMAGES = ", len(test_images))

    runTotal = len(test_images)

    test_segmentations  = os.listdir(image_dir)
    test_segmentations.sort()
    g = xir.Graph.deserialize(model)
    subgraphs = get_child_subgraph_dpu(g)
    all_dpu_runners = []
    assert len(subgraphs) == 1  # only one DPU kernel
    print('\nAPP - Found',len(subgraphs),'subgraphs in',model)
    for i in range(threads):
        all_dpu_runners.append(vart.Runner.create_runner(subgraphs[0], "run"))

    input_fixpos = all_dpu_runners[0].get_input_tensors()[0].get_attr("fix_point")
    input_scale = 2**input_fixpos
    X_test = []
    Y_test = []


    for im  in (test_images) :
        X_test.append(ResizeImageArr(  os.path.join(image_dir, im),input_scale) )
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)
    global out_q
    global class_q
    out_q = [None] * runTotal
    class_q = [None] * runTotal
    
    '''run threads '''
    print('[APP]: Starting',threads,'threads...')
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
    print("[APP]: FPS=%.2f, total frames = %.0f , time=%.4f seconds\n" %(fps,runTotal, timetotal))
    print('saving npy file')
    np.save("segmentation_output_array.npy", out_q)
    np.save("classification_output_array.npy",class_q)
    print('saving done')

    print('starting validation for classification part')
    file = open('metadata.json')
    file = json.load(file)
    test_dir_crops = os.listdir('unseen_data/crops')
    test_dir_crops.sort()
    count = 0
    true = 0
    false = 0
    actual_real=0
    for idx,(i,crops) in enumerate(zip(class_q,test_dir_crops)):
        filename_crops = 'unseen_data/crops/'+crops
        
        crop_img = Image.open(filename_crops)
        output = Sigmoid(i)
            
        name = crops.split('_')[0]

        if file[name+'.mp4']['label'] == 'REAL':
            true_out = 1.0
        else:
            true_out = 0.0

        output = np.round(output)
            
            
        if output == true_out:
            true+=1
        else:
            false+=1
    print('Correct {}, Wrong {}'.format(true, false))
    print('The accuracy on unseen data is = ',100 * (true/len(test_images)))
   
# only used if script is run as 'main' from command line
def main():

  # construct the argument parser and parse the arguments
  ap = argparse.ArgumentParser()
  ap.add_argument('-d', '--images',  type=str,default='images',            help='Path to folder of images.')
  ap.add_argument('-t', '--threads', type=int, default=1, help='Number of threads. Default is 1')
  ap.add_argument('-m', '--model',   type=str,default='unet.xmodel',            help='Path of xmodel')

  args = ap.parse_args()

  print ('Command line options:')
  print (' --images  : ', args.images)
  print (' --threads : ', args.threads)
  print (' --model   : ', args.model)


  app(args.images,args.threads,args.model)

main()

