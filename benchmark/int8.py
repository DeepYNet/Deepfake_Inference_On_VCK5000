import torch_tensorrt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torch
from torch.utils.data import DataLoader, random_split
import os
import sys
import random
import argparse
from evaluate import evaluate
from unet import UNet
from utils.data_loading import CarvanaDataset
from pathlib import Path
import time
from PIL import Image
import numpy as np
import json

file = open('metadata.json')
file = json.load(file)

def preprocess(pil_img, is_mask):

        img_ndarray = np.asarray(pil_img)


        if img_ndarray.ndim == 2 and not is_mask:
            img_ndarray = img_ndarray[np.newaxis, ...]
        elif not is_mask:
            img_ndarray = img_ndarray.transpose((2, 0, 1))

        if not is_mask:
            img_ndarray = img_ndarray / 255
            return img_ndarray

dir_img = Path('fyp_data/crops/')
dir_mask = Path('fyp_data/masks/')
img_scale = 0.5
batch_size=1

#device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = "cuda:0"

net = UNet(n_channels=3, n_classes=2, bilinear=False)
net.load_state_dict(torch.load('checkpoint_epoch_224x224_35.pth',map_location=device))
net.eval()
net.to(device)

tracer_tensor = torch.rand(1,3,224,224)
jit_model = torch.jit.trace(net,tracer_tensor.to(device))
torch.jit.save(jit_model, "unet_jit_model.jit.pt")

baseline_model = torch.jit.load("unet_jit_model.jit.pt").eval()
baseline_model = baseline_model.to(device)

dataset = CarvanaDataset(dir_img, dir_mask, img_scale)
loader_args = dict(batch_size=batch_size, num_workers=1)
train_loader = DataLoader(dataset, shuffle=False, **loader_args)

'''
calibrator = torch_tensorrt.ptq.DataLoaderCalibrator(
            train_loader,
            cache_file="./calibration.cache",
            use_cache=False,
            algo_type=torch_tensorrt.ptq.CalibrationAlgo.ENTROPY_CALIBRATION_2,
            device=torch.device("cuda:0"),
        )
'''

'''
trt_module = torch_tensorrt.compile(baseline_model,
    inputs = [torch_tensorrt.Input((1, 3, 224, 224))], # input shape   
    enabled_precisions = {torch_tensorrt.dtype.half} # Run with FP16
)
'''

calibrator = torch_tensorrt.ptq.DataLoaderCalibrator(train_loader,
                                              use_cache=False,
                                              algo_type=torch_tensorrt.ptq.CalibrationAlgo.MINMAX_CALIBRATION,
                                              device=torch.device('cuda:0'))

compile_spec = {
         "inputs": [torch_tensorrt.Input([1, 3, 224, 224])],
         "enabled_precisions": torch.int8,
         "calibrator": calibrator,
        "truncate_long_and_double": True     
     }
'''
compile_spec = {"inputs": [torch_tensorrt.Input([1, 3, 224, 224])]
               , "enabled_precisions": torch.float
               }
'''
trt_mod = torch_tensorrt.ts.compile(baseline_model, **compile_spec)

true = 0
false = 0
all_imgs = os.listdir('unseen_data/crops/')
all_imgs.sort()
time1 = time.time()
for file_name in all_imgs:
    filename = 'unseen_data/crops/'+file_name

    
    img = Image.open(filename).resize((224,224))
    img = torch.from_numpy(preprocess(img, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device,dtype=torch.float32)
    #img = img.half()
    
    name = file_name.split('_')[0]
    
    if file[name+'.mp4']['label'] == 'REAL':
        true_out = 1.0
    else:
        true_out = 0.0
        
    mask, pred = trt_mod(img)
    pred = torch.round(pred[0][0])
    pred = pred.cpu()
    pred = pred.detach().numpy()
    
    if pred == true_out:
        true+=1
    else:
        print(pred, true_out, file_name)
        false+=1
time2= time.time()
total_time = time2-time1
print('Correct {}, Wrong {}'.format(true, false))
print("Accuracy on test dataset = ",100 * (true/len(all_imgs)))
print('time taken is = ',(total_time))
print('fps = ', len(all_imgs)/total_time)
print('device device name = ',torch.cuda.get_device_name())


