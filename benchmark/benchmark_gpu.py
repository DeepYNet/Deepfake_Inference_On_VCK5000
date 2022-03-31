import json
import os
from unet import UNet
import torch
import numpy as np
from PIL import Image
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'
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


net = UNet(n_channels=3, n_classes=2, bilinear=False)
net.load_state_dict(torch.load('checkpoint_epoch_224x224_35.pth',map_location=device))
net.to(device)
net.eval()

true = 0
false = 0
all_imgs = os.listdir('unseen_data/crops/')
all_imgs.sort()
time1 = time.time()
for file_name in all_imgs:
    filename = 'unseen_data/crops/'+file_name

    
    img = Image.open(filename)
    img = torch.from_numpy(preprocess(img, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)
    
    name = file_name.split('_')[0]
    
    if file[name+'.mp4']['label'] == 'REAL':
        true_out = 1.0
    else:
        true_out = 0.0
        
    mask, pred = net(img)
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




