import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torch
from torch.utils.data import DataLoader, random_split
import os
import sys
import random
import argparse
from pytorch_nndct.apis import torch_quantizer, dump_xmodel
from evaluate import evaluate
from unet import UNet
from utils.data_loading import CarvanaDataset
from pathlib import Path




DIVIDER = '-----------------------------------------'





def quantization(model,build_dir,batch_size,quant_mode):
    img_scale = 0.5
    val_percent = 0.1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dir_img = Path('fyp_data/crops/')
    dir_mask = Path('fyp_data/masks/')
    float_model = build_dir + '/float_model'
    quant_model = build_dir + '/quant_model'
    dataset = CarvanaDataset(dir_img, dir_mask, img_scale)

    finetune = True

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

  

    net = model
    net.load_state_dict(torch.load(os.path.join(float_model,'checkpoint_epoch_224x224_35.pth'),map_location='cuda:0'))
    net.cuda()
    
    if (quant_mode=='test'):
        batch_size = 1
    rand_in = torch.randn([batch_size, 3, 224, 224])
    quantizer = torch_quantizer(quant_mode, net, (rand_in), output_dir=quant_model) 
    quantized_model = quantizer.quant_model
    
    
    if finetune == True:
      if quant_mode == 'calib':
        quantizer.fast_finetune(evaluate, (quantized_model, val_loader,device))
      elif quant_mode == 'test':
        quantizer.load_ft_param()
    

    val_score = evaluate(quantized_model,val_loader,device)
    print('val_score is = ',val_score)

    if quant_mode == 'calib':
        quantizer.export_quant_config()
    if quant_mode == 'test':
        quantizer.export_xmodel(deploy_check=False, output_dir=quant_model)
  
    return

def run_main():

  # construct the argument parser and parse the arguments
  ap = argparse.ArgumentParser()
  ap.add_argument('-d',  '--build_dir',  type=str, default='build',    help='Path to build folder. Default is build')
  ap.add_argument('-q',  '--quant_mode', type=str, default='calib',    choices=['calib','test'], help='Quantization mode (calib or test). Default is calib')
  ap.add_argument('-b',  '--batchsize',  type=int, default=1,        help='Testing batchsize - must be an integer. Default is 100')
  args = ap.parse_args()
  model = UNet(n_channels=3, n_classes=2, bilinear=False)
  print('\n'+DIVIDER)
  print('PyTorch version : ',torch.__version__)
  print(sys.version)
  print(DIVIDER)
  print(' Command line options:')
  print ('--build_dir    : ',args.build_dir)
  print ('--quant_mode   : ',args.quant_mode)
  print ('--batchsize    : ',args.batchsize)
  print(DIVIDER)

  quantization(model,args.build_dir,args.batchsize,args.quant_mode)

  return



if __name__ == '__main__':
    run_main()
