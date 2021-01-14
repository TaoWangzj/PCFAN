"""
paper: Pyramid Channel-based Feature Attention Network for image dehazing 
file: network.py
about: model for PCFAN
author: Tao Wang
date: 01/13/21
"""
# --- Imports --- #
from __future__ import print_function  
import argparse
import torch
import torch.nn as nn  
import torch.optim as optim
from torch.autograd import Variable 
from torch.utils.data import DataLoader
from model.network import Net
from datasets.datasets import DehazingDataset
from os.path import exists, join, basename
import time
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms
from loss.edg_loss import edge_loss
from utils import to_psnr, validation, print_log
import os
import time


# --- Parse hyper-parameters  --- #
parser = argparse.ArgumentParser(description='Training hyper-parameters for neural network')
parser.add_argument('--batchSize', type=int, default=1, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=20, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate. Default=0.01')
parser.add_argument('--threads', type=int, default=15, help='number of threads for data loader to use')
parser.add_argument('--net', default='', help="path to net_Dehazing (to continue training)")
parser.add_argument('--continueEpochs', type=int, default=0, help='continue epochs')
parser.add_argument("--n_GPUs", help='list of GPUs for training neural network', default=[0], type=list)
parser.add_argument('--category', help='Set image category (indoor or outdoor?)', default='indoor', type=str)
opt = parser.parse_args()
print(opt)



# ---  hyper-parameters for training and testing the neural network --- #
train_data_dir = './data/train/ITS/'
train_batch_size = opt.batchSize
val_batch_size = opt.testBatchSize
train_epoch = opt.nEpochs
data_threads = opt.threads
GPUs_list = opt.n_GPUs
category = opt.category
continueEpochs = opt.continueEpochs


# --- Set category-specific hyper-parameters  --- #
if category == 'indoor':
    val_data_dir = './data/test/SOTS/indoor/'
elif category == 'outdoor':
    val_data_dir = './data/test/SOTS/outdoor/'
else:
    raise Exception('Wrong image category. Set it to indoor or outdoor for RESIDE dateset.')



device_ids = GPUs_list
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- Define the network --- #
print('===> Building model')
model = Net()




# --- Define the MSE loss --- #
MSELoss = nn.MSELoss()
MSELoss = MSELoss.to(device)


# --- Multi-GPU --- #
model = model.to(device)
model = nn.DataParallel(model, device_ids=device_ids)



# --- Build optimizer and scheduler --- #
optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999)) 
scheduler = StepLR(optimizer,step_size= train_epoch // 2,gamma=0.1)


# --- Load training data and validation/test data --- #
train_dataset = DehazingDataset(root_dir=train_data_dir, transform=transforms.Compose([transforms.ToTensor()]))
train_dataloader = DataLoader(dataset = train_dataset, batch_size=train_batch_size, num_workers = data_threads, shuffle=True)

test_dataset = DehazingDataset(root_dir = val_data_dir, transform = transforms.Compose([transforms.ToTensor()]), train=False)

test_dataloader = DataLoader(test_dataset, batch_size = val_batch_size, num_workers = data_threads, shuffle=False)


old_val_psnr, old_val_ssim = validation(model, test_dataloader, device, category)
print('old_val_psnr: {0:.2f}, old_val_ssim: {1:.4f}'.format(old_val_psnr, old_val_ssim))
for epoch in range(1 + opt.continueEpochs, opt.nEpochs + 1 + opt.continueEpochs):
    print("Training...")
    scheduler.step()
    epoch_loss = 0
    psnr_list = []
    for iteration, inputs in enumerate(train_dataloader,1):

        haze, gt = Variable(inputs['hazy_image']), Variable(inputs['clear_image'])
        haze = haze.to(device)
        gt = gt.to(device)

        # --- Zero the parameter gradients --- #
        optimizer.zero_grad()

        # --- Forward + Backward + Optimize --- #
        model.train()
        dehaze = model(haze)
        MSE_loss = MSELoss(dehaze, gt)
        EDGE_loss = edge_loss(dehaze, gt, device)
        Loss = MSE_loss +0.01*EDGE_loss
        epoch_loss +=Loss
        Loss.backward()
        optimizer.step()

        if iteration % 100 == 0:
            print("===>Epoch[{}]({}/{}): Loss: {:.4f} MSELoss: {:.4f} EDGELoss: {:.4f}".format(epoch, iteration, len(train_dataloader), Loss.item(), MSE_loss.item(), EDGE_loss.item()))
        
        # --- To calculate average PSNR --- #
        psnr_list.extend(to_psnr(dehaze, gt))

    train_psnr = sum(psnr_list) / len(psnr_list)
    save_checkpoints = './checkpoints'
    if os.path.isdir(save_checkpoints)== False:
        os.mkdir(save_checkpoints)

    # --- Save the network  --- #
    torch.save(model.state_dict(), './checkpoints/{}_haze.pth'.format(category))

    # --- Use the evaluation model in testing --- #
    model.eval()

    val_psnr, val_ssim = validation(model, test_dataloader, device, category)
    
    # --- update the network weight --- #
    if val_psnr >= old_val_psnr:
        torch.save(model.state_dict(), './checkpoints/{}_haze_best.pth'.format(category))
        old_val_psnr = val_psnr
    
#     print(''' 'Date: {0}s, Epoch: [{1}/{2}], Train_PSNR: {3:.2f}, Val_PSNR: {4:.2f}, Val_SSIM: {5:.4f}'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), 
#          epoch, num_epochs, train_psnr, val_psnr, val_ssim)''')
    print_log(epoch+1, train_epoch, train_psnr, val_psnr, val_ssim, category)
    




