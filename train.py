from typing import List, Tuple,Optional
import torch
from torch.utils.data import DataLoader
# from torch.utils.tensorboard.writer import SummaryWriter
import time

from dataset import liverDataset
import unet

from dice_loss import binary_dice_loss,binary_dice_coeff
import numpy as np
import os
import platform
from pathlib import Path
from multiprocessing import cpu_count
import toolkit


# configuration
CONFIG_DEVICE:torch.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ENCODER_LOAD_PATH='./trained_encoder/encoder_18_level4_9798.pth'

suffix:str=time.strftime('%m-%d+%H-%M-%S', time.localtime(time.time()))
WEIGHTS_SAVE_DIR:str='./weights_unet_common'
WEIGHTS_SAVE_DIR+=suffix
if Path(WEIGHTS_SAVE_DIR).is_dir()==False:
    os.mkdir(WEIGHTS_SAVE_DIR)

CONFIG_NUM_WORKERS = 0 if platform.system()=='Windows' else min(max(cpu_count()-2,0),10)

BATCH_SIZE:np.int32=2

USE_BOTTLE_NECK=False

DEBUG_MODE:bool=True

print('-----------configuration-----------')
print('Device:',CONFIG_DEVICE)
print('Workers number:',CONFIG_NUM_WORKERS)
print('-----------------------------------')

# Plotting
LOG_DIR='./log_common'

LOG_DIR+=suffix
if Path(LOG_DIR).is_dir()==False:
    os.mkdir(LOG_DIR)
print('statistics:',LOG_DIR)
SAMPLE_NUM_EPOCH = 3

# neural networks
model=unet.U_net(1,1,)
# ref_model = encoding_unetpp.NestedUNet(1,1,) if USE_BOTTLE_NECK else None
ref_model=None
if ref_model is not None:
    ref_model.load_state_dict(\
        torch.load(ENCODER_LOAD_PATH, map_location='cpu'))
    ref_model.to(CONFIG_DEVICE)
    ref_model.eval()
    print('reference encoder loads successfully √')


# loss functions
bce_loss_func=torch.nn.BCELoss().to(CONFIG_DEVICE)
mse_loss_func=torch.nn.MSELoss().to(CONFIG_DEVICE)

optimizer=torch.optim.Adam(model.parameters())

train_dataset=liverDataset('./dataset/train',None,None)

train_loader=DataLoader(train_dataset,BATCH_SIZE,shuffle=True,num_workers=CONFIG_NUM_WORKERS)

val_dataset=liverDataset('./dataset/val',None,None)

val_loader=DataLoader(val_dataset,BATCH_SIZE,shuffle=True,num_workers=CONFIG_NUM_WORKERS)

# exit()

def train_iteration(model:unet.U_net, \
        ref_model:Optional[unet.U_net], \
        optimizer:torch.optim.Adam, \
        raw_imgs:torch.Tensor,labels:torch.Tensor)->Tuple[float]:
    """
    return float(bce, dice, mse, total_loss,)
    forward + backward + update on raw_imgs
    """
    if model.training == False:
        model.train()

    optimizer.zero_grad()
    # forward
    x1_0,x2_0,x3_0,x4_0,x0_4=model.multi_forward(raw_imgs)

    # calculate loss
    bce:torch.Tensor=bce_loss_func(x0_4,labels)
    dice:torch.Tensor=binary_dice_loss(x0_4,labels)

    total_loss:torch.Tensor= bce+dice

    if ref_model!=None:
        if ref_model.training == True:
            ref_model.eval()
        with torch.no_grad():
            ref_x1_0,ref_x2_0,ref_x3_0,ref_x4_0=ref_model.encode(raw_imgs)
        mse:torch.Tensor = mse_loss_func(x1_0,ref_x1_0) \
            +mse_loss_func(x2_0,ref_x2_0) \
            +mse_loss_func(x3_0,ref_x3_0) \
            +mse_loss_func(x4_0,ref_x4_0)
        total_loss+=mse
    else:
        mse=torch.zeros(1)

    # backward & update
    total_loss.backward()
    optimizer.step()

    return (bce.item(), dice.item(), mse.item(), total_loss.item(),)

def validate(model:unet.U_net, data_loader:DataLoader)->float:
    """
    return float(score)
    """
    if model.training==True:
        model.eval()
    
    score=0.0
    total_count:int=0
    print('<----validate /{}---->'.format(len(data_loader)))
    with torch.no_grad():
        for i,(raw_imgs,labels) in enumerate(data_loader):            
            raw_imgs:torch.Tensor
            labels:torch.Tensor
            raw_imgs, labels=raw_imgs.to(CONFIG_DEVICE),labels.to(CONFIG_DEVICE)

            x0_4:torch.Tensor
            x0_4=model.forward(raw_imgs)

            dice_grade=binary_dice_coeff(x0_4,labels)

            score+=dice_grade.item()*labels.size(0)
            total_count+=labels.size(0)

            if DEBUG_MODE==True:
                assert(dice_grade.item()>=0 and dice_grade.item()<=1)
                print('check reasonal dice score √')
                break
    
    return score/total_count

            
if __name__=='__main__':
    model=model.to(CONFIG_DEVICE)
    model.train()
    
    print(type(optimizer))
    print(type(train_loader))

    modulus:int=int(np.ceil(len(train_loader)/SAMPLE_NUM_EPOCH))

    # Statistics
    bce_loss_batches:List[float]=[]
    dice_loss_batches:List[float]=[]
    mse_loss_batches:List[float]=[]

    bce_loss_epochs:List[float]=[]
    dice_loss_epochs:List[float]=[]
    mse_loss_epochs:List[float]=[]

    dice_score_epochs:List[List[float]] =[]
    
    for epoch in range(20):
        if epoch>=20:
            ref_model=None
        bce_loss, dice_loss, mse_loss, total_loss=0.0, 0.0, 0.0, 0.0
        total_count:int=0
        print('------epoch{}------'.format(epoch))
        print('<======Train, total batches: {}======>'.format(len(train_loader)))
        for i,(raw_imgs,labels) in enumerate(train_loader):
            raw_imgs:torch.Tensor
            labels:torch.Tensor
            raw_imgs,labels = raw_imgs.to(CONFIG_DEVICE),labels.to(CONFIG_DEVICE)

            bce, dice, mse, total= \
                train_iteration(model,ref_model,optimizer,raw_imgs,labels)
            
            bce_loss+=bce*labels.size(0)
            dice_loss+=dice*labels.size(0)
            mse_loss+=mse*labels.size(0)
            total_loss+=total*labels.size(0)
            total_count+=labels.size(0)
            if i%modulus==0:
                print('\tProgress: {}/{}| loss: bce={}, dice={},mse={}, total={}' \
                    .format(i,len(train_loader), bce,dice,mse, total))
                bce_loss_batches.append(bce)
                dice_loss_batches.append(dice)
                mse_loss_batches.append(mse)
            if(DEBUG_MODE==True):
                break
        print()
        print('-------Train done, loss: bce={}, dice={}, mse={}, total={},--------'\
            .format(bce_loss/total_count,dice_loss/total_count, \
                    mse_loss/total_count,total_loss/total_count))
        bce_loss_epochs.append(bce_loss/total_count)
        dice_loss_epochs.append(dice_loss/total_count)
        mse_loss_epochs.append(mse_loss/total_count)
        print('<======eval======>')
        dice_score \
            = validate(model,val_loader)
        dice_arr=(0,0,0,dice_score,)
        dice_score_epochs.append(dice_score)
        print('dice score: ',dice_score)
        best_level=np.argmax(dice_arr)

        torch.save(model.state_dict(),os.path.join(WEIGHTS_SAVE_DIR,'unet_{}_level{}_{:04d}.pth'.format(epoch,best_level+1,int(dice_arr[best_level]*10000))))

        if DEBUG_MODE==True:
            break
        
    """
    Plot the statistics
    """
    dice_score_epochs_:np.ndarray=np.array(dice_score_epochs)

    if USE_BOTTLE_NECK==False:
        mse_loss_batches=None
        mse_loss_epochs=None
    
    toolkit.log_statistics(LOG_DIR,SAMPLE_NUM_EPOCH,dice_loss_batches,bce_loss_batches,mse_loss_batches,dice_loss_epochs,bce_loss_epochs,mse_loss_epochs,dice_score_epochs_)