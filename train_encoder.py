from typing import List, Tuple,Optional
import torch
from torch.utils.data import DataLoader
import time

from dataset import liverDataset
import encoding_unet

from dice_loss import binary_dice_loss,binary_dice_coeff
import numpy as np
import os
import platform
from pathlib import Path
from multiprocessing import cpu_count
import toolkit


# configuration
CONFIG_DEVICE:torch.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')


suffix:str=time.strftime('%m-%d+%H-%M-%S', time.localtime(time.time()))
WEIGHTS_SAVE_DIR:str='./weights_unet_encoder'
WEIGHTS_SAVE_DIR+=suffix
if Path(WEIGHTS_SAVE_DIR).is_dir()==False:
    os.mkdir(WEIGHTS_SAVE_DIR)

CONFIG_NUM_WORKERS = 0 if platform.system()=='Windows' else min(max(cpu_count()-2,0),10)

BATCH_SIZE:np.int32=2

DEBUG_MODE:bool=True

print('-----------configuration-----------')
print('Device:',CONFIG_DEVICE)
print('Workers number:',CONFIG_NUM_WORKERS)
print('-----------------------------------')

# Plotting
LOG_DIR='./log_encoder'
LOG_DIR+=suffix
if Path(LOG_DIR).is_dir()==False:
    os.mkdir(LOG_DIR)
print('statistics:',LOG_DIR)
SAMPLE_NUM_EPOCH = 3

# neural networks
model=encoding_unet.U_net(1,1,)


# loss functions
bce_loss_func=torch.nn.BCELoss().to(CONFIG_DEVICE)

optimizer=torch.optim.Adam(model.parameters())

train_dataset=liverDataset('./dataset/train',None,None)

train_loader=DataLoader(train_dataset,BATCH_SIZE,shuffle=True,num_workers=CONFIG_NUM_WORKERS)

val_dataset=liverDataset('./dataset/val',None,None)

val_loader=DataLoader(val_dataset,BATCH_SIZE,shuffle=True,num_workers=CONFIG_NUM_WORKERS)

# exit()

def train_iteration(model:encoding_unet.U_net, \
        optimizer:torch.optim.Adam, \
        labels:torch.Tensor)->Tuple[float]:
    """
    return float(bce, dice, total_loss,)
    forward + backward + update on labels
    """
    if model.training == False:
        model.train()

    optimizer.zero_grad()
    # forward
    x1_0,x2_0,x3_0,x4_0,x0_4=model.multi_forward(labels)

    # calculate loss
    bce:torch.Tensor=bce_loss_func(x0_4,labels)
    dice:torch.Tensor=binary_dice_loss(x0_4,labels)

    total_loss:torch.Tensor= bce+dice

    # backward & update
    total_loss.backward()
    optimizer.step()

    return (bce.item(), dice.item(), total_loss.item(),)

def validate(model:encoding_unet.U_net, data_loader:DataLoader)->float:
    """
    return float(score)
    """
    if model.training==True:
        model.eval()
    
    score=0.0
    total_count:int=0
    print('<----validate /{}---->'.format(len(data_loader)))
    with torch.no_grad():
        for i,(_,labels) in enumerate(data_loader):            
            labels:torch.Tensor
            labels=labels.to(CONFIG_DEVICE)

            x0_4:torch.Tensor
            x0_4=model.forward(labels)

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

    bce_loss_epochs:List[float]=[]
    dice_loss_epochs:List[float]=[]

    dice_score_epochs:List[List[float]] =[]
    
    for epoch in range(20):
        bce_loss, dice_loss, total_loss=0.0, 0.0, 0.0
        total_count:int=0
        print('------epoch{}------'.format(epoch))
        print('<======Train, total batches: {}======>'.format(len(train_loader)))
        for i,(_,labels) in enumerate(train_loader):
            labels:torch.Tensor
            labels = labels.to(CONFIG_DEVICE)

            bce, dice, total= \
                train_iteration(model,optimizer,labels)
            
            bce_loss+=bce*labels.size(0)
            dice_loss+=dice*labels.size(0)
            total_loss+=total*labels.size(0)
            total_count+=labels.size(0)
            if i%modulus==0:
                print('\tProgress: {}/{}| loss: bce={}, dice={}, total={}' \
                    .format(i,len(train_loader), bce,dice, total))
                bce_loss_batches.append(bce)
                dice_loss_batches.append(dice)
            if(DEBUG_MODE==True):
                break
        print()
        print('-------Train done, loss: bce={}, dice={}, total={},--------'\
            .format(bce_loss/total_count,dice_loss/total_count, \
                    total_loss/total_count))
        bce_loss_epochs.append(bce_loss/total_count)
        dice_loss_epochs.append(dice_loss/total_count)
        print('<======eval======>')
        dice_score \
            = validate(model,val_loader)
        dice_arr=(0,0,0,dice_score,)
        dice_score_epochs.append(dice_score)
        print('dice score: ',dice_score)
        best_level=np.argmax(dice_arr)

        torch.save(model.state_dict(),os.path.join(WEIGHTS_SAVE_DIR,'encoder_{}_level{}_{:04d}.pth'.format(epoch,4,int(dice_score*10000))))

        if DEBUG_MODE==True:
            break
        
    """
    Plot the statistics
    """
    dice_score_epochs_:np.ndarray=np.array(dice_score_epochs)

    
    toolkit.log_statistics(LOG_DIR,SAMPLE_NUM_EPOCH,dice_loss_batches,bce_loss_batches,None,dice_loss_epochs,bce_loss_epochs,None,dice_score_epochs_)