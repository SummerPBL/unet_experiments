import csv
from importlib.resources import path
from ntpath import join
import torch
from torch.utils.tensorboard.writer import SummaryWriter
from typing import List,Optional
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import os

def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = (torch.arange(kernel_size).reshape(-1, 1),
          torch.arange(kernel_size).reshape(1, -1))
    filt = (1 - torch.abs(og[0] - center) / factor) * \
           (1 - torch.abs(og[1] - center) / factor)
    weight = torch.zeros((in_channels, out_channels,
                          kernel_size, kernel_size))
    weight[range(in_channels), range(out_channels), :, :] = filt
    return weight

def plot(title: str,myWriter: SummaryWriter,data:List[float]):
    for i, y in enumerate(data):
        myWriter.add_scalar(title,y,i)

def log_statistics(\
        log_dir:str, \
        samples_per_epoch:int, \
        dice_loss_per_batches:List[float], \
        bce_loss_per_batches:Optional[List[float]]=None,\
        mse_loss_per_batches:Optional[List[float]]=None, \
            \
        dice_loss_per_epoch:Optional[List[float]]=None, \
        bce_loss_per_epoch:Optional[List[float]]=None, \
        mse_loss_per_epoch:Optional[List[float]]=None, \
            \
        dice_scores_per_epoch:np.ndarray=None,\
        )->None:
    if __name__ =='__main__':
        assert(len(dice_loss_per_batches)%samples_per_epoch==0)
    plt.rcParams["font.family"] = "Times New Roman"

    tmp_x_axis_per_batches=np.linspace(0, \
        (len(dice_loss_per_batches)-1)/samples_per_epoch,\
        len(dice_loss_per_batches))
    print('长度',len(dice_loss_per_batches))
    print('分子=', len(dice_loss_per_batches)-1)
    print('末尾',(len(dice_loss_per_batches)-1)/samples_per_epoch)

    print('逐个batch x轴',tmp_x_axis_per_batches)

    # ---------------per batches--------------- #
    fig, axes = plt.subplots(1, 1, figsize=(18, 6))
    # 设置最小刻度间隔
    axes.xaxis.set_minor_locator(MultipleLocator(1/samples_per_epoch))

    # 画网格线
    axes.grid(which='minor', c='lightgrey')
    # 设置x、y轴标签
    axes.set_ylabel("loss value")
    axes.set_xlabel("epochs")
    # 设置x轴刻度
    axes.set_xticks(tmp_x_axis_per_batches)

    # 数据点注明具体数值
    axes.plot(tmp_x_axis_per_batches,dice_loss_per_batches,\
        linestyle='-', color='red', marker='s', linewidth=1.5,label='dice')
    for x,y in zip(tmp_x_axis_per_batches,dice_loss_per_batches):
        axes.text(x,y,'%.2f'%y,ha='left',va='bottom')

    if bce_loss_per_batches is not None:
        axes.plot(tmp_x_axis_per_batches,bce_loss_per_batches,\
            linestyle='-', color='green', marker='8', linewidth=1.5,label='BCE')
        for x,y in zip(tmp_x_axis_per_batches,bce_loss_per_batches):
            axes.text(x,y,'%.2f'%y,ha='left',va='bottom')
    if mse_loss_per_batches is not None:
        axes.plot(tmp_x_axis_per_batches,mse_loss_per_batches,\
            linestyle='-', color='orange', marker='p', linewidth=1.5,label='MSE')
        for x,y in zip(tmp_x_axis_per_batches,mse_loss_per_batches):
            axes.text(x,y,'%.2f'%y,ha='left',va='bottom')
    
    # 多条曲线标注
    axes.legend()
    if __name__ =='__main__':
        plt.show()
    fig.savefig(os.path.join(log_dir,'loss per batches.jpg'), dpi=800)
    
    with open(os.path.join(log_dir,'loss per batches.csv'),'w',newline='') as file:
        main_writer=csv.writer(file)
        table_header=['epoch','dice loss',]
        if bce_loss_per_batches is not None:
            table_header.append('BCE loss')
        if mse_loss_per_batches is not None:
            table_header.append('MSE loss')
        main_writer.writerow(table_header)
        for i in range(0,len(dice_loss_per_batches)):
            epoch=tmp_x_axis_per_batches[i]
            dice_loss=dice_loss_per_batches[i]
            tmp_row=[epoch,dice_loss,]
            if bce_loss_per_batches is not None:
                tmp_row.append(bce_loss_per_batches[i])
            if mse_loss_per_batches is not None:
                tmp_row.append(mse_loss_per_batches[i])
            main_writer.writerow(tmp_row)
    
    print('losses in each batch done √')


    #------------loss per epoch------------#
    if dice_loss_per_epoch is None:
        return
    tmp_x_axis_per_epoch=tuple(range(0,len(dice_loss_per_epoch)))
    if __name__ == '__main__':
        print('x轴',tmp_x_axis_per_epoch)
    fig, axes = plt.subplots(1, 1, figsize=(12, 6))
    # 设置最小刻度间隔
    axes.xaxis.set_minor_locator(MultipleLocator(1))

    # 画网格线
    axes.grid(which='minor', c='lightgrey')
    # 设置x、y轴标签
    axes.set_ylabel("loss value")
    axes.set_xlabel("epochs")
    # 设置x轴刻度
    axes.set_xticks(tmp_x_axis_per_epoch)

    # 数据点注明具体数值
    axes.plot(tmp_x_axis_per_epoch,dice_loss_per_epoch,\
        linestyle='-', color='red', marker='s', linewidth=1.5,label='dice')
    for x,y in zip(tmp_x_axis_per_epoch,dice_loss_per_epoch):
        axes.text(x,y,'%.4f'%y,ha='left',va='bottom')
    if bce_loss_per_epoch is not None:
        axes.plot(tmp_x_axis_per_epoch,bce_loss_per_epoch,\
            linestyle='-', color='green', marker='8', linewidth=1.5,label='BCE')
        for x,y in zip(tmp_x_axis_per_epoch,bce_loss_per_epoch):
            axes.text(x,y,'%.4f'%y,ha='left',va='bottom')
    if mse_loss_per_epoch is not None:
        axes.plot(tmp_x_axis_per_epoch,mse_loss_per_epoch,\
            linestyle='-', color='orange', marker='p', linewidth=1.5,label='MSE')
        for x,y in zip(tmp_x_axis_per_epoch,mse_loss_per_epoch):
            axes.text(x,y,'%.4f'%y,ha='left',va='bottom')

    # 多条曲线标注
    axes.legend()
    if __name__ == '__main__':
        plt.show()
    fig.savefig(os.path.join(log_dir,'loss per epoch.jpg'), dpi=800)

    with open(os.path.join(log_dir,'loss per epoch.csv'),'w',newline='') as file:
        main_writer=csv.writer(file)
        table_header=['epoch','dice loss',]
        if bce_loss_per_epoch is not None:
            table_header.append('BCE loss')
        if mse_loss_per_epoch is not None:
            table_header.append('MSE loss')
        main_writer.writerow(table_header)
        for i in range(0,len(dice_loss_per_epoch)):
            tmp_row=[i,dice_loss_per_epoch[i],]
            if bce_loss_per_epoch is not None:
                tmp_row.append(bce_loss_per_epoch[i])
            if mse_loss_per_epoch is not None:
                tmp_row.append(mse_loss_per_epoch[i])
            main_writer.writerow(tmp_row)
    
    print('losses in each epoch done √')

    #------------1个dice score--------------#
    if dice_scores_per_epoch is None:
        return
    # dice_scores_per_epoch= dice_scores_per_epoch.squeeze()
    tmp_x_axis_per_epoch=tuple(range(0,len(dice_scores_per_epoch)))
    if __name__ == '__main__':
        print('4个dice score, x轴',tmp_x_axis_per_epoch)

    fig, axes = plt.subplots(1, 1, figsize=(18, 6))
    # 设置最小刻度间隔
    axes.xaxis.set_minor_locator(MultipleLocator(1))

    # 画网格线
    axes.grid(which='minor', c='lightgrey')
    # 设置x、y轴标签
    axes.set_ylabel("dice score")
    axes.set_xlabel("epochs")
    # 设置x轴刻度
    axes.set_xticks(tmp_x_axis_per_epoch)

    # 数据点注明具体数值
    axes.plot(tmp_x_axis_per_epoch,dice_scores_per_epoch,\
        linestyle='-', color='red', marker='8', linewidth=1.5,label='level 1')
    for x,y in zip(tmp_x_axis_per_epoch,dice_scores_per_epoch):
        axes.text(x,y,'%.4f'%y,ha='left',va='bottom')

    # 多条曲线标注
    axes.legend()
    if __name__ == '__main__':
        plt.show()
    fig.savefig(os.path.join(log_dir,'dice scores per epoch.jpg'), dpi=800)

    with open(os.path.join(log_dir,'dice scores per epoch.csv'),'w',newline='') as file:
        main_writer=csv.writer(file)
        main_writer.writerow(('epoch','dice score',))
        for i in range(0,len(dice_scores_per_epoch)):
            tmp_row=(i,dice_scores_per_epoch[i],)
            main_writer.writerow(tmp_row)

if __name__ == '__main__':
    dice_loss_per_batches=(1.5937461419539019,1.1031421422958374,0.5851144194602966,0.45622390508651733,0.2765046954154968,0.28186720609664917,0.24353008288325687,0.23502856492996216,0.22691103093551868,0.28186720609664917,0.2652295231819153,0.23502856492996216)
    bce_loss_per_batches=np.array(dice_loss_per_batches)
    bce_loss_per_batches= bce_loss_per_batches**2+1
    mse_loss_per_batches=np.cos(bce_loss_per_batches)

    dice_loss_per_epoch=dice_loss_per_batches[1::3]
    bce_loss_per_epoch=np.array(dice_loss_per_epoch)
    bce_loss_per_epoch=bce_loss_per_epoch**2+1
    mse_loss_per_epoch=np.cos(bce_loss_per_epoch)

    dice_scores=np.array(
        [54,57,67,76,23],
    )
    dice_scores=np.array(dice_scores)


    log_statistics('./tmp_log',4,dice_loss_per_batches,bce_loss_per_batches,None,dice_loss_per_epoch,bce_loss_per_epoch,mse_loss_per_epoch,dice_scores)

    # log_statistics('./tmp_log',4,dice_loss_per_batches,None,mse_loss_per_batches,dice_loss_per_epoch,bce_loss_per_epoch,None,None)