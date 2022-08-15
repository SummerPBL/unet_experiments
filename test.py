import torch

from unet import U_net
from dice_loss import binary_dice_coeff

import numpy as np

from PIL import Image
import torchvision

from torch.utils.data import DataLoader

from dataset import liverDataset

import matplotlib.pyplot as plt


def effect_show(rawimg_path:str,label_path:str,model:U_net)->float:
    img=Image.open(rawimg_path).convert('L')
    label=Image.open(label_path).convert('1')

    trans_func=torchvision.transforms.ToTensor()
    img_arr, label_arr=trans_func(img), trans_func(label)

    input=img_arr.unsqueeze(0)
    # print(img_arr.size(),input.shape)
    if model.training==True:
        model.eval()
    with torch.no_grad():
        pred:torch.Tensor = model.forward(input)
    pred.round_()

    plt.subplot(1,3,1)
    plt.imshow(torch.squeeze(img_arr).numpy(),cmap='gray')
    plt.title('original image')

    plt.subplot(1,3,2)
    plt.imshow(torch.squeeze(label_arr).numpy(),cmap='gray')
    plt.title('ground truth')

    plt.subplot(1,3,3)
    plt.imshow(torch.squeeze(pred).detach().numpy(),cmap='gray')
    plt.title('prediction')

    plt.show()

    dice_grade:torch.Tensor =binary_dice_coeff(pred,label_arr.unsqueeze(0))
    return dice_grade.item()

if __name__ == '__main__':
    val_dataset=liverDataset('./dataset/val',None,None)

    val_loader=DataLoader(val_dataset,4,shuffle=True,num_workers=0)

    model=U_net(1,1,)
    # model.load_state_dict(torch.load('D:/microsoft_PBL/attention_unet/weights/attention_unet_10_10_0.04192398488521576',map_location='cpu'))

    model.load_state_dict(torch.load('D:/microsoft_PBL/unet_experiments/trained_models/unet_19_level4_9624.pth',map_location='cpu'))

    model.eval()

    # effect=validate(model,val_loader)

    # print('平均得分',effect)

    # exit()

    print(model.training)

    grade=effect_show('D:/microsoft_PBL/attention_unet/dataset/val/40_77.png','D:/microsoft_PBL/attention_unet/dataset/val/40_77_mask.png',model)

    print('dice score:%.3f'%grade)