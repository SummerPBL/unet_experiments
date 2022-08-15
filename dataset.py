from torch.utils.data import Dataset
import torch
from PIL import Image
import os
from typing import List,Tuple,Optional
import torchvision


class liverDataset(Dataset):
    def __init__(self, folder:str, raw_transform=Optional[torchvision.transforms.transforms.Compose] , label_transform=Optional[torchvision.transforms.transforms.Compose]) -> None:
        super().__init__()

        filenames=os.listdir(folder)

        self.pairs:List[Tuple(str)] =[]

        self.raw_transform = raw_transform
        if raw_transform==None:
            self.raw_transform=  torchvision.transforms.ToTensor()
            
        
        self.label_transform = label_transform
        if label_transform==None:
            self.label_transform= torchvision.transforms.ToTensor()

        for label in filenames:
            prefix=label[0:label.rfind('.')]
            if not prefix.endswith('mask'):
                continue
            # prefix=14_374_mask -> 14_374
            rawimg=prefix[0:prefix.rfind('_')]+'.png'
            self.pairs.append((
                os.path.join(folder,rawimg),
                os.path.join(folder,label),
            ))

    
    def __getitem__(self, index) -> Tuple[torch.Tensor]:
        img=Image.open(self.pairs[index][0]).convert('L')
        label=Image.open(self.pairs[index][1]).convert('1')
        return self.raw_transform(img), self.label_transform(label)

    def __len__(self)->int:
        return len(self.pairs)

if __name__ == '__main__':
    myDataset=liverDataset('./dataset/train',None,None)

    import torchvision
    t=torchvision.transforms.ToTensor()
    print(type(t))
    print(len(myDataset))

    x_transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    print(type(x_transforms))
    label:torch.Tensor
    img,label=myDataset[10]
    print(img.shape,label.shape)


