import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import random

split_times=3
range_number=10

class LimitedFoV(object):
    def __init__(self, fov=360.):
        self.fov = fov

    def __call__(self, x):
        # print(x.shape)
        angle = random.randint(0, 359)
        rotate_index = int(angle / 360. * x.shape[2])
        fov_index = int(self.fov / 360. * x.shape[2])
        if rotate_index > 0:
            img_shift = torch.zeros(x.shape)
            img_shift[:,:, :rotate_index] = x[:,:, -rotate_index:]
            img_shift[:,:, rotate_index:] = x[:,:, :(x.shape[2] - rotate_index)]
        else:
            img_shift = x
        return img_shift[:,:,:fov_index]


def input_transform_fov(size, fov):
    return transforms.Compose([
        transforms.Resize(size=tuple(size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
        LimitedFoV(fov=fov),
    ])

def input_transform(size):
    return transforms.Compose([
        transforms.Resize(size=tuple(size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
    ])


# pytorch version of SATELLITE loader
class SATELLITE(torch.utils.data.Dataset):

    def __init__(self,
                 mode,
                 img,
                 print_bool,
                 args=None,
                 root='C:/Users/xusir/Desktop/college/projects/Eyemap/satellite_resources/XueyuanRoad/L21/XueyuanRoad.jpg'): 
        super(SATELLITE, self).__init__()

        self.args = args
        self.img=img
        self.root = root
        self.mode = mode
        self.sat_size = [256, 256]
        self.sat_size_default = [256, 256]
        self.img=img
        if args.sat_res != 0:
            self.sat_size = [args.sat_res, args.sat_res]

        if print_bool:
            print(self.sat_size)

        self.sat_ori_size = [750, 750]


        self.transform_reference = input_transform(size=self.sat_size)
        

        self.to_tensor = transforms.ToTensor()

        if print_bool:
            print('SATELLITE: load %s' % self.root)
       

    def __getitem__(self, index, debug=False):
        width, height = self.img.size
        blocks = []
        
        if self.mode=="w":    
            
            blocks += [self.img.crop((0, 0, width // 2, height)), self.img.crop((width // 2, 0, width, height)), self.img.crop((width // 4, 0, 3 * width // 4, height))]
        elif self.mode=="w&h":
            
            blocks += [self.img.crop((0, 0, width // 2, height//2)), self.img.crop((width // 2, 0, width, height//2)), self.img.crop((width // 4, 0, 3 * width // 4, height//2)),
                    self.img.crop((0, height//2, width // 2, height)), self.img.crop((width // 2, height//2, width, height)), self.img.crop((width // 4, height//2, 3 * width // 4, height)),
                    self.img.crop((0, height//4, width // 2, 3*height//4)), self.img.crop((width // 2, height//4, width, 3*height//4)), self.img.crop((width // 4, height//4, 3 * width // 4, 3*height//4))]
        elif self.mode=="h":
            
            blocks += [self.img.crop((0, 0, width, height//2)),
                    self.img.crop((0, height//2, width, height)),
                    self.img.crop((0, height//4, width, 3*height//4))]
        else:
            blocks = [self.img]


        

        # mode 1:split width;mode 2:split width and height;mode 3:split height
        if self.mode=="w":
            i=index//10
            block=blocks[i]
            bw, bh = block.size
            if bw > 256 and bh > 256:
                x = random.randint(0, bw - 256)
                y = random.randint(0, bh - 256)
                sample = block.crop((x, y, x + 256, y + 256))
            img_reference=sample
            img_reference=self.transform_reference(img_reference)
            return img_reference, torch.tensor(index), 0
        if self.mode=="w&h":
            i=index//10
            block=blocks[i]
            bw, bh = block.size
            if bw > 256 and bh > 256:
                x = random.randint(0, bw - 256)
                y = random.randint(0, bh - 256)
                sample = block.crop((x, y, x + 256, y + 256))
            img_reference=sample
            img_reference=self.transform_reference(img_reference)
            return img_reference, torch.tensor(index), 0
        if self.mode=="h":
            i=index//10
            block=blocks[i]
            bw, bh = block.size
            if bw > 256 and bh > 256:
                x = random.randint(0, bw - 256)
                y = random.randint(0, bh - 256)
                sample = block.crop((x, y, x + 256, y + 256))
            img_reference=sample
            img_reference=self.transform_reference(img_reference)
            return img_reference, torch.tensor(index), 0
        else:
            print('not implemented!!')
            raise Exception
        



    def __len__(self):
        if self.mode=="w":
            return split_times*range_number
        if self.mode=="w&h":
            return split_times*split_times*range_number
        if self.mode=="h":
            return split_times*range_number
        
      
       
