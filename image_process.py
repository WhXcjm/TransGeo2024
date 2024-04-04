import torch
from PIL import Image
import random
import numpy
import torchvision.transforms as transforms
find_index=[]
def input_transform(size):
    return transforms.Compose([
        transforms.Resize(size=tuple(size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
    ])




def split_and_sample(img, min_width=256,min_height=256):
    width, height = img.size
    blocks = []
    img_index=[]

    if width // 2 > min_width and height // 2 <= height:    
        blocks += [img.crop((0, 0, width // 2, height)), img.crop((width // 2, 0, width, height)), img.crop((width // 4, 0, 3 * width // 4, height))]
    elif width // 2 > min_width and height // 2 > height:
        blocks += [img.crop((0, 0, width // 2, height//2)), img.crop((width // 2, 0, width, height//2)), img.crop((width // 4, 0, 3 * width // 4, height//2)),
                   img.crop((0, height//2, width // 2, height)), img.crop((width // 2, height//2, width, height)), img.crop((width // 4, height//2, 3 * width // 4, height)),
                   img.crop((0, height//4, width // 2, 3*height//4)), img.crop((width // 2, height//4, width, 3*height//4)), img.crop((width // 4, height//4, 3 * width // 4, 3*height//4))]
    elif width // 2 <= min_width and height // 2 > height:
        blocks += [img.crop((0, 0, width, height//2)),
                   img.crop((0, height//2, width, height)),
                   img.crop((0, height//4, width, 3*height//4))]
    else:
        blocks = [img]

    for block in blocks:
        for i in range(10):
            bw, bh = block.size
            if bw > 256 and bh > 256:
                x = random.randint(0, bw - 256)
                y = random.randint(0, bh - 256)
                sample = block.crop((x, y, x + 256, y + 256))
                

 

class img_process():
    def __init__(mode,img):
        self.mode=mode
        self.img=img







def main():
    img = Image.open("C:/Users/xusir/Desktop/college/projects/Eyemap/CVPR_subset/CVPR_subset/bingmap/19/0000002.jpg").convert('RGB')
    sat_size = [256, 256]
    grd_size = [112, 616]
    whole_height=img.height
    whole_width=img.width
    split_and_sample(img)
    transform_reference = input_transform(size=sat_size)




if __name__ == '__main__':
    main()