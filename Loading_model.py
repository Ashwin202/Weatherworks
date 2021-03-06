import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transform
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
import cv2 as cv
import PIL
from PIL import Image
import numpy as np

import config as C

       
def accuracy(out,labels):
    _,pred=torch.max(out,dim=1)
    return torch.tensor((torch.sum(pred==labels).item()/len(pred)))


class Weatherclass(nn.Module):
    def __init__(self):
        super().__init__()
        self.network=nn.Sequential(
            nn.Conv2d(3,32,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),#out: 64x16x16

            nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(128,128,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2), #out: 128x8x8
            
            nn.Conv2d(128,256,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2), #out: 256x4x4

            nn.Flatten(),
            nn.Linear(256*4*4,1024),
            nn.ReLU(),
            nn.Linear(1024,512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,4)
        )
    def forward(self,xb):
        return self.network(xb)


model = Weatherclass()
model.load_state_dict(torch.load(C.MODEL_PATH))
model.eval()


def predict_image(img, model):
    yb = model(img)
    _, preds = torch.max(yb, dim=1)    
    return C.OUT_CLASSES[preds]


        
def trial(pic_path):
    img=Image.open(pic_path)
    pil_to_tensor=transform.ToTensor()(img).unsqueeze_(0)    
    return predict_image(pil_to_tensor, model)


