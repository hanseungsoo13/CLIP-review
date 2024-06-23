import os
import torch
from torchvision.datasets import CIFAR100
import pickle
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms



os.makedirs('./dataset',exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

cifar100 = CIFAR100(root = './dataset', download=True, train=True,transform=transform)

def unpickle(file):
    with open(file,'rb') as fo:
        dict = pickle.load(fo,encoding='bytes')
    return dict

with open('./dataset/cifar-100-python/meta','rb') as infile:
    data = pickle.load(infile,encoding="latin1")
    classes = data['fine_label_names']

os.makedirs('./dataset/train_image',exist_ok=True)
os.makedirs('./dataset/test_image',exist_ok=True)
for c in classes:
    os.makedirs('./dataset/train_image/{}'.format(c),exist_ok=True)
    os.makedirs('./dataset/test_image/{}'.format(c),exist_ok=True)

print('unpacking train file')

def data_save(image_path = './dataset/cifar-100-python/',
              save_path = './dataset/train_image/',
              mode='train'):
    train_file = unpickle(image_path+mode)
    train_data = train_file[b'data']
    
    train_data_reshape = np.reshape(train_data,(-1,32,32,3))
    print(train_data_reshape.shape)
    train_labels = train_file[b'fine_labels']
    train_filename = train_file[b'filenames']

    for idx in range(len(train_data)):
        label = train_labels[idx]
        image = Image.fromarray(train_data_reshape[idx])

        image.save(save_path+'{}/{}'.format(classes[label],train_filename[idx].decode('utf-8')))

data_save()