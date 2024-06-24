import os
import torch
from torchvision.datasets import CIFAR100
import pickle
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
from torchtext.transforms import CLIPTokenizer
from torch.nn.utils.rnn import pad_sequence

os.makedirs('./dataset',exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
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

class cifar100_dataset(Dataset):
    def __init__(self,path='./dataset/cifar-100-python/',mode='train',transform=transform):
        super().__init__()
        self.transform = transform
        self.path = path
        self.mode = mode
        self.train_file = self.unpickle(self.path+self.mode)
        self.classes = self.unpickle2(self.path+'meta')
        
        MERGES_FILE = "http://download.pytorch.org/models/text/clip_merges.bpe"
        ENCODER_FILE = "http://download.pytorch.org/models/text/clip_encoder.json"
        self.tokenizer = CLIPTokenizer(MERGES_FILE,ENCODER_FILE,)

    def unpickle(self,file):
        with open(file,'rb') as fo:
            dict = pickle.load(fo,encoding='bytes')
        return dict
    
    def unpickle2(self,file):
        with open(file,'rb') as infile:
            data = pickle.load(infile,encoding="latin1")
            classes = data['fine_label_names']
        return classes

    def text_input(self,text):
        try:
            text = list(set(sum(text,[])))
            text = [int(i) for i in text]
        except:
            text = [int(i) for i in text]
        return text

    
    def __getitem__(self, idx):
        image_data = self.train_file[b'data']
        image_data_reshape = np.reshape(image_data,(-1,32,32,3))

        image = Image.fromarray(image_data_reshape[idx,:,:,:])
        if transform is not None:
            image = self.transform(image)
        label = self.classes[self.train_file[b'coarse_labels'][idx]]
        
        text = self.text_input(self.tokenizer(self.classes))
        label = self.text_input(self.tokenizer(label))
        label_score = torch.zeros(len(text))
        label_index = [i for i, value in enumerate(text) if value in label]
        label_score[label_index]=1

        return image, text, label_score
    
    def __len__(self):
        return len(self.train_file[b'data'])