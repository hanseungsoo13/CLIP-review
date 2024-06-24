from CLIP import CLIP
import torch
from torchinfo import summary
from Resnet50 import ModifiedResNet, BottleNeck
from transformer import Transformer
from torchtext.transforms import CLIPTokenizer
from torch.utils.data import DataLoader
from dataset import cifar100_dataset
from torch.optim import Adam
from torch import nn
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else 'cpu'

def collate_fn(batchDummy):
    x = torch.stack([torch.Tensor(batch[2]) for batch in batchDummy]).to(torch.int64).to(device)
    i = torch.stack([torch.Tensor(batch[0]) for batch in batchDummy]).to(device)
    t = torch.stack([torch.Tensor(batch[1]) for batch in batchDummy]).to(torch.int64).to(device)
    return i,t,x

model = CLIP(v_layers=[3,6,3,2],v_heads=8,v_input_size=224,v_emb_size=64,
      max_length=115,vocab_size=50000,n_layers=12,model_dim=2048, attn_heads=8).to(device)

#t=torch.Tensor(tokenizer("An image of a dog")).to(device)
#print(t)
#visual_model = ModifiedResNet(layers=[3,6,3,2],heads=8,input_size=224,emb_size=64).to(device) #(1,2048)
#text_model = Transformer(10,5000,8,64,8,model.build_attention_mask()).to(device)

#summary(model,[(5,3,224,224),(5,128)],dtypes=[torch.cuda.FloatTensor,torch.long],device=device)
#summary(visual_model,(1,3,224,224),dtypes=[torch.cuda.FloatTensor],device=device)
#summary(text_model,(1,10),dtypes=[torch.long],device=device)

#t = torch.rand([1,3,224,224]).to(device)

#out = model(t)
#print(out.shape)

config = {
    'batch_size':5,
    'epoch':1,
    'optimizer': Adam(model.parameters()),
    'loss': nn.CrossEntropyLoss()
    }
cifar100 = cifar100_dataset(mode='train')
train_loader = DataLoader(cifar100,batch_size=config['batch_size'],collate_fn=collate_fn)
optimizer = config['optimizer']
loss = config['loss']

for epoch in range(config['epoch']):
    mean_loss = []
    batch_loss = 0

    model.train()
    for id, (i, t, l) in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()
        l1,_ = model(i,t)
        batch_loss = loss(l1,l)
        batch_loss.backward()
        optimizer.step()

        mean_loss.append(batch_loss)

    print(f"Mean loss was {sum(mean_loss)/len(mean_loss)}")


        


