import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange,reduce,repeat
from collections import OrderedDict

class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1):
        super().__init__()

        self.residual_function = nn.Sequential([
            nn.Conv2d(in_channel, out_channel, 1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel,out_channel,3,padding=1,bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(stride) if stride >1 else nn.Identity(),
            nn.Conv2d(out_channel,out_channel*BottleNeck.expansion,1,bias=False),
            nn.BatchNorm2d(out_channel*BottleNeck.expansion),
        ])

        
        self.downsample = None
        self.stride = stride

        if stride>1 or out_channel != in_channel*BottleNeck.expansion:
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(in_channel, out_channel * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(out_channel * self.expansion)),
            ]))
        
        self.relu3 = nn.ReLU(inplace=True)
    def forward(self, x: torch.Tensor):
        identity = x
        out = self.residual_function(x)

        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu3(out)
        return out

class AttentionPool2d(nn.Module):
    def __init__(self,resnet_out:int,emb_dim:int,num_heads:int):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(resnet_out**2+1,emb_dim))
        self.k_proj = nn.Linear(emb_dim,emb_dim)
        self.q_proj = nn.Linear(emb_dim,emb_dim)
        self.v_proj = nn.Linear(emb_dim,emb_dim)
        self.c_proj = nn.Linear(emb_dim,emb_dim)
        self.num_heads = num_heads
    
    def forward(self, x):
        x = rearrange(x,'b c h w -> (h w) b c')
        x = torch.cat([x.mean(dim=0,keepdim=True),x],dim=0) #keep_dim: mean해도 차원이 줄어들지 않음
        x = x + self.positional_embedding(x)[:,None,:]
        x,_ = F.multi_head_attention_forward(
            query = x[:1],
            key = x,
            value = x,
            embed_dim_to_check = x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False            
        )
        return x.squeeze(0)

class ModifiedResNet(nn.Module):
    def __init__(self,layers,heads:int,input_size=224,emb_size=64):
        super.__init__()
        self.emb_size=emb_size
        self.input_size = input_size
        self.heads = heads

    #the 3-layer stem
        self.conv1 = nn.Conv2d(3,self.emb_size//2,kernel_size=3,stride=2,padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(self.emb_size // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(self.emb_size // 2, self.emb_size // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.emb_size // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(self.emb_size // 2, self.emb_size, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.emb_size)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        self.in_channels = self.emb_size
        self.layer1 = self.make_layer(64,layers[0])
        self.layer2 = self.make_layer(128,layers[1],stride=2)
        self.layer3 = self.make_layer(256,layers[2],stride=2)
        self.layer4 = self.make_layer(512,layers[3],stride=2)

        emb_dim = self.emb_size*32 # ResNet feature dimension
        self.attnpool = AttentionPool2d(resnet_out=7,emb_dim=emb_dim,num_heads = self.heads)

    def make_layer(self,out_channels,blocks, stride=1):
        layers = []

        for _ in range(blocks):
            layers.append(BottleNeck(self.in_channels,out_channels,stride))
            self.in_channels = out_channels*BottleNeck.expansion
        
        return nn.Sequential(*layers)
    
    def forward(self,x):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
        
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x

